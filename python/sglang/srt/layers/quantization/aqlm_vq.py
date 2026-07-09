# SPDX-License-Identifier: Apache-2.0
# AQLM / additive vector-quantization support for SGLang.
#
# Checkpoint format (HF `quant_method: "aqlm"`, e.g.
# dbw6/Llama-3.1-8B-Instruct-AQLM-8bit-8x8):
#   <linear>.codes     int8  [out_features, in_features // in_group_size,
#                             num_codebooks]     (8-bit codes stored as int8)
#   <linear>.codebooks fp16  [num_codebooks, 2**nbits, out_group_size,
#                             in_group_size]
#   <linear>.scales    fp16  [out_features // out_group_size, 1, 1, 1]
#
#   W[o, g*ig:(g+1)*ig] = scales[o] * sum_c codebooks[c, codes[o, g, c]]
#
# Adapted from vLLM's (since removed) aqlm.py, updated for current SGLang
# parameter/loader interfaces. Adds two research knobs (env vars):
#
#   SGLANG_AQLM_VQ_MODE
#       cache    (default) dequantize once after loading, keep an FP16
#                weight resident; fastest, no memory saving. Correctness
#                reference for the other modes.
#       dequant  dequantize on every forward (embedding_bag); true VQ memory
#                footprint, slow. Torch fallback path.
#       triton   fused code-gather GEMM (no FP16 weight materialization).
#   SGLANG_AQLM_VQ_CODEBOOKS
#       int in [1, num_codebooks]: use only the first k codebooks
#       (progressive / any-precision prefix). Default: all.

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.quantization.base_config import (
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.utils import set_weight_attrs

logger = logging.getLogger(__name__)


def _mode() -> str:
    return os.environ.get("SGLANG_AQLM_VQ_MODE", "cache")


def _num_codebooks_override() -> Optional[int]:
    v = os.environ.get("SGLANG_AQLM_VQ_CODEBOOKS")
    return int(v) if v else None


def get_int_dtype(nbits: int) -> torch.dtype:
    if nbits <= 8:
        return torch.int8
    if nbits <= 16:
        return torch.int16
    if nbits <= 32:
        return torch.int32
    raise ValueError(f"No dtype available for {nbits}-bit codebooks")


@torch.inference_mode()
def unpack_int_data(data: torch.Tensor, nbits: int) -> torch.Tensor:
    return data.to(torch.int64) % (2**nbits)


@torch.inference_mode()
def dequantize_weight(
    codes: torch.Tensor,        # [out, in_groups, num_codebooks] int64 (unpacked)
    codebooks: torch.Tensor,    # [num_codebooks, size, out_group, in_group]
    scales: Optional[torch.Tensor] = None,  # [out_groups, 1, 1, 1]
    num_codebooks_used: Optional[int] = None,
) -> torch.Tensor:
    """Reconstruct FP16 weight; optionally only the first k codebooks."""
    num_codebooks, codebook_size, out_group_size, in_group_size = codebooks.shape
    if num_codebooks_used is not None and num_codebooks_used < num_codebooks:
        codes = codes[..., :num_codebooks_used]
        codebooks = codebooks[:num_codebooks_used]
        num_codebooks = num_codebooks_used
    num_out_groups, num_in_groups, _ = codes.shape[-3:]
    out_features = num_out_groups * out_group_size
    in_features = num_in_groups * in_group_size
    codebook_offsets = torch.arange(
        0, num_codebooks * codebook_size, codebook_size, device=codes.device
    )
    reconstructed = F.embedding_bag(
        codes.flatten(0, -2) + codebook_offsets,
        codebooks.flatten(0, 1).flatten(-2, -1),
        mode="sum",
    )
    reconstructed = reconstructed.view(
        list(codes.shape[:-3])
        + [num_out_groups, num_in_groups, out_group_size, in_group_size]
    )
    if scales is not None:
        reconstructed = reconstructed.mul(scales)
    return reconstructed.swapaxes(-3, -2).reshape(
        list(codes.shape[:-3]) + [out_features, in_features]
    )


def dequantize_partitioned(
    codes: torch.Tensor,
    codebooks: torch.Tensor,
    scales: torch.Tensor,
    output_partition_sizes: List[int],
    nbits: int,
    num_codebooks_per_partition: int,
    num_codebooks_used: Optional[int] = None,
) -> torch.Tensor:
    """Dequantize a (possibly merged QKV / gate_up) layer's weight.

    codebooks of the logical partitions are concatenated along dim 0.
    """
    unpacked = unpack_int_data(codes, nbits)
    weights = []
    off_out, off_cb = 0, 0
    for size in output_partition_sizes:
        weights.append(
            dequantize_weight(
                unpacked.narrow(0, off_out, size),
                codebooks.narrow(0, off_cb, num_codebooks_per_partition),
                scales.narrow(0, off_out, size),
                num_codebooks_used,
            )
        )
        off_out += size
        off_cb += num_codebooks_per_partition
    return torch.cat(weights, dim=0)


class AqlmVqConfig(QuantizationConfig):
    """Config class for AQLM-format additive VQ checkpoints."""

    def __init__(
        self,
        in_group_size: int,
        nbits_per_codebook: int,
        num_codebooks: int,
        out_group_size: int,
        modules_to_not_convert: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.in_group_size = in_group_size
        self.nbits_per_codebook = nbits_per_codebook
        self.num_codebooks = num_codebooks
        self.out_group_size = out_group_size
        self.modules_to_not_convert = modules_to_not_convert or []
        assert self.out_group_size == 1, "out_group_size > 1 not supported"
        self.pack_factor = self.in_group_size * self.out_group_size

    def __repr__(self) -> str:
        return (
            f"AqlmVqConfig(in_group={self.in_group_size}, "
            f"nbits={self.nbits_per_codebook}, "
            f"num_codebooks={self.num_codebooks})"
        )

    @classmethod
    def get_name(cls) -> str:
        return "aqlm_vq"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.half]

    @classmethod
    def get_min_capability(cls) -> int:
        return 60

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return []

    def get_scaled_act_names(self) -> List[str]:
        return []

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg, user_quant) -> Optional[str]:
        # Claim HF checkpoints with quant_method == "aqlm" when the user asks
        # for aqlm_vq (or does not specify anything).
        if hf_quant_cfg is None:
            return None
        if hf_quant_cfg.get("quant_method", "").lower() == "aqlm" and user_quant in (
            None,
            "aqlm_vq",
        ):
            return "aqlm_vq"
        return None

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AqlmVqConfig":
        in_group_size = cls.get_from_keys(config, ["in_group_size"])
        nbits_per_codebook = cls.get_from_keys(config, ["nbits_per_codebook"])
        num_codebooks = cls.get_from_keys(config, ["num_codebooks"])
        out_group_size = cls.get_from_keys(config, ["out_group_size"])
        modules_to_not_convert = cls.get_from_keys_or(
            config, ["linear_weights_not_to_quantize", "modules_to_not_convert"], None
        )
        return cls(
            in_group_size,
            nbits_per_codebook,
            num_codebooks,
            out_group_size,
            modules_to_not_convert,
        )

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, LinearBase):
            # lm_head is a ParallelLMHead (not LinearBase); embeddings and
            # norms never reach here. linear_weights_not_to_quantize lists
            # norm/embed/lm_head weights only in this checkpoint family, but
            # honor it for generality (entries end with ".weight").
            stripped = [
                m[: -len(".weight")] if m.endswith(".weight") else m
                for m in self.modules_to_not_convert
            ]
            if any(prefix.startswith(m) or m.endswith(prefix) for m in stripped):
                return UnquantizedLinearMethod()
            return AqlmVqLinearMethod(self)
        return None


class AqlmVqLinearMethod(LinearMethodBase):
    """Linear method for AQLM-format additive VQ weights."""

    def __init__(self, quant_config: AqlmVqConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        del input_size, output_size
        if params_dtype != torch.half:
            raise ValueError("aqlm_vq only supports float16 activations")
        if input_size_per_partition % self.quant_config.in_group_size != 0:
            raise ValueError("input size not aligned with in_group_size")
        output_size_per_partition = sum(output_partition_sizes)
        if output_size_per_partition % self.quant_config.out_group_size != 0:
            raise ValueError("output size not aligned with out_group_size")

        codes = Parameter(
            torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.quant_config.pack_factor,
                self.quant_config.num_codebooks,
                dtype=get_int_dtype(self.quant_config.nbits_per_codebook),
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            codes,
            {
                "input_dim": 1,
                "output_dim": 0,
                "packed_dim": 1,
                "pack_factor": self.quant_config.pack_factor,
            },
        )

        codebooks = Parameter(
            torch.empty(
                self.quant_config.num_codebooks * len(output_partition_sizes),
                2**self.quant_config.nbits_per_codebook,
                self.quant_config.out_group_size,
                self.quant_config.in_group_size,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            codebooks,
            {
                # fixed-size shards concatenated along dim 0 (one per logical
                # partition of a merged layer)
                "is_metadata": True,
                "output_partition_sizes": output_partition_sizes,
            },
        )

        scales = Parameter(
            torch.empty(
                output_size_per_partition // self.quant_config.out_group_size,
                1,
                1,
                1,
                dtype=params_dtype,
            ),
            requires_grad=False,
        )
        set_weight_attrs(
            scales,
            {
                "output_dim": 0,
                "packed_dim": 0,
                "pack_factor": self.quant_config.out_group_size,
            },
        )

        layer.register_parameter("codes", codes)
        set_weight_attrs(codes, extra_weight_attrs)
        layer.register_parameter("codebooks", codebooks)
        set_weight_attrs(codebooks, extra_weight_attrs)
        layer.register_parameter("scales", scales)
        set_weight_attrs(scales, extra_weight_attrs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        mode = _mode()
        k = _num_codebooks_override()
        if k is not None:
            logger.info(
                "aqlm_vq: progressive prefix with %d/%d codebooks",
                k,
                self.quant_config.num_codebooks,
            )
        if mode == "cache":
            weight = dequantize_partitioned(
                layer.codes.data,
                layer.codebooks.data,
                layer.scales.data,
                getattr(layer.codebooks, "output_partition_sizes"),
                self.quant_config.nbits_per_codebook,
                self.quant_config.num_codebooks,
                k,
            )
            layer.register_parameter(
                "weight", Parameter(weight, requires_grad=False)
            )
            # free VQ tensors; keep attribute names valid
            layer.codes = None
            layer.codebooks = None
            layer.scales = None
        elif mode == "triton":
            from sglang.srt.layers.quantization.aqlm_vq_triton import (
                prepare_triton_layer,
            )

            prepare_triton_layer(layer, self.quant_config, k)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mode = _mode()
        if mode == "cache":
            return F.linear(x, layer.weight, bias)
        if mode == "triton":
            from sglang.srt.layers.quantization.aqlm_vq_triton import (
                triton_gemm,
            )

            return triton_gemm(layer, x, bias)
        # mode == "dequant": torch fallback, dequantize per call
        weight = dequantize_partitioned(
            layer.codes.data,
            layer.codebooks.data,
            layer.scales.data,
            getattr(layer.codebooks, "output_partition_sizes"),
            self.quant_config.nbits_per_codebook,
            self.quant_config.num_codebooks,
            _num_codebooks_override(),
        )
        return F.linear(x, weight, bias)
