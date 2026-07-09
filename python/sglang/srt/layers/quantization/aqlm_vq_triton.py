# SPDX-License-Identifier: Apache-2.0
# Fused Triton GEMM for AQLM additive-VQ weights (codebook-centric,
# EVA/CodeGEMM style): never materializes the FP16 weight matrix.
#
#   y[t, o] = scales[o] * sum_g sum_c dot(codebooks[c, codes[o, g, c]],
#                                         x[t, g*ig:(g+1)*ig])
#
# Two-stage psumbook formulation (EVA insight: input-codebook computation
# turns GEMV into GEMM):
#   stage 1 (small GEMM): P[c, t, g, e] = dot(codebooks[c, e], x[t, g])
#                         for all codebook entries e -- computed with
#                         torch.einsum (tensor cores), tiny FLOPs.
#   stage 2 (gather-add): y[t, o] = sum_g sum_c P[c, t, g, codes[o, g, c]]
#                         fused Triton kernel; weight traffic = codes only
#                         (num_codebooks bytes per in_group of 8 weights =
#                         1 B/weight for 8x8, vs 2 B/weight FP16).
#
# For small verify/decode batches the psumbook P fits in L2 and the kernel
# is bounded by codes traffic, i.e. ~1/2 the FP16 GEMM traffic at 8 books
# and proportionally less with fewer (progressive) codebooks.

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_add_kernel(
    p_ptr,        # [num_books, T, G, 256] fp16 psumbook
    codes_ptr,    # [O, G, num_books] uint8
    scales_ptr,   # [O] fp16
    y_ptr,        # [T, O] fp16
    T, O, G,
    NUM_BOOKS: tl.constexpr,
    stride_pc, stride_pt, stride_pg,
    stride_co, stride_cg,
    stride_yt, stride_yo,
    BLOCK_T: tl.constexpr, BLOCK_O: tl.constexpr, BLOCK_G: tl.constexpr,
):
    pid_o = tl.program_id(0)
    pid_t = tl.program_id(1)
    offs_o = pid_o * BLOCK_O + tl.arange(0, BLOCK_O)
    offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
    mask_o = offs_o < O
    mask_t = offs_t < T

    acc = tl.zeros((BLOCK_T, BLOCK_O), dtype=tl.float32)
    for g0 in range(0, G, BLOCK_G):
        offs_g = g0 + tl.arange(0, BLOCK_G)
        mask_g = offs_g < G
        for c in tl.static_range(NUM_BOOKS):
            # codes tile: (BLOCK_O, BLOCK_G)
            code = tl.load(
                codes_ptr + offs_o[:, None] * stride_co
                + offs_g[None, :] * stride_cg + c,
                mask=mask_o[:, None] & mask_g[None, :], other=0,
            ).to(tl.int32)
            # psum gather: (BLOCK_T, BLOCK_O, BLOCK_G)
            p = tl.load(
                p_ptr + c * stride_pc
                + offs_t[:, None, None] * stride_pt
                + offs_g[None, None, :] * stride_pg
                + code[None, :, :],
                mask=mask_t[:, None, None]
                & (mask_o[:, None] & mask_g[None, :])[None, :, :],
                other=0.0,
            )
            acc += tl.sum(p.to(tl.float32), axis=2)

    scale = tl.load(scales_ptr + offs_o, mask=mask_o, other=0.0)
    acc = acc * scale[None, :].to(tl.float32)
    tl.store(
        y_ptr + offs_t[:, None] * stride_yt + offs_o[None, :] * stride_yo,
        acc.to(tl.float16),
        mask=mask_t[:, None] & mask_o[None, :],
    )


def aqlm_psum_gemm(
    x: torch.Tensor,          # [T, in_features] fp16
    codes_u8: torch.Tensor,   # [O, G, num_books] uint8 (unpacked)
    codebooks: torch.Tensor,  # [num_books, 256, 1, in_group] fp16
    scales: torch.Tensor,     # [O] fp16
    block_t: int = 16,
    block_o: int = 64,
    block_g: int = 16,
) -> torch.Tensor:
    T, in_features = x.shape
    O, G, num_books = codes_u8.shape
    ig = codebooks.shape[-1]
    assert G * ig == in_features

    # stage 1: psumbook  [num_books, T, G, 256]
    xg = x.reshape(T, G, ig)
    cb = codebooks.reshape(num_books, -1, ig)
    p = torch.einsum("tgi,cei->ctge", xg, cb).contiguous()

    y = torch.empty(T, O, dtype=torch.float16, device=x.device)
    grid = (triton.cdiv(O, block_o), triton.cdiv(T, block_t))
    _gather_add_kernel[grid](
        p, codes_u8, scales, y,
        T, O, G,
        NUM_BOOKS=num_books,
        stride_pc=p.stride(0), stride_pt=p.stride(1), stride_pg=p.stride(2),
        stride_co=codes_u8.stride(0), stride_cg=codes_u8.stride(1),
        stride_yt=y.stride(0), stride_yo=y.stride(1),
        BLOCK_T=block_t, BLOCK_O=block_o, BLOCK_G=block_g,
    )
    return y


def prepare_triton_layer(layer, quant_config, num_codebooks_used: Optional[int]):
    """Repack loaded AQLM tensors for the fused kernel (called once)."""
    k = num_codebooks_used or quant_config.num_codebooks
    codes = layer.codes.data.to(torch.int16) % 256   # unpack int8 -> [0,256)
    layer.vq_codes_u8 = codes[..., :k].to(torch.uint8).contiguous()
    # codebooks of merged partitions are concatenated along dim 0 as
    # [num_partitions * num_codebooks, ...]; regroup to per-partition list
    parts = getattr(layer.codebooks, "output_partition_sizes")
    ncb = quant_config.num_codebooks
    cb = layer.codebooks.data
    layer.vq_codebooks = [
        cb[i * ncb : i * ncb + k].contiguous() for i in range(len(parts))
    ]
    layer.vq_partition_sizes = list(parts)
    layer.vq_scales = layer.scales.data.reshape(-1).contiguous()
    layer.codes = None
    layer.codebooks = None
    layer.scales = None


def triton_gemm(layer, x: torch.Tensor, bias: Optional[torch.Tensor]):
    xf = x.reshape(-1, x.shape[-1]).contiguous()
    outs = []
    off = 0
    for size, cb in zip(layer.vq_partition_sizes, layer.vq_codebooks):
        outs.append(
            aqlm_psum_gemm(
                xf,
                layer.vq_codes_u8.narrow(0, off, size),
                cb,
                layer.vq_scales.narrow(0, off, size),
            )
        )
        off += size
    y = torch.cat(outs, dim=1) if len(outs) > 1 else outs[0]
    if bias is not None:
        y += bias
    return y.reshape(*x.shape[:-1], y.shape[-1])
