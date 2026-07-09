"""Token-level trace dumper for speculative decoding research.

Enabled by setting the env var ``SPEC_TRACE_DIR`` to a writable directory
before launching the server. When unset (the default), every hook is a no-op
and the serving path is unchanged.

What is recorded (JSONL, one file per scheduler process):

* ``kind="meta"``  -- one record at first write: model / quantization /
  draft-model identity, so traces are self-describing.
* ``kind="q"``     -- the draft (EAGLE3) next-token distribution at the *root*
  of each draft tree: top-K probabilities over the vocab, keyed by request id
  and the output position it predicts. Captured from the eager draft-extend
  path, so trace runs must use ``--disable-cuda-graph``.
* ``kind="v"``     -- one record per request per verify step: the committed
  token ids, the target/verifier top-K logprobs at every committed position,
  the draft tree candidates, and which tree nodes were accepted.

These hooks live on the spec-v1 ``EAGLEWorker`` path, so trace runs must also
use ``--disable-overlap-schedule``.
"""

from __future__ import annotations

import json
import os
import threading
from typing import List, Optional

import torch

TRACE_TOPK = int(os.environ.get("SPEC_TRACE_TOPK", "32"))

_lock = threading.Lock()
_file = None
_meta_written = False


def enabled() -> bool:
    return bool(os.environ.get("SPEC_TRACE_DIR"))


def _get_file():
    global _file
    if _file is None:
        trace_dir = os.environ["SPEC_TRACE_DIR"]
        os.makedirs(trace_dir, exist_ok=True)
        path = os.path.join(trace_dir, f"trace_pid{os.getpid()}.jsonl")
        _file = open(path, "a", buffering=1024 * 1024)
    return _file


def _write(record: dict) -> None:
    global _meta_written
    with _lock:
        fh = _get_file()
        if not _meta_written:
            _meta_written = True
            meta = {"kind": "meta"}
            try:
                from sglang.srt.server_args import get_global_server_args

                args = get_global_server_args()
                meta.update(
                    model_path=args.model_path,
                    quantization=args.quantization,
                    draft_model_path=args.speculative_draft_model_path,
                    dtype=args.dtype,
                    trace_topk=TRACE_TOPK,
                )
            except Exception:
                pass
            fh.write(json.dumps(meta) + "\n")
        fh.write(json.dumps(record) + "\n")
        fh.flush()


def _round(values: List[float], ndigits: int = 5) -> List[float]:
    return [round(v, ndigits) for v in values]


def trace_draft_q(
    reqs,
    next_token_logits: torch.Tensor,
    out_pos_offset: int,
    hot_token_id: Optional[torch.Tensor] = None,
) -> None:
    """Record the draft model's root next-token distribution.

    Args:
        reqs: ``batch.reqs`` aligned with the logits rows.
        next_token_logits: (num_reqs, vocab) draft logits (one row per req).
        out_pos_offset: added to ``len(req.output_ids)`` to get the output
            position this distribution predicts. 0 for draft-extend after
            decode (output_ids already includes the committed tokens); +1 for
            the draft extend after prefill (the first generated token is not
            yet in output_ids and the root predicts the position after it).
        hot_token_id: EAGLE3 draft-vocab -> target-vocab id map (the model's
            ``d2t``-derived ``hot_token_id``). The draft logits are over the
            reduced draft vocab; without this remap the recorded ids would be
            draft-vocab indices, meaningless in target space.
    """
    if next_token_logits.shape[0] != len(reqs):
        # Row/req misalignment (e.g. a request finished this step and the
        # extend batch was filtered). Skip rather than risk a bad join.
        return
    probs = torch.softmax(next_token_logits.float(), dim=-1)
    k = min(TRACE_TOPK, probs.shape[-1])
    top_p, top_i = torch.topk(probs, k, dim=-1)
    if hot_token_id is not None:
        top_i = hot_token_id.to(top_i.device)[top_i]
    top_p = top_p.cpu().tolist()
    top_i = top_i.cpu().tolist()
    for i, req in enumerate(reqs):
        _write(
            {
                "kind": "q",
                "rid": req.rid,
                "out_pos": len(req.output_ids) + out_pos_offset,
                "q_top_ids": top_i[i],
                "q_top_probs": _round(top_p[i]),
            }
        )


def trace_verify_step(batch, spec_info, res, accepted_logits: torch.Tensor) -> None:
    """Record one verify step: committed tokens + target top-K per position.

    Args:
        batch: the ScheduleBatch (``req.output_ids`` already includes the
            tokens committed by this verify step).
        spec_info: the EagleVerifyInput (draft tree candidates).
        res: EagleVerifyOutput.
        accepted_logits: target logits already filtered to accepted rows,
            i.e. ``logits_output.next_token_logits`` after the
            ``res.accept_indices`` slice in ``EAGLEWorker.verify``.
    """
    counts = [n + 1 for n in res.num_correct_drafts_per_req_cpu]
    if len(counts) != len(batch.reqs) or sum(counts) != accepted_logits.shape[0]:
        return

    logprobs = torch.log_softmax(accepted_logits.float(), dim=-1)
    k = min(TRACE_TOPK, logprobs.shape[-1])
    top_lp, top_i = torch.topk(logprobs, k, dim=-1)
    top_lp = top_lp.cpu().tolist()
    top_i = top_i.cpu().tolist()

    commit_ids = res.accept_tokens.cpu().tolist()
    accept_flat = res.accept_indices.cpu().tolist()
    dtn = spec_info.draft_token_num
    candidates = spec_info.draft_token.reshape(len(batch.reqs), dtn).cpu().tolist()

    offset = 0
    for i, req in enumerate(batch.reqs):
        c = counts[i]
        _write(
            {
                "kind": "v",
                "rid": req.rid,
                "out_len_after": len(req.output_ids),
                "n_commit": c,
                "commit_ids": commit_ids[offset : offset + c],
                "tgt_top_ids": top_i[offset : offset + c],
                "tgt_top_logprobs": [
                    _round(row) for row in top_lp[offset : offset + c]
                ],
                "cand_ids": candidates[i],
                "accept_rows": [a - i * dtn for a in accept_flat[offset : offset + c]],
            }
        )
        offset += c
