"""Determine the correct MatMul substitution for the two attention contractions.

The upstream `Einsum` calls reference an axis that does not exist in the tensor
they are given (see the notes in `ablate_attention.py`), so they cannot be used
directly as a specification. This probe instead enumerates every plausible
substitution and reports which ones produce a tensor that the NEXT operation in
the pipeline accepts - that is the real constraint, and it is checkable.

Run directly:

    python probe_einsum.py
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# This script lives OUTSIDE the deployment repository: it is analysis only and
# nothing in alike_lightglue_ONNX&RKNN_deploy/ imports it.  It needs that repository
# on sys.path for `model` and `paths`, and it writes its output here, next to
# itself, so the repository never depends on any report it produces.
_REPO = Path(__file__).resolve().parents[1] / "alike_lightglue_ONNX&RKNN_deploy"
if not _REPO.is_dir():
    raise SystemExit(f"cannot find the deployment repo at {_REPO}; edit _REPO at the top of this file")
sys.path.insert(0, str(_REPO))
HERE = Path(__file__).resolve().parent        # outputs/ - where reports go
import paths  # noqa: E402

CKPT = str(paths.TRAINED_CKPT)


def main():
    from model import load_lightglue_stage
    m = load_lightglue_stage(CKPT).eval().matcher
    cb = m.transformers[0].cross_attn

    B, N0, N1 = 1, 7, 5
    x0 = torch.randn(B, N0, 256)
    x1 = torch.randn(B, N1, 256)

    def heads(t):
        return t.unflatten(-1, (cb.heads, -1)).transpose(1, 2)

    qk0, qk1 = heads(cb.to_qk(x0)), heads(cb.to_qk(x1))
    v0, v1 = heads(cb.to_v(x0)), heads(cb.to_v(x1))
    print(f"B={B} N0={N0} N1={N1} heads={cb.heads}")
    for nm, t in (("qk0", qk0), ("qk1", qk1), ("v0", v0), ("v1", v1)):
        print(f"  {nm}: {tuple(t.shape)}")

    # `sim[i, j]` is the (query from 0, key from 1) score. It must be
    # (B, H, N0, N1) so that softmax(-1) normalises over the KEYS of image 1.
    sim_01 = torch.matmul(qk0, qk1.transpose(-2, -1))
    print(f"\nsim_01 = matmul(qk0, qk1.T): {tuple(sim_01.shape)}  (want {B},{cb.heads},{N0},{N1})")

    print("\n--- m0 candidates: einsum('bhij,bhjd->bhid', attn01, v1) ---")
    attn01 = F.softmax(sim_01, dim=-1)
    print(f"  attn01 {tuple(attn01.shape)}, v1 {tuple(v1.shape)}")
    for name, a in (("attn01", attn01), ("attn01.transpose(-2,-1)", attn01.transpose(-2, -1)),
                    ("attn01.transpose(-3,-1)", attn01.transpose(-3, -1))):
        for vn, v in (("v1", v1), ("v1.transpose(-2,-1)", v1.transpose(-2, -1))):
            try:
                r = torch.matmul(a, v)
                print(f"  OK   matmul({name}, {vn}) -> {tuple(r.shape)}")
            except RuntimeError as e:
                print(f"  fail matmul({name}, {vn}) -> {str(e)[:60]}")

    print("\n--- m1 candidates: the reverse direction, must consume v0 ---")
    # The reverse direction scores (query from 1, key from 0) = sim_01.T axes.
    sim_10 = torch.matmul(qk1, qk0.transpose(-2, -1))
    print(f"  sim_10 = matmul(qk1, qk0.T): {tuple(sim_10.shape)}")
    for label, s in (("sim_01.transpose(-2,-1)", sim_01.transpose(-2, -1)),
                     ("sim_10 (recomputed)", sim_10)):
        print(f"  -- {label}: {tuple(s.shape)}")
        attn = F.softmax(s, dim=-1)
        for name, a in (("attn", attn),
                        ("attn.transpose(-2,-1)", attn.transpose(-2, -1)),
                        ("attn.transpose(-3,-1)", attn.transpose(-3, -1))):
            try:
                r = torch.matmul(a, v0)
                ok = "OK  " if r.shape[-1] == qk0.shape[-1] else "BAD "
                print(f"     {ok} matmul({name}, v0) -> {tuple(r.shape)}"
                      + ("" if r.shape[-1] == qk0.shape[-1]
                         else f"   (expected last dim {qk0.shape[-1]})"))
            except RuntimeError as e:
                print(f"     fail matmul({name}, v0) -> {str(e)[:55]}")

    print("\n--- the contract the output must satisfy ---")
    print("  after flatten(-2), to_out expects last dim = "
          f"{cb.to_out.in_features} (= heads*head_dim = {cb.heads}*{qk0.shape[-1]})")


if __name__ == "__main__":
    main()
