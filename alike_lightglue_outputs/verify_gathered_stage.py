#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Does the WIRED-IN stage still compute what the dense stage computed?

`probe_head_restructure.py` verifies the arithmetic in isolation, on tensors it
assembles itself.  That is not the same claim as "the module now in
`model/alike_stage.py` produces the same output", and the gap between the two
has been where this project's bugs have lived: an ablation that reported a
configuration it never applied, a checkpoint whose parameter count agreed for the
wrong reason, a probe that measured the wrong sampling mode.

So this drives BOTH stages through the real `forward` - the same entry point the
export uses - and compares the three outputs the deployment consumes.

Why the comparison is on keypoints AND descriptors AND scores:
  * keypoints and scores come from the score map.  The restructure leaves the score
    channel dense and untouched, so they SHOULD be bit-identical, and any difference
    means the trunk was not actually shared.
  * the descriptor is the thing that changed, and its unit is cosine because the
    matcher consumes it by angle.

Usage
    python verify_gathered_stage.py --checkpoint <ckpt>
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Analysis-only script, kept OUTSIDE the deployment repository: nothing in
# alike_lightglue_ONNX&RKNN_deploy/ imports or reads any output from it.  It needs that
# repository on sys.path for `model` and `paths`, and writes its reports HERE.
_REPO = Path(__file__).resolve().parents[1] / "alike_lightglue_ONNX&RKNN_deploy"
if not _REPO.is_dir():
    raise SystemExit(f"cannot find the deployment repo at {_REPO}; edit _REPO here")
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))
HERE = Path(__file__).resolve().parent        # outputs/ - where reports go
import paths  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default=str(paths.TRAINED_CKPT))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--keypoints", type=int, default=512)
    ap.add_argument("--batches", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from model import load_alike_stage

    dense = load_alike_stage(args.checkpoint, top_k=args.keypoints,
                             gathered_head=False).eval()
    new = load_alike_stage(args.checkpoint, top_k=args.keypoints,
                           gathered_head=True).eval()

    n_dense = sum(p.numel() for p in dense.parameters())
    n_new = sum(p.numel() for p in new.parameters())
    print("=== wired-in stage vs dense stage ===")
    print(f"  dense  params: {n_dense:,}")
    print(f"  new    params: {n_new:,}   (delta {n_new - n_dense:+,} - the head weight")
    print("                             is referenced, not copied, so this is 0)")
    print(f"  dense module : {type(dense.desc_sample).__name__}")
    print(f"  new   module : {type(new.desc_sample).__name__}")
    print()

    rows = []
    for i in range(args.batches):
        torch.manual_seed(1000 + i)
        img = torch.rand(2, 3, args.size, args.size)
        with torch.no_grad():
            k_d, d_d, s_d = dense(img)
            k_n, d_n, s_n = new(img)

        kp = float((k_d - k_n).abs().max())
        sc = float((s_d - s_n).abs().max())
        cos = F.cosine_similarity(d_d, d_n, dim=-1)          # (B, K)
        rows.append({"batch": i, "kp_max_abs": kp, "score_max_abs": sc,
                     "desc_cos_median": float(cos.median()),
                     "desc_cos_min": float(cos.min()),
                     "desc_max_abs": float((d_d - d_n).abs().max())})
        print(f"  batch {i}: keypoints max|Δ| {kp:.3e}   scores max|Δ| {sc:.3e}"
              f"   desc cos med {cos.median():.7f} min {cos.min():.7f}")

    worst_kp = max(r["kp_max_abs"] for r in rows)
    worst_sc = max(r["score_max_abs"] for r in rows)
    worst_cos = min(r["desc_cos_min"] for r in rows)
    print()
    print("=== verdict ===")
    print(f"  keypoints    max|Δ| = {worst_kp:.3e}   "
          f"{'BIT-IDENTICAL' if worst_kp == 0 else 'DIFFERS'}")
    print(f"  scores       max|Δ| = {worst_sc:.3e}   "
          f"{'BIT-IDENTICAL' if worst_sc == 0 else 'DIFFERS'}")
    print(f"  descriptor   cos min = {worst_cos:.7f}")
    ok = worst_kp == 0.0 and worst_sc == 0.0 and worst_cos > 1 - 1e-6
    print()
    if ok:
        print("  PASS.  The score path is bit-identical - so the trunk really is")
        print("  shared rather than re-derived - and the descriptor is identical to")
        print("  float32 associativity.  For reference the fp16 NPU conversion alone")
        print("  costs descriptor cosine 0.9997, four orders of magnitude more.")
    else:
        print("  FAIL.  Do not export this; the wired-in stage is not the one that")
        print("  was verified.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"batches": rows, "desc_sample_dense": type(dense.desc_sample).__name__,
             "desc_sample_new": type(new.desc_sample).__name__,
             "params_dense": n_dense, "params_new": n_new,
             "pass": bool(ok)}, indent=2))
        print(f"\n[verify] -> {args.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
