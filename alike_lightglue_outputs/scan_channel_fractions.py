#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Fractional channel pruning of the ALIKE backbone, measured on BOTH outputs.

WHY THE WHOLE-BLOCK SCAN WAS NOT ENOUGH
---------------------------------------
`scan_alike_channels.py` zeroes each block entirely and every block came out above
tolerance (best 0.177 against a 0.02 line).  That answers "can this block be
removed" and NOT "can 25 % of its channels be removed" - the two are different
questions with different answers, and only the second one is what channel pruning
asks.

WHY THE TWO OUTPUTS ARE MEASURED SEPARATELY, WHICH IS THE WHOLE POINT HERE
-------------------------------------------------------------------------
`ALikeNet.forward` splits the head as

    descriptor_map = x[:, :-1]                            # channels 0 .. 127
    scores_map     = sigmoid(x[:, -1]).unsqueeze(1)       # channel 128

The SCORE is the LAST channel and the DESCRIPTOR is the first 128.  So pruning
descriptor channels leaves the score map bit-identical, and a sensitivity scan that
only watches the score map would report **zero damage while destroying the
descriptor** - which is the quantity the matcher actually consumes.  Every number
below is therefore reported twice: once for the score map and once for the
descriptor map, and a block is only "safe" if it is safe on both.

This is the same trap as the `heads 4->2` ablation in a new place: a measurement
that watches the wrong output reports success.

WHY DESCRIPTORS ARE MEASURED AS COSINE RATHER THAN L1
-----------------------------------------------------
The descriptor map is L2-normalised before it is sampled, and the matcher's
decision is driven by the ANGLE between descriptors.  An L1 change in the raw map
is not comparable between channels with different magnitudes; cosine is the unit
the downstream consumer uses.  Both are printed for the score map's sake - it is
consumed as a raw value, so L1 is the right unit there.

Usage
    python scan_channel_fractions.py --checkpoint <ckpt> \
        --out reports/alike_channel_fractions.json
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

#: Blocks to test, highest MAC share first (from `probe_flops_split.py`).
#: `convhead2` leads at 24.0 % of the entire pipeline.
BLOCKS = [
    "net.convhead2",        # 4.3G MACs, 24.0 % of the pipeline
    "net.block1.conv2",     # 604M
    "net.block2.conv2",     # 604M
    "net.block2.conv1",     # 302M
    "net.block3.conv2",     # 151M
    "net.conv1",            # 134M
    "net.block1.conv1",     # 113M
    "net.block4.conv2",     # 37.7M
]

KEEP_RATIOS = [0.9, 0.75, 0.5]


@torch.no_grad()
def outputs(stage, img):
    s, d = stage.dense_maps(img)
    return s, d


@torch.no_grad()
def measure(stage, img, base_s, base_d, block, keep_ratio):
    """Zero the least-important OUTPUT channels of `block`, measure both outputs.

    Zeroing rather than physically removing: shapes stay valid so one measurement
    is one forward pass and no surgery.  The error is measured on the OUTPUTS, so
    whether the channel was later squeezed out is irrelevant to the question asked.
    """
    mod = dict(stage.named_modules())[block]
    w = mod.weight.data
    imp = w.abs().sum(dim=(1, 2, 3))                 # (out_channels,)
    n_keep = max(1, int(round(imp.numel() * keep_ratio)))
    if n_keep >= imp.numel():
        return None
    keep = imp.topk(n_keep).indices
    mask = torch.zeros_like(imp, dtype=torch.bool)
    mask[keep] = True

    saved_w, saved_b = w.clone(), (mod.bias.data.clone()
                                   if mod.bias is not None else None)
    w[~mask] = 0.0
    if mod.bias is not None:
        mod.bias.data[~mask] = 0.0
    try:
        s, d = outputs(stage, img)
    finally:
        w.copy_(saved_w)
        if saved_b is not None:
            mod.bias.data.copy_(saved_b)

    # score map: consumed as a raw value, so relative L1 is the right unit
    s_l1 = float((s - base_s).abs().mean() / base_s.abs().mean().clamp(min=1e-9))
    # descriptor: consumed by ANGLE, so cosine is the right unit.  Measured
    # per-channel-pixel and then averaged, matching how the matcher sees it.
    bd = F.normalize(base_d, p=2, dim=1)
    dd = F.normalize(d, p=2, dim=1)
    cos = (bd * dd).sum(dim=1)                       # (B, H, W)
    return {"score_relL1": s_l1, "desc_cos_mean": float(cos.mean()),
            "desc_cos_min": float(cos.min()), "channels_kept": n_keep,
            "channels_total": int(imp.numel())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default=str(paths.TRAINED_CKPT))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--keypoints", type=int, default=512)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from model import load_alike_stage

    # gathered_head=False: the scan measures the DENSE maps, which the gathered
    # head never materialises (dense_maps raises under the default).
    stage = load_alike_stage(args.checkpoint, top_k=args.keypoints,
                             gathered_head=False).eval()
    have = {n for n, _ in stage.named_modules()}
    missing = [b for b in BLOCKS if b not in have]
    if missing:
        raise SystemExit(f"block names not found: {missing}")

    torch.manual_seed(0)
    img = torch.rand(1, 3, args.size, args.size)
    base_s, base_d = outputs(stage, img)

    print(f"=== fractional OUTPUT-channel pruning, {args.size}x{args.size} ===")
    print("  score map measured as relative L1; descriptor as cosine "
          "(it is L2-normalised before use)")
    print("  reference: the deployed fp16 pipeline sits at descriptor cos 0.9997,")
    print("  so anything below ~0.999 is worse than the whole NPU conversion.")
    print()

    rows = []
    for block in BLOCKS:
        mod = dict(stage.named_modules())[block]
        n_out = mod.weight.shape[0]
        print(f"--- {block}  ({n_out} output channels)")
        for kr in KEEP_RATIOS:
            r = measure(stage, img, base_s, base_d, block, kr)
            if r is None:
                continue
            r.update({"block": block, "keep_ratio": kr, "n_out": n_out})
            rows.append(r)
            print(f"      keep {kr:>4.2f}  ({r['channels_kept']:3d}/{n_out:3d} ch)"
                  f"   score L1 {r['score_relL1']:.5f}"
                  f"   desc cos {r['desc_cos_mean']:.6f}"
                  f"  min {r['desc_cos_min']:.6f}")
        print()

    print("=== verdict ===")
    safe = [r for r in rows if r["score_relL1"] < 0.02
            and r["desc_cos_mean"] > 0.999]
    if safe:
        print("  configurations that hold BOTH outputs inside the fp16 budget:")
        for r in safe:
            print(f"    {r['block']:22s} keep {r['keep_ratio']:.2f}  "
                  f"score L1 {r['score_relL1']:.5f}  desc cos {r['desc_cos_mean']:.6f}")
    else:
        print("  NONE.  Every fractional cut that removes real work moves at least")
        print("  one of the two outputs by more than the whole fp16 conversion costs")
        print("  (score L1 < 0.02 and descriptor cos > 0.999).  The backbone is not")
        print("  carrying removable redundancy at these widths.")
        print()
        print("  This is a LOWER BOUND on the damage, as with every truncation in")
        print("  this project: each channel was trained with its neighbours present.")
        print("  A fine-tune could recover some of it - but see the head rather than")
        print("  guessing: `convhead2` is the one block where the answer is not a")
        print("  pruning question at all.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"keep_ratios": KEEP_RATIOS, "rows": rows,
             "n_safe": len(safe)}, indent=2))
        print(f"\n[scan] -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
