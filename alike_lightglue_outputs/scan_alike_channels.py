#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Per-block channel sensitivity of the ALIKE backbone, driven by MEASURED MACs.

WHY THE SCAN IS ORDERED BY MACs AND NOT BY PARAMETERS
-----------------------------------------------------
`scripts/probe_flops_split.py` measured the two stages and the answer overturns the
parameter-based intuition:

    stage 1 (backbone + DKD)   6.5G MACs   35.9 % of the pipeline
    stage 2 (matcher)         11.6G MACs   64.1 %

against **2.8 % of the parameters**.  The backbone is not negligible; it is a
third of the work.  And inside it one convolution dominates:

    net.convhead2    4.3G   66.7 % of stage 1   24.0 % of everything
    net.block1.conv2 604M    9.3 %
    net.block2.conv2 604M    9.3 %

So a channel-pruning budget spent on the `block*` trunk would be chasing 2.15G
MACs spread over a dozen convolutions with residual connections between them,
while the single largest item in the graph is a 1x1 conv whose output is **99.8 %
discarded** (128 of its 129 channels are descriptor values, and only 512 of the
262,144 pixels are ever sampled).

WHAT THIS SCRIPT MEASURES, AND WHAT IT IS FOR
---------------------------------------------
For each named block: zero the block's output, re-run, and report the relative L1
change of the final score map.  Low sensitivity means the block can be considered
for removal; high means it cannot.  This is the gate that has to pass before any
channels are cut, because a block already at the tolerance cannot be recovered by
fine-tuning.

Per-channel selection INSIDE a block is a separate step and is not attempted for
blocks whose whole-output sensitivity is above the tolerance - see the printout at
the end, which ranks the blocks against the tolerance rather than just listing
numbers.

Usage
    python scan_alike_channels.py --checkpoint <ckpt>
"""
import argparse
import sys
from pathlib import Path

import torch

# Analysis-only script, kept OUTSIDE the deployment repository: nothing in
# alike_lightglue_ONNX&RKNN_deploy/ imports or reads any output from it.  It needs that
# repository on sys.path for `model` and `paths`, and writes its reports HERE.
_REPO = Path(__file__).resolve().parents[1] / "alike_lightglue_ONNX&RKNN_deploy"
if not _REPO.is_dir():
    raise SystemExit(f"cannot find the deployment repo at {_REPO}; edit _REPO here")
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(_REPO / "pruning"))
HERE = Path(__file__).resolve().parent        # outputs/ - where reports go
import paths  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default=str(paths.TRAINED_CKPT))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--keypoints", type=int, default=512)
    ap.add_argument("--tol", type=float, default=0.02)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from structured import channel_sensitivity
    from model import load_alike_stage

    # gathered_head=False: the scan measures the DENSE maps, which the gathered
    # head never materialises (dense_maps raises under the default).
    stage = load_alike_stage(args.checkpoint, top_k=args.keypoints,
                             gathered_head=False).eval()
    torch.manual_seed(0)                    # reproducible probe image
    img = torch.rand(1, 3, args.size, args.size)

    # The blocks worth asking about: everything with a conv in the backbone trunk,
    # plus the head.  Grouped rather than per-conv because a ResBlock's two convs
    # and its downsample form one residual unit - zeroing one of the pair measures
    # something that cannot occur (the add would still carry the other).
    blocks = [
        "net.conv1", "net.block1.conv1", "net.block1.conv2",
        "net.conv2", "net.block2.conv1", "net.block2.conv2",
        "net.block2.downsample",
        "net.conv3", "net.block3.conv1", "net.block3.conv2",
        "net.block3.downsample",
        "net.conv4", "net.block4.conv1", "net.block4.conv2",
        "net.block4.downsample",
        "net.convhead2",
    ]
    # `stage.net.named_modules()` yields names WITHOUT the `net.` prefix, so
    # filtering against it removes every entry and the scan silently measures
    # nothing - it printed "0 of 0 blocks" and the summary read as a verdict.
    # Use the STAGE's names, because that is the namespace `channel_sensitivity`
    # looks modules up in.
    have = {n for n, _ in stage.named_modules()}
    missing = [b for b in blocks if b not in have]
    if missing:
        raise SystemExit(
            f"{len(missing)} of {len(blocks)} block names do not exist on the "
            f"stage: {missing[:5]}. The scan would silently measure nothing.")
    blocks = [b for b in blocks if b in have]

    def forward(model):
        # The score map is the only output the detector actually acts on, so it is
        # the right thing to measure sensitivity against: a change that does not
        # move the score map cannot move a keypoint.
        scores, _ = model.dense_maps(img)
        return scores

    sens = channel_sensitivity(stage, blocks, forward, tol=args.tol)

    print("=== whole-block output sensitivity (relative L1 change of the score "
          f"map, tolerance {args.tol}) ===")
    print(f"{'block':26s} {'sensitivity':>12s}   verdict")
    for name, v in sorted(sens.items(), key=lambda kv: kv[1]):
        verdict = ("PRUNABLE CANDIDATE" if v <= args.tol
                   else "over tolerance")
        print(f"  {name:24s} {v:12.5f}   {verdict}")
    n_ok = sum(1 for v in sens.values() if v <= args.tol)
    print()
    print(f"  {n_ok} of {len(sens)} blocks are at or below the tolerance.")
    if n_ok == 0:
        print("  => channel pruning has no room at this tolerance: every block's")
        print("     output is load-bearing on the score map. Raising the tolerance")
        print("     would mean accepting a detector whose keypoints move, which the")
        print("     deployment cannot absorb - see the accuracy section of the README.")
    print()
    print("  NOTE: this is a LOWER BOUND on the damage, as with every truncation in")
    print("  this project - the block was trained with its channels present. And it")
    print("  measures the SCORE MAP, so a block that matters only to the descriptor")
    print("  would read as harmless here. The descriptor head is checked separately.")

    if args.out:
        import json
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(
            {"tolerance": args.tol, "sensitivity": sens,
             "n_punnable_at_tol": n_ok}, indent=2))
        print(f"\n[scan] -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
