#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Where exactly is the residual in variant C, and is it a real defect or float noise?

`probe_head_restructure.py` reports variant C at cosine median 1.0000000 with a
minimum of 0.9999996, but ALSO a `max |Δ|` of 0.029 - and those two disagree.  For
unit vectors a cosine of 0.9999996 means an angle of ~9e-4, so the largest
component difference should be ~9e-4, not 29e-3.  One of the two numbers is
describing something the other is not, and guessing which is how a real defect gets
shipped.

So this locates it: which keypoint, which corner, which channel, and what the
coordinate and its validity are.  A residual that sits on a boundary or on one
channel is a bug; a residual spread evenly across all keypoints is float32
reassociation between `conv2d` and `einsum` doing the same dot product in a
different order.

Usage
    python diag_restructure_gap.py --checkpoint <ckpt>
"""
import argparse
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

from probe_head_restructure import (capture_head_input, shipped_sampled,
                                    variant_descriptor)  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default=str(paths.TRAINED_CKPT))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--keypoints", type=int, default=512)
    args = ap.parse_args()

    from model import load_alike_stage
    from model.gather_ops import to_pixel

    # gathered_head=False: the diagnosis compares against the dense map that
    # capture_head_input (imported above) materialises.
    stage = load_alike_stage(args.checkpoint, top_k=args.keypoints,
                            descriptor_interp="bilinear",
                            gathered_head=False).eval()
    torch.manual_seed(0)
    img = torch.rand(1, 3, args.size, args.size)

    scores, desc_map_norm, x1234 = capture_head_input(stage, img)
    kp, _ = stage.dkd(scores)
    ref = shipped_sampled(stage, desc_map_norm, kp[..., :2], args.size)
    got = variant_descriptor(stage, x1234, kp[..., :2], args.size, "C")

    d = (ref - got).abs()                     # (B, 128, K)
    b, c, k = d.shape
    per_kp = d.amax(dim=1)[0]                 # (K,) worst channel per keypoint
    worst = int(per_kp.argmax())

    print("=== where is the residual? ===")
    print(f"  overall max |Δ|          : {float(d.max()):.4e}")
    print(f"  median over keypoints    : {float(per_kp.median()):.4e}")
    print(f"  90th pct over keypoints  : {float(per_kp.kthvalue(int(0.9*k)).values):.4e}")
    print(f"  worst keypoint index     : {worst}")
    print(f"  its |Δ|                  : {float(per_kp[worst]):.4e}")
    print("  how many keypoints exceed 9e-4 (the cos-implied bound): "
          f"{int((per_kp > 9e-4).sum())}/{k}")
    print()

    # --- the geometry of the worst keypoint -------------------------------
    xp, yp = to_pixel(stage.dkd._normalise(kp[..., :2], args.size, args.size),
                      args.size, args.size)
    x0 = torch.floor(xp).long()
    y0 = torch.floor(yp).long()
    print("=== the worst keypoint's geometry ===")
    print(f"  sub-pixel coord   : ({float(xp[0, worst]):.4f}, {float(yp[0, worst]):.4f})")
    print(f"  corner x0..x0+1   : {int(x0[0, worst])} .. {int(x0[0, worst]) + 1}")
    print(f"  corner y0..y0+1   : {int(y0[0, worst])} .. {int(y0[0, worst]) + 1}")
    oob = int(((x0[0] + 1) > args.size - 1).sum() + ((y0[0] + 1) > args.size - 1).sum()
              + (x0[0] < 0).sum() + (y0[0] < 0).sum())
    print(f"  out-of-bounds corners across ALL keypoints: {oob}")
    print()

    # --- per-corner error: if the residual is global, all corners suffer ----
    print("=== is it one corner or all of them? ===")
    from model.gather_ops import _flatten_gather
    w_desc = stage.net.convhead2.weight[:-1].reshape(128, 128)
    b_desc = stage.net.convhead2.bias[:-1]
    hh = ww = args.size
    row_off = (torch.arange(128, dtype=torch.long) * (hh * ww)).reshape(1, 128, 1)
    for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
        xi, yi = x0[0] + dx, y0[0] + dy
        raw = _flatten_gather(x1234[0:1],
                              yi.clamp(0, hh - 1) * ww + xi.clamp(0, ww - 1),
                              row_off)
        mine = F.normalize(torch.einsum("oc,ck->ok", w_desc, raw[0])
                           + b_desc[:, None], p=2, dim=0)
        # the shipped corner: gather from the NORMALISED dense map
        theirs = _flatten_gather(desc_map_norm[0:1],
                                 yi.clamp(0, hh - 1) * ww + xi.clamp(0, ww - 1),
                                 row_off)[0]
        theirs = F.normalize(theirs, p=2, dim=0)
        cd = float((mine - theirs).abs().max())
        print(f"  corner (dx={dx}, dy={dy}): max |Δ| on the worst keypoint = {cd:.4e}")
    print()
    print("  If every corner shows the same order of magnitude, the residual is the")
    print("  dot product being accumulated differently by `conv2d` and `einsum` -")
    print("  float32 reassociation, not a defect.  A single corner showing a much")
    print("  larger error would be a boundary bug.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
