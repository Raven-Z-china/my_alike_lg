#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Can the descriptor head be computed only where it is read?  Three variants, tested.

THE FINDING THIS FOLLOWS FROM
-----------------------------
`probe_flops_split.py` measured the pipeline and one convolution dominates:

    net.convhead2   4.33G MACs   24.0 % of everything
    (next largest)  604M          3.3 %

and `scan_channel_fractions.py` showed it cannot be pruned: cutting its descriptor
channels leaves the score map BIT-IDENTICAL (score L1 = 0.00000 at every keep ratio)
while the descriptor cosine falls to 0.921.  The head is 129 output channels of which
ONE is the score and 128 are the descriptor.

WHAT IS WASTED
--------------
`convhead2` is a 1x1 convolution - a per-pixel linear map - applied to all 512x512
pixels, producing 129 x 262,144 values, and the detector then reads 512 of those
pixels.  33.6M descriptor values are computed per frame and **65,536 are used
(0.2 %)**.  Only the score channel needs to be dense, because NMS and top-k act on
the whole map.

WHY THE SAMPLING MODE DECIDES WHETHER THIS IS EVEN POSSIBLE
-----------------------------------------------------------
`AlikeStage.forward` does

    descriptor_map = F.normalize(descriptor_map, p=2, dim=1)   # per pixel
    desc = self.desc_sample(descriptor_map, x_pix, y_pix)      # BILINEAR
    desc = F.normalize(desc, p=2, dim=1)                       # per keypoint

and the deployed graph is the `bilinear` one (`weights/original/alike_stage_bil.onnx`).  The
per-pixel normalisation is NON-LINEAR, so it does NOT commute with bilinear
interpolation:

    normalize_perpixel(W @ bilinear(x))  !=  bilinear(normalize_perpixel(W @ x))

An earlier version of this probe measured the substitution on a NEAREST sample
(`round`) - where the two DO agree, because selecting one pixel commutes with any
per-pixel function - and reported "exact".  That result was true and covered the
wrong mode: it said nothing about the bilinear path that actually ships.  This is
the same failure shape as the order-unmatched descriptor comparison and the
`heads 4->2` ablation - **a measurement that watches the wrong thing reports
success.**

THE THREE VARIANTS
------------------
    A  nearest gather of raw features -> head rows -> normalise
       Exact versus a NEAREST dense path, and that is all.  Kept as the control
       that shows the sampling mode is what breaks the naive form.

    B  bilinear sample of raw features -> head rows -> normalise
       The obvious cheap version.  Not equal to the shipped path, and the gap is
       measured below.

    C  FOUR-CORNER: gather raw features at the 4 integer corners, apply the head
       rows to EACH corner, normalise EACH corner, then combine with the bilinear
       weights.
       This is the SAME arithmetic as the shipped path, restricted to the corners
       that are actually needed: `bilinear(normalize_perpixel(W x))` evaluated only
       at the sample points.  It should be exact to float32 associativity.

Only C is a valid substitution.  The reference it is compared against is the real
`GatherSampleBilinear` applied to the real normalised dense map, i.e. the shipped
computation, so the comparison is against the thing that runs.

Usage
    python probe_head_restructure.py --checkpoint <ckpt>
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


@torch.no_grad()
def capture_head_input(stage, img):
    """Run the real forward and capture what `convhead2` receives.

    The trunk is NOT re-implemented here, deliberately: it is not `block1..block4`
    in sequence, it is four blocks at four resolutions, four 1x1 projections to
    `dim // 4`, a multi-scale upsampling concat and a 1x1 head prep.  A
    reimplementation that drifts from the model would compare two things that are
    not the shipped pipeline and report a difference that is the copy's fault.
    One hook on the head's INPUT gives both paths the same starting tensor.
    """
    net = stage.net
    caught = {}

    def hook(mod, inp, out):
        caught["x"] = inp[0].detach()

    h = net.convhead2.register_forward_hook(hook)
    try:
        scores_map, desc_map = stage.dense_maps(img)
    finally:
        h.remove()
    return scores_map, desc_map, caught["x"]


def _head_rows(stage):
    """The descriptor rows of `convhead2` as a (128, 128) matrix and its bias."""
    w = stage.net.convhead2.weight
    b = stage.net.convhead2.bias
    w_desc = w[:-1].reshape(w.shape[0] - 1, w.shape[1])
    b_desc = (b[:-1] if b is not None
              else torch.zeros(w_desc.shape[0], device=w.device, dtype=w.dtype))
    return w_desc, b_desc


@torch.no_grad()
def shipped_sampled(stage, desc_map_norm, kp, size):
    """THE REFERENCE at the SAMPLER's output, before the stage's final normalise.

    `AlikeStage.forward` is

        desc = self.desc_sample(descriptor_map, x_pix, y_pix)   # bilinear
        desc = F.normalize(desc, p=2, dim=1)                    # per keypoint

    and that second line is why this function stops one step early.  Comparing the
    final descriptors would fold the per-keypoint normalisation into BOTH sides and
    hide everything the interpolation did: a bilinear combination of unit vectors is
    not unit, so `|shipped - variant|` on the FINAL vectors is dominated by a
    magnitude difference that the next normalise removes anyway.  An earlier version
    of this file compared the final vectors and reported `max |Δ| = 0.029` alongside
    a cosine minimum of 0.9999996 - two numbers that cannot both describe the same
    comparison, which is what gave the mistake away.

    The sampler output is the honest interface: it is where the two implementations
    must agree, and the normalise after it is shared unchanged by both.
    """
    from model.gather_ops import to_pixel
    xp, yp = to_pixel(stage.dkd._normalise(kp, size, size), size, size)
    return stage.desc_sample(desc_map_norm, xp, yp)       # GatherSampleBilinear


def _valid(xp, yp, h, w, dtype):
    """0/1 validity mask for a pixel coordinate, matching `gather_ops` exactly."""
    xi, yi = xp.long(), yp.long()
    vx = (xi >= 0).to(dtype) * (xi <= w - 1).to(dtype)
    vy = (yi >= 0).to(dtype) * (yi <= h - 1).to(dtype)
    return (vx * vy).unsqueeze(1)


def _gather(x1234, xp, yp, row_off, valid=None):
    """Gather a (B, C, H, W) feature map at pixel coords.

    `valid` is applied AFTER the gather and only when the caller passes it, which
    mirrors `GatherSampleBilinear.corner`.  Getting that ORDER wrong is a real bug
    and it was in the first version of this file: masking the RAW features to zero
    and then applying the head leaves `bias`, and `normalize(bias)` is a unit
    vector rather than a zero vector - so an out-of-bounds corner contributed a
    full-strength direction instead of nothing, giving a 0.029 maximum error on the
    keypoints that reach the edge.

    And they do reach it: the DKD border mask keeps map indices 3..509 and the
    sub-pixel offset spans [-2, 2], so a keypoint can sit at 511.x and its
    `x0 + 1` corner is outside the map.
    """
    from model.gather_ops import _flatten_gather
    b, c, h, w = x1234.shape
    xi, yi = xp.long(), yp.long()
    g = _flatten_gather(x1234, yi.clamp(0, h - 1) * w + xi.clamp(0, w - 1),
                        row_off)
    return g if valid is None else g * valid


@torch.no_grad()
def variant_descriptor(stage, x1234, kp, size, mode):
    """The restructured descriptor.  `mode` selects A / B / C."""
    from model.gather_ops import to_pixel
    w_desc, b_desc = _head_rows(stage)
    b_, c_, h, w_ = x1234.shape
    row_off = (torch.arange(b_ * c_, dtype=torch.long) * (h * w_)).reshape(
        b_, c_, 1)
    row_off = row_off.to(x1234.device)

    xp, yp = to_pixel(stage.dkd._normalise(kp, size, size), size, size)
    if mode == "A":
        # nearest: the control.  Exact against a nearest dense path only.
        return torch.einsum("oc,bck->bok", w_desc,
                            _gather(x1234, xp.round(), yp.round(), row_off)) \
            + b_desc[None, :, None]

    if mode == "B":
        # bilinear on RAW features, then the head, then normalise.  The obvious
        # cheap form, and NOT equal to the shipped path.
        x0, y0 = torch.floor(xp), torch.floor(yp)
        wx = (xp - x0).unsqueeze(1)
        wy = (yp - y0).unsqueeze(1)
        g00 = _gather(x1234, x0, y0, row_off)
        g01 = _gather(x1234, x0 + 1, y0, row_off)
        g10 = _gather(x1234, x0, y0 + 1, row_off)
        g11 = _gather(x1234, x0 + 1, y0 + 1, row_off)
        raw = (g00 * (1 - wx) + g01 * wx) * (1 - wy) \
            + (g10 * (1 - wx) + g11 * wx) * wy
        # the head applied to the interpolated FEATURES - the form that does not
        # commute with the per-pixel normalisation
        return torch.einsum("oc,bck->bok", w_desc, raw) + b_desc[None, :, None]

    if mode == "C":
        # FOUR-CORNER: head and normalise PER CORNER, then combine.  Same
        # arithmetic as the dense path, only at the corners that are read - and
        # with the validity mask applied at the SAME point in the pipeline, i.e.
        # after the normalisation.  See `_gather` for why the order matters.
        x0, y0 = torch.floor(xp), torch.floor(yp)
        wx = (xp - x0).unsqueeze(1)
        wy = (yp - y0).unsqueeze(1)

        def corner(xi, yi):
            # per-corner: head, THEN normalise - which is what the dense path does
            # per pixel.  The validity mask goes on after the normalise, matching
            # `GatherSampleBilinear.corner` exactly.
            v = _valid(xi, yi, h, w_, x1234.dtype)
            raw = _gather(x1234, xi, yi, row_off)
            d = torch.einsum("oc,bck->bok", w_desc, raw) + b_desc[None, :, None]
            return F.normalize(d, p=2, dim=1) * v

        d00 = corner(x0, y0)
        d01 = corner(x0 + 1, y0)
        d10 = corner(x0, y0 + 1)
        d11 = corner(x0 + 1, y0 + 1)
        top = d00 * (1 - wx) + d01 * wx
        bot = d10 * (1 - wx) + d11 * wx
        return top * (1 - wy) + bot * wy

    raise ValueError(mode)


def macs_dense(size=512, c=128):
    return (c + 1) * c * size * size


def macs_c(size=512, k=512, c=128):
    # score channel dense + 4 corners x (c x c) matmul at K points
    return c * size * size + 4 * c * c * k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default=str(paths.TRAINED_CKPT))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--keypoints", type=int, default=512)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from model import load_alike_stage

    # gathered_head=False: the probe compares variants AGAINST the dense
    # descriptor map, so the stage must be able to produce it.
    stage = load_alike_stage(args.checkpoint, top_k=args.keypoints,
                             descriptor_interp="bilinear",
                             gathered_head=False).eval()
    torch.manual_seed(0)
    img = torch.rand(1, 3, args.size, args.size)

    scores, desc_map_norm, x1234 = capture_head_input(stage, img)
    kp, _ = stage.dkd(scores)
    ref = shipped_sampled(stage, desc_map_norm, kp[..., :2], args.size)

    print("=== which substitution reproduces the SHIPPED bilinear path? ===")
    print("  reference = GatherSampleBilinear on the per-pixel-normalised dense")
    print("  map, at the SAMPLER's output - the stage's final `F.normalize` is")
    print("  applied to both sides afterwards and is not part of the substitution.")
    print("  (this is `weights/original/alike_stage_bil.onnx`, descriptor_interp='bilinear')")
    print()
    results = {}
    for mode, label in (("A", "nearest gather + head"),
                        ("B", "bilinear on RAW features + head"),
                        ("C", "4-corner: head+normalise per corner, then combine")):
        d = variant_descriptor(stage, x1234, kp[..., :2], args.size, mode)
        cos = F.cosine_similarity(ref, d, dim=1)
        mx = float((ref - d).abs().max())
        # after the SAME final normalise the stage applies - the number the
        # matcher actually sees
        cos_f = F.cosine_similarity(F.normalize(ref, dim=1),
                                   F.normalize(d, dim=1), dim=1)
        results[mode] = {"max_abs_sampler": mx,
                         "cos_median_sampler": float(cos.median()),
                         "cos_min_sampler": float(cos.min()),
                         "cos_min_final": float(cos_f.min()), "label": label}
        print(f"  {mode}: {label}")
        print(f"       sampler: max |Δ| {mx:.3e}  cos med {cos.median():.7f}  "
              f"min {cos.min():.7f}")
        print(f"       after the stage's final normalise: cos min {cos_f.min():.7f}")
    print()

    md, mc = macs_dense(args.size, 128), macs_c(args.size, args.keypoints, 128)
    total_dense = 18.1e9
    total_c = total_dense - md + mc
    print("=== cost of the head, for variant C ===")
    print(f"  dense (current)  : {md/1e9:6.3f} G MACs")
    print(f"  variant C        : {mc/1e6:6.1f} M MACs   "
          f"(score dense {128*args.size**2/1e6:.1f}M + 4 corners x "
          f"{128*128*args.keypoints/1e6:.1f}M)")
    print(f"  saving           : {md/mc:6.1f}x on that convolution")
    print(f"  pipeline         : {total_dense/1e9:.2f} G -> {total_c/1e9:.2f} G MACs "
          f"({100*(total_c-total_dense)/total_dense:+.1f} %)")
    print()

    ok_c = results["C"]["cos_min_final"] > 1 - 1e-5
    ok_b = results["B"]["cos_min_final"] > 1 - 1e-5
    print("=== verdict ===")
    print(f"  A (nearest)  : cos min {results['A']['cos_min_final']:.7f}  "
          f"-> {'agrees' if results['A']['cos_min_final'] > 1-1e-5 else 'DIFFERS from the shipped bilinear path'}")
    print(f"  B (raw)      : cos min {results['B']['cos_min_final']:.7f}  "
          f"-> {'agrees' if ok_b else 'DIFFERS - NOT a valid substitution'}")
    print(f"  C (corners)  : cos min {results['C']['cos_min_final']:.7f}  "
          f"-> {'EXACT' if ok_c else 'differs'}")
    print()
    if ok_c:
        print("  ADOPT C.  It is the shipped arithmetic evaluated only at the corners")
        print("  that are read, so it is exact and carries none of the 'lower bound'")
        print("  caveat that every pruning number in this project does.")
    if not ok_b:
        print("  B IS THE TRAP: it looks like the same idea, is 4x cheaper than C, and")
        print("  is wrong because the per-pixel normalisation does not commute with")
        print("  bilinear interpolation.")
    if results["A"]["cos_min_final"] < 1 - 1e-5:
        print("  A is the earlier mistake, now visible: it is exact only against a")
        print("  NEAREST dense path, which is not the one this deployment ships.")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps({
            "reference": "GatherSampleBilinear on per-pixel-normalised dense map",
            "variants": results,
            "macs_dense": md, "macs_variant_c": mc,
            "speedup_on_conv": md / mc,
            "pipeline_macs_dense": total_dense, "pipeline_macs_c": total_c,
            "adopt": "C" if ok_c else None}, indent=2))
        print(f"\n[probe] -> {args.out}")
    return 0 if ok_c else 1


if __name__ == "__main__":
    sys.exit(main())
