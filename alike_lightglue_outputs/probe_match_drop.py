#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Which STAGE loses the matches?  Decomposed, because one number fits three stories.

THE OBSERVATION THAT NEEDS EXPLAINING
-------------------------------------
End to end, the fp16 RKNN pipeline emits ~346 matches/pair against torch's ~391
(-11 %), while landing the SAME number of inliers (313.6 against 314.0, -0.1 %).
Fewer matches, same inliers, higher precision - a good outcome for a deployment,
but an unexplained one, and "the matcher is fine" does not follow from it.

WHAT WAS ALREADY RULED OUT, AND HOW
-----------------------------------
  * STAGE 2 PICKS DIFFERENT PARTNERS?  No.  Fed IDENTICAL stage-1 outputs, the two
    backends agree on the partner for 264 of 264 jointly-matched keypoints
    (`partner_disagrees: 0`) and differ by 4 dropped / 3 added out of 512, i.e.
    ~1 match per pair.  So the matcher's decisions are the same given the same
    input.
  * FP16 UNDERFLOW OF THE `-inf` THAT `filter_matches` RELIES ON?  No.  The scores
    at the dropped matches are 0.003-0.043, three orders of magnitude above fp16's
    subnormal floor (5.96e-08), so they are nowhere near the `mscores > 0`
    boundary.  This was the first hypothesis and the log line
    (`range [-inf, 0.0] ... out of the float16`) made it look likely; measuring
    the actual scores killed it.

So a per-pair discrepancy of ~1 in stage 2 cannot produce ~44 end to end, and the
cause must be the INTERACTION: stage 1 emits slightly different keypoints and
descriptors, stage 2 then sees different inputs.  This probe measures exactly that
by crossing the stages:

    A  fx    stage 1  ->  ONNX  stage 2     (the reference)
    B  rknn  stage 1  ->  ONNX  stage 2     <- isolates STAGE 1's effect
    D  fx    stage 1  ->  rknn  stage 2     <- isolates STAGE 2's effect
    C  rknn  stage 1  ->  rknn  stage 2     <- the deployed path

A against B is the detector's contribution; A against D is the matcher's; C is
what ships.  If B lands near C, stage 1 is the whole story; if D lands near A,
stage 2 is exonerated and the detector is where the remaining work is.

WHY "fx" AND NOT "torch" - AND WHY THAT IS NOT A COMPROMISE
----------------------------------------------------------
`A` is the exported ONNX stage 1, NOT the torch model, and the reason is
practical: torch's pipeline needs `omegaconf`, which the RKNN environment does not
have and should not need.  That does not weaken the experiment.  The ONNX export
of stage 1 is already measured bit-exact against torch (keypoint offset 0.000000 px,
`reports/accuracy.json`), so "ONNX stage 1" and "torch stage 1" are the same inputs
to within float32; what this probe is separating is the CONVERTED fp16 stage 1 from
the float32 one, and for that the float32 reference can be either.

Usage
    python probe_match_drop.py --pairs 24
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

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



def split_matcher_outputs(outs):
    """Identify `matches0` and `mscores0` by VALUE, since they share a shape.

    THE BUG THIS REPLACES, WHICH PASSED AS A RESULT
    ----------------------------------------------
    The first version did `ms = next(x for x in o if x is not m)` AFTER
    `m = m.astype(np.int64)`.  `astype` returns a NEW array, so `x is not m` was
    true for every element and the generator returned `outs[0]` regardless -
    silently handing back `matches0` as the score tensor.  It was caught only
    because the reported score range was `[-1, 511]`, which is the range of an
    INDEX array and cannot be a set of match scores.  The lesson is that
    "identify by identity" is fragile when a casting step sits in between; the
    test has to be on a property of the DATA.

    Both outputs are `(1, K)` and the simulator returns both as float32, so shape
    and dtype cannot separate them.  `matches0` is integer-valued with -1 for
    unmatched; `mscores0` is a probability in [0, 1].  A tensor whose entries all
    round-trip through `int` and whose minimum is negative is the match array.
    """
    cands = [np.asarray(o) for o in outs]
    int_like = [c for c in cands
                if np.allclose(c, np.round(c)) and float(np.min(c)) < 0]
    if not int_like:
        raise RuntimeError(
            "no integer-valued output found; cannot tell matches from scores. "
            f"shapes/dtypes: {[(c.shape, str(c.dtype), float(c.min()), float(c.max())) for c in cands]}")
    m = int_like[0]
    rest = [c for c in cands if c is not m]
    if not rest:
        raise RuntimeError("only one output returned")
    scores = rest[0]
    if float(scores.min()) < 0 or float(scores.max()) > 1.0001:
        raise RuntimeError(
            f"the candidate score tensor has range [{scores.min()}, {scores.max()}], "
            f"which is not a probability - the identification is wrong")
    return m.astype(np.int64), scores.astype(np.float32)


def load_pairs(n, size):
    import cv2
    out = []
    for seq in sorted(paths.hpatches().iterdir()):
        if not seq.is_dir() or seq.name[0] == "i":
            continue
        ref = seq / "1.ppm"
        for q in range(2, 7):
            qf = seq / f"{q}.ppm"
            if ref.is_file() and qf.is_file():
                a = cv2.imread(str(ref), cv2.IMREAD_GRAYSCALE)
                b = cv2.imread(str(qf), cv2.IMREAD_GRAYSCALE)
                if a is None or b is None:
                    continue
                a = cv2.resize(a, (size, size), interpolation=cv2.INTER_LINEAR)
                b = cv2.resize(b, (size, size), interpolation=cv2.INTER_LINEAR)
                out.append((np.stack([a] * 3, -1), np.stack([b] * 3, -1)))
                break
        if len(out) >= n:
            break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=24)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--s1-onnx", default=str(_REPO / "weights/original/alike_stage_bil.onnx"))
    ap.add_argument("--s2-onnx", default=str(_REPO / "weights/optimized/lightglue_stage_d7.onnx"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import onnxruntime as ort
    from convert_to_rknn import capture_build_log
    from rknn.api import RKNN

    so = ort.SessionOptions()
    so.log_severity_level = 3
    s1_onnx = ort.InferenceSession(args.s1_onnx, so, providers=["CPUExecutionProvider"])
    s2_onnx = ort.InferenceSession(args.s2_onnx, so, providers=["CPUExecutionProvider"])

    def rk_build(onnx, norm):
        rk = RKNN(verbose=True)      # construction-time verbose: the warnings need it
        cfg = {"target_platform": "rk3588", "float_dtype": "float16"}
        if norm:
            cfg["mean_values"] = [[0, 0, 0]]
            cfg["std_values"] = [[255, 255, 255]]
        rk.config(**cfg)
        rk.load_onnx(model=str(onnx))
        cap = capture_build_log()
        try:
            with cap:
                rk.build(do_quantization=False)
        finally:
            cap.cleanup()
        rk.init_runtime(target=None)
        return rk

    print("[drop] building interpreters (once)")
    rk1 = rk_build(args.s1_onnx, True)
    rk2 = rk_build(args.s2_onnx, False)

    pairs = load_pairs(args.pairs, args.size)
    print(f"[drop] {len(pairs)} HPatches pairs, {args.size}px\n")

    def matcher_onnx(k, d):
        m, ms = s2_onnx.run(None, {
            "keypoints0": k[0:1], "keypoints1": k[1:2],
            "descriptors0": d[0:1], "descriptors1": d[1:2]})
        return m.astype(np.int64), ms.astype(np.float32)

    def matcher_rknn(k, d):
        o = rk2.inference(inputs=[k[0:1].astype(np.float32),
                                  k[1:2].astype(np.float32),
                                  d[0:1].astype(np.float32),
                                  d[1:2].astype(np.float32)])
        return split_matcher_outputs(o)

    rows = []
    for i, (a, b) in enumerate(pairs):
        u8 = np.stack([a, b]).astype(np.uint8)                       # (2,H,W,3) NHWC
        nchw = u8.transpose(0, 3, 1, 2).astype(np.float32) / 255.0

        # --- stage 1, both ways --------------------------------------------
        ko, do_, _ = s1_onnx.run(None, {"image": nchw})
        outs1 = [np.asarray(o) for o in
                 rk1.inference(inputs=[u8], data_format="nhwc")]
        kr = next(o for o in outs1 if o.ndim == 3 and o.shape[-1] == 2).astype(np.float32)
        dr = next(o for o in outs1 if o.ndim == 3 and o.shape[-1] == 128).astype(np.float32)

        # A reference / B stage-1 replaced / D stage-2 replaced / C deployed
        mA, _ = matcher_onnx(ko, do_)
        mB, _ = matcher_onnx(kr, dr)
        mD, _ = matcher_rknn(ko, do_)
        mC, _ = matcher_rknn(kr, dr)

        def nv(m):
            return int((m[0] >= 0).sum())

        # How far apart the two detectors are.  BOTH of these are matched by
        # nearest neighbour first: `topk` returns its selection in score order and
        # that order is backend-specific, so comparing element `i` of the two
        # outputs compares two DIFFERENT keypoints.  Done positionally this reads
        # as "kd 0.32 px, descriptor cosine 0.69", which looks like a broken
        # detector and is purely an ordering artefact - the failure mode this
        # project has already documented twice.
        # `cKDTree.query` returns `(distance, index)` in that order.  Unpacking it
        # the other way gives float distances where integer indices are expected,
        # and NumPy's error for that is `arrays used as indices must be of integer
        # type` - which points at the indexing rather than at the swap.
        dist, idx = cKDTree(ko[0]).query(kr[0], k=1)
        kp_shift = float(np.median(dist))
        do_matched = do_[0][idx]
        d_cos = float(np.median(np.abs(
            (do_matched * dr[0]).sum(-1) /
            (np.linalg.norm(do_matched, axis=-1) *
             np.linalg.norm(dr[0], axis=-1) + 1e-12))))

        # Score at the matches the RKNN matcher DROPS, and how many sit below what
        # fp16 can represent.  `filter_matches` keeps a match when
        # `mscores0 > filter_threshold` and `mscores0 = exp(max0.values)`, so a
        # score that underflows to exactly 0 is DROPPED - and exp(-20) is 2e-9,
        # which is fine in float32 and subnormal-to-zero in float16.  This is the
        # mechanism the fp16 range warnings point at; counting here either
        # confirms it or rules it out.
        _, ms_onnx_d = matcher_onnx(ko, do_)
        dropped = (mA[0] >= 0) & (mD[0] < 0)
        ms_at_drop = ms_onnx_d[0][dropped]
        added = (mA[0] < 0) & (mD[0] >= 0)

        rows.append({"pair": i, "A_onnx1_onnx2": nv(mA), "B_rknn1_onnx2": nv(mB),
                     "C_rknn1_rknn2": nv(mC), "D_onnx1_rknn2": nv(mD),
                     "kp_median_shift_px": kp_shift, "desc_median_cos": d_cos,
                     "dropped_by_matcher": int(dropped.sum()),
                     "added_by_matcher": int(added.sum()),
                     "n_dropped_score_lt_1e-4": int((ms_at_drop < 1e-4).sum()),
                     "n_dropped_score_zero": int((ms_at_drop == 0).sum()),
                     "dropped_score_p90": float(np.percentile(ms_at_drop, 90))
                     if dropped.any() else 0.0,
                     "A_vs_D_agree": int((mA[0] == mD[0]).sum())})
        print(f"    [{i+1}/{len(pairs)}] A={nv(mA):3d} B={nv(mB):3d} "
              f"C={nv(mC):3d} D={nv(mD):3d}  kp {kp_shift:.4f}px  "
              f"desc cos {d_cos:.6f}  drop={int(dropped.sum()):3d} "
              f"(<1e-4: {int((ms_at_drop < 1e-4).sum())})")

    rk1.release()
    rk2.release()

    def mean(key):
        return float(np.mean([r[key] for r in rows]))

    rep = {"pairs": len(rows), "size": args.size,
           "A_onnx1_onnx2_matches": mean("A_onnx1_onnx2"),
           "B_rknn1_onnx2_matches": mean("B_rknn1_onnx2"),
           "C_rknn1_rknn2_matches": mean("C_rknn1_rknn2"),
           "D_onnx1_rknn2_matches": mean("D_onnx1_rknn2"),
           "stage1_effect_matches": mean("B_rknn1_onnx2") - mean("A_onnx1_onnx2"),
           "stage2_effect_matches": mean("D_onnx1_rknn2") - mean("A_onnx1_onnx2"),
           "total_effect_matches": mean("C_rknn1_rknn2") - mean("A_onnx1_onnx2"),
           "kp_median_shift_px": mean("kp_median_shift_px"),
           "desc_median_cos": mean("desc_median_cos"),
           "dropped_by_matcher": mean("dropped_by_matcher"),
           "added_by_matcher": mean("added_by_matcher"),
           "dropped_score_lt_1e-4": mean("n_dropped_score_lt_1e-4"),
           "dropped_score_zero": mean("n_dropped_score_zero"),
           "rows": rows}
    print("\n=== per-pair match counts, averaged ===")
    print("  A  onnx stage1 -> ONNX stage2 (reference) : "
          f"{rep['A_onnx1_onnx2_matches']:7.2f}")
    print("  B  rknn stage1 -> ONNX stage2             : "
          f"{rep['B_rknn1_onnx2_matches']:7.2f}   "
          f"({rep['stage1_effect_matches']:+.2f}  <- detector)")
    print("  D  onnx stage1 -> rknn stage2             : "
          f"{rep['D_onnx1_rknn2_matches']:7.2f}   "
          f"({rep['stage2_effect_matches']:+.2f}  <- matcher)")
    print("  C  rknn stage1 -> rknn stage2 (deployed)  : "
          f"{rep['C_rknn1_rknn2_matches']:7.2f}   "
          f"({rep['total_effect_matches']:+.2f}  <- ships)")
    print("\n  detector (order-matched): keypoint median offset "
          f"{rep['kp_median_shift_px']:.4f} px, descriptor median cosine "
          f"{rep['desc_median_cos']:.6f}")
    print(f"  matcher: drops {rep['dropped_by_matcher']:.1f}/pair, "
          f"adds {rep['added_by_matcher']:.1f}/pair")
    print(f"    of the dropped, {rep['dropped_score_zero']:.1f} had score exactly 0 "
          f"and {rep['dropped_score_lt_1e-4']:.1f} had score < 1e-4")
    print("    (0 == fp16 underflow of `exp`, which `mscores > 0` then rejects)")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(rep, indent=2))
        print(f"\n[drop] -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
