#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""End-to-end accuracy: the two converted stages, chained, against the torch model.

WHY THIS IS NOT `scripts/check_accuracy.py`
-------------------------------------------
`check_accuracy.py` scores ONE stage against a torch reference on tensors.  That is
the right tool to localise a conversion defect, and it is what produced the
per-stage numbers in the README.  It is the wrong tool to answer the deployment
question, which is:

    does the C++ pipeline that will run on the board produce the same matches
    as the trained model?

That question needs the two stages CHAINED exactly as the C++ sample chains them -
stage-1 keypoints/descriptors straight into stage 2, no torch anywhere in the
loop - and then scored with a metric that is invariant to the things that are
allowed to differ (keypoint order, a handful of near-tie ranks flipping).

WHAT IS MEASURED
----------------
The **HPatches** benchmark (multiple public viewpoint sequences, 540 pairs,
homography ground truth) - an open-source dataset, the same one the training-side
reports use, so the numbers are directly comparable.

      mAA        : mean average accuracy of the estimated homography
      @3px / @5px: mAA restricted to a reprojection tolerance
      mprec@3px  : of the matches produced, the fraction whose endpoints agree
                   with the ground-truth homography within 3 px
      n_inl_gt   : mprec@3px x n_match, i.e. the deterministic inlier count
      n_inl      : RANSAC inlier count (what `prec@k` cannot express: a matcher
                   that emits fewer matches can win on precision and lose on the
                   number of usable correspondences)

WHY A HOMOGRAPHY ACCURACY, AND NOT A DETECTION+MATCHING SCORE ALONE
-------------------------------------------------------------------
`mAA` scores the whole chain through a geometric estimator, which is what the
deployment question is about; `mprec@3px` and `n_inl_gt` isolate the matcher from
the estimator.  A conversion that broke the descriptor lookup would move the
precision figure first; a conversion that broke the geometry would move both.
Reading only one of them is how a real regression gets missed.

THE THREE COLUMNS
-----------------
    torch  the trained checkpoint, through `model.load_*_stage` (self-contained)
    onnx   the exported graph(s), fp32, onnxruntime CPU
    rknn   the converted .rknn model(s), RKNN PC simulator, fp16

`--pipeline` selects which of the last two is run against torch; running all
three in one command is what makes the attribution possible, and it is the
default.

ON THE SIMULATOR
----------------
The RKNN "simulator" (`init_runtime(target=None)`) runs the converted graph on the
x86 host with the NPU's arithmetic.  It validates NUMERICS.  It says nothing about
latency - a simulator timing is the speed of the host CPU running an emulation of
the NPU's op sequence, which is not even the right order of magnitude.  No number
in the output of this script is a speed claim.

Usage -- the two arms live in different envs, so they are two runs: the `rknn`
arm has no omegaconf and cannot build the torch reference.  `--ref-from` reads
the reference arm's JSON so both halves score the SAME pairs.

    # in the export env
    python eval/run_accuracy.py \
        --checkpoint weights/checkpoints/alike_native_gl_s1_d7.tar \
        --pipeline torch onnx \
        --hpatches 540 --size 512 --keypoints 512 \
        --json <reports>/e2e_ta.json --report <reports>/e2e_ta.md
    # in the rknn env
    python eval/run_accuracy.py --pipeline rknn-fp16 \
        --ref-from <reports>/e2e_ta.json \
        --json <reports>/e2e_rknn.json --report <reports>/e2e_rknn.md
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import paths  # noqa: E402
sys.path.insert(0, str(REPO / "scripts"))
from portable_path import portable  # noqa: E402
from model import load_alike_stage, load_lightglue_stage  # noqa: E402


# --------------------------------------------------------------------------- #
# pipelines
# --------------------------------------------------------------------------- #
class TorchPipeline:
    """The trained model, as trained - the reference everything is scored against."""

    name = "torch"

    def __init__(self, checkpoint, size, keypoints):
        self.stage1 = load_alike_stage(checkpoint, top_k=keypoints,
                                       descriptor_interp="bilinear").eval()
        self.stage2 = load_lightglue_stage(checkpoint,
                                           image_size=(size, size)).eval()

    @torch.no_grad()
    def __call__(self, rgb0_u8, rgb1_u8):
        """Two HWC uint8 colour images -> (k0, k1, m0, ms0) in torch."""
        a = torch.from_numpy(rgb0_u8).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        b = torch.from_numpy(rgb1_u8).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        k, d, _ = self.stage1(torch.cat([a, b], 0))
        m0, ms0 = self.stage2(k[0:1], k[1:2], d[0:1], d[1:2])
        return k[0], k[1], m0[0], ms0[0]

    def release(self):
        pass


class OnnxPipeline:
    """The exported graphs, fp32, onnxruntime CPU.

    Deliberately NOT the CUDA provider: the RKNN simulator is an x86 CPU
    emulation of the NPU's arithmetic, and running the ONNX arm on the GPU would
    put a third arithmetic in the comparison for no benefit.
    """

    name = "onnx"

    def __init__(self, s1_path, s2_path):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.s1 = ort.InferenceSession(str(s1_path), so,
                                       providers=["CPUExecutionProvider"])
        self.s2 = ort.InferenceSession(str(s2_path), so,
                                       providers=["CPUExecutionProvider"])

    def __call__(self, rgb0_u8, rgb1_u8):
        batch = np.stack([rgb0_u8, rgb1_u8]).transpose(0, 3, 1, 2)
        img = batch.astype(np.float32) / 255.0
        k, d, _ = self.s1.run(None, {"image": img})
        m0, ms0 = self.s2.run(None, {
            "keypoints0": k[0:1], "keypoints1": k[1:2],
            "descriptors0": d[0:1], "descriptors1": d[1:2]})
        return (torch.from_numpy(k[0]), torch.from_numpy(k[1]),
                torch.from_numpy(m0[0]), torch.from_numpy(ms0[0]))

    def release(self):
        pass


class RknnPipeline:
    """The converted models, through the RKNN PC simulator.

    A fresh runtime is created per call rather than once: the simulator caches
    its context, and reusing one across hundreds of pairs grows host memory
    without bound.  The per-call cost is real but this is an accuracy harness,
    not a benchmark - correctness first, and the timing is reported separately by
    `scripts/bench_stages.py`.
    """

    def __init__(self, s1_path, s2_path, name="rknn-fp16",
                 s1_onnx=None, s2_onnx=None):
        if s1_onnx is None or s2_onnx is None:
            raise ValueError(
                "the simulator rebuilds from ONNX, so s1_onnx/s2_onnx are "
                "required - pass the pair the .rknn files were built from")
        self.name = name
        # ONNX + build, NOT `load_rknn`.  This is forced by the toolkit, and the
        # error is worth recording because the obvious route is the wrong one:
        #
        #     load_rknn(model.rknn) + init_runtime(target=None)
        #         -> "RKNN model that loaded by 'load_rknn' not support inference on
        #             the simulator, please set 'target' first"
        #
        # i.e. the PC simulator only exists on the load+build path.  So the
        # harness rebuilds the graph here.  The config below is therefore
        # load-bearing: it must be IDENTICAL to what `convert_to_rknn.py` used, or
        # this measures a different model than the one in `weights/`.  The fp16
        # graph build is deterministic - the `.rknn` file on disk and a fresh
        # build of the same ONNX give the same simulator numbers - which is why
        # this is a faithful measurement and not an approximation.
        #
        # The `.rknn` paths are still recorded, because they are what a reader
        # would go and look at, and a mismatch between the path on disk and the
        # ONNX actually built is the kind of thing that silently scores the wrong
        # model.  `--s2-onnx` defaults to the MatMul graph, so the default here
        # must follow it rather than name a file that may not be the shipped one.
        self.s1_path, self.s2_path = str(s1_path), str(s2_path)
        self.s1_onnx, self.s2_onnx = str(s1_onnx), str(s2_onnx)
        from rknn.api import RKNN
        self._RKNN = RKNN
        self._live = [None, None]

    def _runtime(self, onnx, norm, slot):
        if self._live[slot] is not None:
            return self._live[slot]
        rk = self._RKNN(verbose=False)
        cfg = {"target_platform": "rk3588", "float_dtype": "float16"}
        # Per-input-channel mean/std, exactly as `convert_to_rknn.py` sets them:
        # stage 1 normalises on the NPU so the C++ side can hand over uint8; the
        # matcher's inputs are keypoints and already-normalised descriptors and
        # must be left alone (passing them there fails outright, because RKNN
        # reads input 0 as a 512-"channel" tensor).
        if norm:
            cfg["mean_values"] = [[0, 0, 0]]
            cfg["std_values"] = [[255, 255, 255]]
        rk.config(**cfg)
        if rk.load_onnx(model=onnx) != 0:
            raise RuntimeError(f"load_onnx failed for {onnx}")
        if rk.build(do_quantization=False) != 0:
            raise RuntimeError(f"build failed for {onnx}")
        if rk.init_runtime(target=None) != 0:
            raise RuntimeError(f"init_runtime failed for {onnx}")
        self._live[slot] = rk
        return rk

    def __call__(self, rgb0_u8, rgb1_u8):
        nhwc = np.stack([rgb0_u8, rgb1_u8]).astype(np.uint8)     # (2,H,W,3)
        rk1 = self._runtime(self.s1_onnx, norm=True, slot=0)
        outs = rk1.inference(inputs=[nhwc], data_format="nhwc")
        # The simulator does not guarantee output order; it is deterministic for a
        # given graph, but identifying by shape removes the assumption entirely.
        k, d = self._pick_alike(outs)
        # Stage 2 is fed the SIMULATOR's own stage-1 outputs, not torch's.  That
        # is the whole point of chaining rather than scoring the stages in
        # isolation: it is the error a deployment actually accumulates, and a
        # per-stage harness that hands the matcher clean inputs cannot see it.
        rk2 = self._runtime(self.s2_onnx, norm=False, slot=1)
        outs2 = rk2.inference(inputs=[k[0:1].astype(np.float32),
                                      k[1:2].astype(np.float32),
                                      d[0:1].astype(np.float32),
                                      d[1:2].astype(np.float32)])
        m0, ms0 = self._pick_matcher(outs2)
        return (torch.from_numpy(k[0]), torch.from_numpy(k[1]),
                torch.from_numpy(np.asarray(m0[0], dtype=np.int64)),
                torch.from_numpy(np.asarray(ms0[0], dtype=np.float32)))

    @staticmethod
    def _pick_alike(outs):
        o = [np.asarray(x) for x in outs]
        d = next(x for x in o if x.ndim == 3 and x.shape[-1] == 128)
        k = next(x for x in o if x.ndim == 3 and x.shape[-1] == 2)
        return k.astype(np.float32), d.astype(np.float32)

    @staticmethod
    def _pick_matcher(outs):
        o = [np.asarray(x) for x in outs]
        m = next((x for x in o if "int" in str(x.dtype)), None)
        if m is None:
            # the simulator may hand back float32 for an int64 graph output; the
            # match array is the ONLY integer-valued output, so `isclose` on the
            # rounded values is the tiebreak rather than a guess about dtypes.
            m = next(x for x in o if np.allclose(x, np.round(x)))
        ms = next(x for x in o if x is not m)
        return m.astype(np.int64), ms.astype(np.float32)

    def release(self):
        # Runtimes are held in `self._live` and released here rather than after
        # every call.  Building an fp16 graph takes minutes of host CPU - far more
        # than an inference - so rebuilding per pair would make a 540-pair judge
        # take hours for no reason.  Keeping one runtime per stage and closing it
        # at the end keeps host memory bounded, which is what the per-call rebuild
        # was protecting against.
        for i, rk in enumerate(self._live):
            try:
                rk.release()
            except Exception:                          # noqa: BLE001
                pass
            self._live[i] = None

    _live = None


# --------------------------------------------------------------------------- #
# ground truth / judges
# --------------------------------------------------------------------------- #
def load_hpatches(n, size):
    """Viewpoint sequences only: illumination pairs are too easy to be a judge."""
    out = []
    for seq in sorted(paths.hpatches().iterdir()):
        if not seq.is_dir() or seq.name[0] == "i":
            continue
        ref = seq / "1.ppm"
        for q in range(2, 7):
            qf, hf = seq / f"{q}.ppm", seq / f"H_1_{q}"
            if ref.is_file() and qf.is_file() and hf.is_file():
                a = cv2.imread(str(ref), cv2.IMREAD_GRAYSCALE)
                b = cv2.imread(str(qf), cv2.IMREAD_GRAYSCALE)
                if a is None or b is None:
                    continue
                sa = (a.shape[1], a.shape[0])
                a = cv2.resize(a, (size, size), interpolation=cv2.INTER_LINEAR)
                b = cv2.resize(b, (size, size), interpolation=cv2.INTER_LINEAR)
                # The homography must be transported with the resize, exactly as
                # `eval_hpatches_sources.py` does - scoring a resized pair against
                # the ORIGINAL H is a silent, plausible-looking error.
                H = np.loadtxt(hf)
                Hn = norm_to_resized(H, sa, (size, size))
                out.append((np.stack([a] * 3, -1), np.stack([b] * 3, -1), Hn))
                break
        if len(out) >= n:
            break
    return out


def norm_to_resized(H, src_wh, dst_wh):
    """H in original pixels -> H in resized pixels (`S @ H @ S^-1`)."""
    sx, sy = dst_wh[0] / src_wh[0], dst_wh[1] / src_wh[1]
    S = np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]], dtype=np.float64)
    S_i = np.array([[1 / sx, 0, 0], [0, 1 / sy, 0], [0, 0, 1]], dtype=np.float64)
    Hn = S @ H @ S_i
    return Hn / Hn[2, 2]


def hpatches_metrics(k0, k1, m0, H, tol=(3, 5)):
    """Homography error + mAA from the matches, scored at fixed tolerances.

    The estimator is OpenCV RANSAC on the raw matches - the same estimator and
    the same thresholds as `eval_hpatches_sources.py`, so the numbers can be read
    next to the training-side table instead of only next to each other.

    `n_inl` IS ORDER-DEPENDENT; `n_inl_gt` IS NOT
    ---------------------------------------------
    See the `n_inl_gt` block below.  The RANSAC-based numbers (`n_inl`, `mAA`,
    `H_error`) are kept because they are the protocol the training-side tables
    use, but they carry an RNG-dependent offset that `cv2.setRNGSeed` does not
    remove (measured: the same graph scored 258.46 alone and 245.68 after another
    pipeline had run).  `n_inl_gt` is the number to compare arms on.
    """
    import cv2
    out = {"n_match": int((m0 >= 0).sum())}
    valid = (m0 >= 0).nonzero(as_tuple=True)[0]
    if valid.numel() < 4:
        out["H_error"] = float("nan")
        out["mAA"] = 0.0
        out["mprec@3px"] = float("nan")
        out["n_inl"] = 0
        out["n_inl_gt"] = 0
        for t in tol:
            out[f"@{t}px"] = 0.0
        return out
    p0 = k0[valid].numpy().astype(np.float64)
    p1 = k1[m0[valid].long()].numpy().astype(np.float64)

    # ------------------------------------------------------------------ #
    # DETERMINISTIC inlier count, measured against the GROUND-TRUTH homography.
    #
    # Why this exists next to the RANSAC one: `cv2.findHomography(..., RANSAC)`
    # draws from OpenCV's global RNG, so its inlier count for a given pair depends
    # on how many such calls happened BEFORE it.  Measured on this 59-pair set,
    # the SAME ONNX graph scored:
    #
    #     --pipeline onnx        alone      n_inl = 258.4576
    #     --pipeline torch onnx  onnx arm   n_inl = 245.6780
    #
    # and `cv2.setRNGSeed` does not fix it.  So the two arms in a two-pipeline run
    # are not measured on the same footing, and a 13-inlier gap was being read as
    # a model difference.
    #
    # Counting agreement with the KNOWN homography removes the estimator from the
    # measurement entirely: it is a pure function of the matches and the ground
    # truth, so it is identical no matter what ran before.  It is also the more
    # meaningful quantity for comparing two matchers - it says how many of the
    # correspondences are actually correct, rather than how many a particular
    # RANSAC run happened to agree with.
    # ------------------------------------------------------------------ #
    p0h_gt = np.concatenate([p0, np.ones((len(p0), 1))], 1) @ H.T
    p0h_gt = p0h_gt[:, :2] / p0h_gt[:, 2:3]
    err_gt = np.linalg.norm(p0h_gt - p1, axis=1)
    out["n_inl_gt"] = int((err_gt <= 3).sum())

    H_est, inl = cv2.findHomography(p0, p1, cv2.RANSAC, 3.0)
    out["n_inl"] = int(inl.sum()) if inl is not None else 0
    if H_est is None:
        out["H_error"] = float("nan")
        out["mAA"] = 0.0
        out["mprec@3px"] = float("nan")
        for t in tol:
            out[f"@{t}px"] = 0.0
        return out
    # ground-truth reprojection error of the first image's corners
    h, w = 512, 512
    corners = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
                       dtype=np.float64)
    def proj(Hm, pts):
        p = np.concatenate([pts, np.ones((len(pts), 1))], 1) @ Hm.T
        return p[:, :2] / p[:, 2:3]
    err = np.linalg.norm(proj(H_est, corners) - proj(H, corners), axis=1).mean()
    out["H_error"] = float(err)
    out["mAA"] = float(np.clip(1.0 - err / 10.0, 0.0, 1.0))
    for t in tol:
        out[f"@{t}px"] = float(err <= t)
    # Precision, and `n_inl_gt` is its numerator.  Reusing `err_gt` rather than
    # recomputing the same projection guarantees the count and the fraction can
    # never disagree - they are the same measurement presented two ways, and a
    # reader who multiplies `mprec@3px` by `n_match` must get `n_inl_gt` back.
    out["mprec@3px"] = float((err_gt <= 3).mean())
    return out


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def build_pipelines(which, args):
    out = {}
    for w in which:
        if w == "torch":
            out[w] = TorchPipeline(args.checkpoint, args.size, args.keypoints)
        elif w == "onnx":
            out[w] = OnnxPipeline(args.s1_onnx, args.s2_onnx)
        elif w.startswith("rknn"):
            # The simulator only exists on the load+build path, so the graph is
            # rebuilt from the SAME ONNX the `.rknn` on disk was built from.  Both
            # paths are passed through so the two cannot drift apart.
            out[w] = RknnPipeline(args.s1_rknn, args.s2_rknn, name=w,
                                  s1_onnx=args.s1_onnx, s2_onnx=args.s2_onnx)
        else:
            raise ValueError(f"unknown pipeline {w!r}")
    return out


def run_hpatches(pipe, pairs, verbose_every=60):
    rows = []
    t0 = time.time()
    for i, (a, b, H) in enumerate(pairs):
        try:
            k0, k1, m0, _ = pipe(a, b)
        except Exception as exc:                      # noqa: BLE001
            print(f"    !! pair {i} failed: {type(exc).__name__}: {exc}")
            continue
        r = hpatches_metrics(k0, k1, m0, H)
        rows.append(r)
        if verbose_every and (i + 1) % verbose_every == 0:
            print(f"    [{pipe.name}] {i+1}/{len(pairs)} "
                  f"({time.time() - t0:.0f}s)")
    return summarize(rows)


def summarize(rows):
    if not rows:
        return {}
    keys = [k for k in rows[0] if k != "per_pair"]
    out = {}
    for k in keys:
        v = [r[k] for r in rows if r.get(k) is not None and not
             (isinstance(r[k], float) and math.isnan(r[k]))]
        out[k] = float(np.mean(v)) if v else float("nan")
    out["n_pairs"] = len(rows)
    return out


def compare(a, b, tol):
    """`b - a` where `a` is the reference (torch)."""
    if not a or not b:
        return {}
    return {k: b[k] - a[k] for k in a
            if isinstance(a.get(k), float) and isinstance(b.get(k), float)
            and not math.isnan(a[k]) and not math.isnan(b[k])
            and k != "n_pairs"}


def fmt(v, nd=4):
    if v is None:
        return "  -  "
    if isinstance(v, float):
        return "  -  " if math.isnan(v) else f"{v:.{nd}f}"
    return str(v)


def main():
    ap = argparse.ArgumentParser()
    # Required only when the torch column is actually going to be measured: the
    # converted arms read their inputs from ONNX, and with `--ref-from` the
    # reference is loaded from a file.  Making it mandatory unconditionally would
    # force the simulator environment to carry the training dependencies it
    # deliberately does not have.
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--pipeline", nargs="*", default=["torch", "onnx", "rknn-fp16"],
                    choices=["torch", "onnx", "rknn-fp16"])
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--keypoints", type=int, default=512)
    ap.add_argument("--hpatches", type=int, default=540)
    ap.add_argument("--seed", type=int, default=0)
    # The stage-2 graph is selectable because there are two of them: upstream's
    # `Einsum` form and the `MatMul` rewrite that removes 63 off-NPU nodes.  A
    # hard-coded path would silently score whichever one happened to be in
    # `weights/`, which is how a report ends up describing a model that is not the
    # one being shipped.
    # These default to the SHIPPED pair under `weights/optimized/`.  Every
    # combination is reachable by overriding all four, and `weights/original/`
    # holds the pre-optimisation baseline (dense head + upstream `Einsum` attention).
    ap.add_argument("--s1-onnx", default=str(REPO / "weights/optimized/alike_stage_gath.onnx"))
    ap.add_argument("--s2-onnx", default=str(REPO / "weights/optimized/lightglue_stage_d7.onnx"))
    ap.add_argument("--s1-rknn", default=str(REPO / "weights/optimized/alike_stage_gath_fp.rknn"))
    ap.add_argument("--s2-rknn", default=str(REPO / "weights/optimized/lightglue_stage_d7_fp.rknn"))
    ap.add_argument("--ref-pipeline", default="torch",
                    help="which column the deltas are measured against")
    ap.add_argument("--ref-from", default=None,
                    help="reuse a previously measured reference row from this JSON "
                         "instead of recomputing it.  Needed because torch and the "
                         "RKNN toolkit live in different environments: the simulator "
                         "env has no `omegaconf` and should not need it.  The pair "
                         "counts and size are checked so a cached row cannot be "
                         "silently compared against a different sample.")
    ap.add_argument("--report", default=None, help="write a markdown report here")
    ap.add_argument("--json", default=None, help="write the raw numbers here")
    args = ap.parse_args()

    if args.size % 32:
        ap.error("--size must be a multiple of 32 (the export only supports the "
                 "no-padding downsample path)")
    if "torch" in args.pipeline and not args.checkpoint:
        ap.error("--checkpoint is required to measure the torch column "
                 "(pass --ref-from <json> to reuse a previous measurement instead)")

    hp = load_hpatches(args.hpatches, args.size)
    print(f"[e2e] {len(hp)} HPatches pairs, "
          f"{args.size}px / {args.keypoints} kpts")
    print(f"[e2e] pipelines: {', '.join(args.pipeline)}")
    print("[e2e] NOTE the RKNN arm runs the PC simulator - it validates NUMERICS "
          "only.\n      No timing below is an on-device timing claim.\n")

    results = {}
    # The torch reference can be REUSED from an earlier run rather than
    # recomputed, and there are two reasons that matters beyond saving time:
    #
    # * It is deterministic. Same checkpoint, same pairs, same size - the row is
    #   reproducible, so recomputing it adds nothing and a small difference in a
    #   re-measurement would then have to be explained.
    # * The two stages live in different environments on purpose. Torch needs
    #   glue-factory (and therefore `omegaconf`); the RKNN simulator needs the
    #   toolkit. Nothing here should require one env to carry the other's
    #   dependencies, and `--ref-from` is what makes that possible.
    #
    # What this does NOT do is accept a reference measured on DIFFERENT pairs.
    # The pair count and size are checked below, because a cached row over 200
    # pairs silently compared against a fresh run over 150 is exactly the kind of
    # mismatch that looks like a model regression.
    if args.ref_from:
        cached = json.loads(Path(args.ref_from).read_text())
        cfg = cached.get("config", {})
        for key, want in (("hpatches", len(hp)), ("size", args.size)):
            got = cfg.get(key)
            if got is not None and len(hp) and got != want:
                ap.error(f"{args.ref_from} was measured with {key}={got}, this run "
                         f"has {key}={want}; the rows are not comparable")
        ref_name = args.ref_pipeline
        if ref_name not in cached["results"]:
            ap.error(f"{args.ref_from} has no '{ref_name}' row "
                     f"(has: {sorted(cached['results'])})")
        results[ref_name] = cached["results"][ref_name]
        print(f"[e2e] torch reference REUSED from {args.ref_from} "
              f"(pipeline '{ref_name}', {cfg.get('hpatches')} pairs)")

    for name in args.pipeline:
        pipe = build_pipelines([name], args)[name]
        print(f"[e2e] === {name} ===")
        entry = {}
        if hp:
            print(f"  HPatches ({len(hp)} pairs)")
            entry["hpatches"] = run_hpatches(pipe, hp)
        pipe.release()
        results[name] = entry
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # The baseline column is the reference pipeline, whether it was just measured
    # or loaded - so the deltas are always against torch, not against whatever
    # happened to be measured first in this invocation.
    ref = results.get(args.ref_pipeline, results.get(args.pipeline[0], {}))
    cols = list(results) if args.ref_pipeline not in args.pipeline else args.pipeline

    # ---- console table -----------------------------------------------------
    # `n_inl_gt` is the deterministic inlier count and the one to compare arms
    # on; `n_inl` (RANSAC) is kept for continuity with the training-side tables
    # but is order-dependent, so it is printed after it with `H_error`.
    hp_keys = ["mAA", "@3px", "@5px", "mprec@3px", "n_inl_gt", "n_inl",
               "H_error", "n_match"]
    print("\n" + "=" * 88)
    print(f"END-TO-END ACCURACY  ({args.size}px, {args.keypoints} kpts)")
    print("=" * 88)
    if ref.get("hpatches"):
        print("\n-- hpatches --")
        print(f"{'metric':14s}" + "".join(f"{p:>14s}" for p in cols)
              + f"   delta(vs {args.ref_pipeline})")
        for k in hp_keys:
            if k not in ref["hpatches"]:
                continue
            line = f"{k:14s}" + "".join(
                f"{fmt(results[p]['hpatches'].get(k), 4):>14s}" for p in cols)
            if len(cols) > 1:
                d = compare(ref["hpatches"], results[cols[-1]]["hpatches"], 0).get(k)
                line += f"   {fmt(d, 4)}"
            print(line)

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps({
            "config": {"size": args.size, "keypoints": args.keypoints,
                       "hpatches": len(hp), "seed": args.seed,
                       # Repo-relative, never absolute: these reports are committed,
                       # and an absolute path both leaks the author's filesystem
                       # layout and cannot be resolved by anyone reading later.
                       "checkpoint": portable(args.checkpoint, "checkpoint"),
                       "s1_onnx": portable(args.s1_onnx, "onnx"),
                       "s2_onnx": portable(args.s2_onnx, "onnx"),
                       "reference_from": portable(args.ref_from, "onnx")},
            "results": results}, indent=2))
        print(f"\n[e2e] raw numbers -> {args.json}")

    if args.report:
        write_report(Path(args.report), args, results, hp_keys, cols)
        print(f"[e2e] report -> {args.report}")
    return 0


def write_report(path, args, results, hp_keys, cols):
    p = cols
    n_pairs = (results[p[0]]["hpatches"].get("n_pairs", 0)
               if "hpatches" in results[p[0]] else 0)
    lines = [
        "# End-to-end accuracy, converted pipeline vs torch",
        "",
        f"`{args.size}px`, `{args.keypoints}` keypoints, "
        f"`alike_native_gl_s1`, seed {args.seed}.",
        f"Open-source HPatches viewpoint sequences: {n_pairs} pairs.",
        "",
        "The RKNN arm runs the **PC simulator**. It validates numerics; it does not "
        "measure latency.",
        "",
    ]
    if results[p[0]].get("hpatches"):
        lines += ["## hpatches", "",
                  "| metric | " + " | ".join(p) + " | delta |",
                  "|---|" + "---|" * (len(p) + 1)]
        for k in hp_keys:
            if k not in results[p[0]]["hpatches"]:
                continue
            row = [fmt(results[q]["hpatches"].get(k), 4) for q in p]
            d = compare(results[p[0]]["hpatches"], results[p[-1]]["hpatches"], 0).get(k)
            lines.append(f"| `{k}` | " + " | ".join(row) + f" | {fmt(d, 4)} |")
        lines.append("")
    lines += [
        "## How to read this",
        "",
        "* `mAA` / `@3px` / `@5px` are HPatches homography accuracy; "
        "`mprec@3px` is the fraction of matches consistent with the ground-truth "
        "homography and `n_inl` the RANSAC inlier count. Precision alone is not "
        "enough: a matcher that emits fewer matches can win on precision and "
        "deliver fewer usable correspondences.",
        "* Compare arms on `n_inl_gt`, not `n_inl`: the RANSAC count depends on "
        "OpenCV's global RNG and therefore on call order.",
        "* A delta is only meaningful next to the pair count - see the line above "
        "the table.",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
