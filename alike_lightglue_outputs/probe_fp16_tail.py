#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Where is each fp16 backend's keypoint tail?  Same 8 pairs, same statistic.

The published fp16 numbers are not comparable with each other:

* RKNN's came from `reports/accuracy.json` - 8 real HPatches pairs, nearest-neighbour
  keypoint distance against torch, one image per pair.
* TensorRT's came from `convert_to_trt.py --ref`, whose input is the EXPORT's dump -
  a random-noise image, kept because it is the tensor the graph was traced with.
  Noise makes near-tie keypoints common, so its tail is not the real-image tail.
* The timing report's TRT row compares on ONE real image against the ONNX CPU
  session, which is a third (and again different) pairing.

This probe removes the difference: one graph family (the dense `_bil` one that
`accuracy.json` measured), one reference (torch), the same 8 pairs, the same
order-matched nearest-neighbour statistic.  Whatever the tails are, they are
comparable here.

Usage
    python probe_fp16_tail.py                # torch vs ONNX, TRT fp16, TRT fp32
    python probe_fp16_tail.py --pairs 4
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1] / "alike_lightglue_ONNX&RKNN_deploy"
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

SIZE = 512
DEFAULT_ENGINES = {
    # the dense graph, i.e. the one accuracy.json's RKNN column measured
    "trt_fp16": _REPO / "weights/original/alike_stage_bil_fp16.engine",
    "trt_fp32": Path("/tmp/alike_stage_bil_fp32.engine"),
}


def stats(k_ref, k_test):
    """Order-matched keypoint disagreement for one view.

    Nearest-neighbour matching, because `topk` returns score-ordered points and a
    backend may legitimately emit the same set in a different order.
    """
    a = k_ref.astype(np.float64)[0]
    b = k_test.astype(np.float64)[0]
    d2 = ((b[:, None, :] - a[None, :, :]) ** 2).sum(-1)
    j = d2.argmin(-1)
    step = np.sqrt(d2[np.arange(len(j)), j])
    return {"median_px": float(np.median(step)), "max_px": float(step.max()),
            "over_1px": int((step > 1.0).sum()), "over_0p5px": int((step > 0.5).sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=8)
    ap.add_argument("--out", default=str(_REPO / "reports" / "fp16_tail.json"),
                    help="default: the repository's reports/ directory")
    args = ap.parse_args()

    import check_accuracy as ca          # same pairs, same preprocessing
    import onnxruntime as ort
    from bench_onnx_speed import TrtRunner
    from model import load_alike_stage
    import torch

    onnx_path = _REPO / "weights/original/alike_stage_bil.onnx"
    model = load_alike_stage(str(_REPO / "weights/checkpoints/alike_native_gl_s1_9L.tar"),
                             top_k=512, descriptor_interp="bilinear",
                             gathered_head=False, batch=1, size=SIZE,
                             verbose=False).eval()
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])

    engines = {}
    for name, path in DEFAULT_ENGINES.items():
        if path.is_file():
            engines[name] = path
        else:
            print(f"[w] no engine for {name} ({path}); skipping that column")

    # Same pairs and same first-image-of-each-pair as `accuracy.json`: the frozen
    # RKNN column is read for these exact files, one image per sequence.
    pairs = ca.load_hpatches_pairs(args.pairs)
    rows = {}
    print(f"{'pair':22s} {'backend':10s} {'median px':>10s} {'max px':>10s} "
          f"{'>0.5px':>7s} {'>1px':>5s}")
    for ref, _query, _homography in pairs:
        name = f"{ref.parent.name}/{ref.name}"
        rgb = ca.read_gray_img(ref, SIZE)
        img = ca.as_tensor(rgb)
        with torch.no_grad():
            k_ref = model(img)[0].numpy()
        # ONNX CPU, fp32 - the column accuracy.json also has
        outs = sess.run(None, {"image": img.numpy()})
        a_col = {"onnx": outs[0]}
        for ename, epath in engines.items():
            feeds = {"image": img.numpy()}
            runner = TrtRunner(epath, feeds)
            got = runner.run()
            for k in got:
                if got[k].shape == (1, 512, 2):
                    a_col[ename] = got[k]
        for bname, k_test in a_col.items():
            s = stats(k_ref, k_test)
            rows.setdefault(bname, []).append(s)
            print(f"{name:22s} {bname:10s} {s['median_px']:10.4f} {s['max_px']:10.3f} "
                  f"{s['over_0p5px']:7d} {s['over_1px']:5d}")

    print(f"\n{'backend':10s} {'median px (mean)':>17s} {'max over pairs':>15s} "
          f"{'pairs with >1px':>16s} {'kp >1px total':>14s}")
    summary = {}
    for bname, rs in rows.items():
        med = float(np.mean([r["median_px"] for r in rs]))
        mx = max(r["max_px"] for r in rs)
        n_pairs = sum(1 for r in rs if r["over_1px"])
        n_kp = sum(r["over_1px"] for r in rs)
        summary[bname] = {"median_px_mean_over_pairs": med, "max_px": mx,
                          "pairs_with_over_1px": n_pairs,
                          "keypoints_over_1px_total": n_kp, "per_pair": rs}
        print(f"{bname:10s} {med:17.4f} {mx:15.3f} {n_pairs:16d} {n_kp:14d}")

    rknn = json.load(open(_REPO / "reports/accuracy.json"))
    med = float(np.mean([r["kp_rknn_median_px"] for r in rknn["pairs"]]))
    mx = max(r["kp_rknn_max_px"] for r in rknn["pairs"])
    n_pairs = sum(1 for r in rknn["pairs"] if r["kp_rknn_over_1px"])
    n_kp = sum(r["kp_rknn_over_1px"] for r in rknn["pairs"])
    summary["rknn_fp16 (from accuracy.json, same pairs)"] = {
        "median_px_mean_over_pairs": med, "max_px": mx,
        "pairs_with_over_1px": n_pairs, "keypoints_over_1px_total": n_kp}
    print(f"{'rknn_fp16':10s} {med:17.4f} {mx:15.3f} {n_pairs:16d} {n_kp:14d}"
          f"   <- read from accuracy.json, same 8 pairs")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n[probe] -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
