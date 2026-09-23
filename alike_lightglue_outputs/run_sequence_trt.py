#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run the shipped TensorRT engines over a folder of images: reference vs each other.

This is the executable counterpart of `cpp/infer_sequence.cpp` - same workflow (the
reference image is extracted ONCE and reused, then one extraction + one match per
further frame), same per-frame output, but it runs HERE instead of on the board.
The board sample needs librknn_api and an aarch64 build; the GPU does not.

Why it exists: "the sample compiles" is not "the matching works".  This produces the
match images and lets the numbers be inspected on the actual demo images, which is
what a reader of the sample would look at first.

Usage
    python run_sequence_trt.py --images "../alike_lightglue_ONNX&RKNN_deploy/image"
    python run_sequence_trt.py --images <dir> --out out/ --no-vis --limit 4
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1] / "alike_lightglue_ONNX&RKNN_deploy"
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "eval"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

SIZE = 512


def load_image(path):
    """BGR file -> the RGB 512x512 uint8 image the network takes."""
    import cv2
    im = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None:
        return None
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)


def draw(reference_rgb, f0, other_rgb, f1, matches, mscores, title, out_path):
    import cv2
    # (1, K, 2) -> (K, 2): a keypoint is `k[i, 0]`/`k[i, 1]` only after that squeeze.
    k0 = np.asarray(f0[0]).reshape(-1, 2)
    k1 = np.asarray(f1[0]).reshape(-1, 2)
    kp0 = [cv2.KeyPoint(float(k0[i, 0]), float(k0[i, 1]), 4) for i in range(len(k0))]
    kp1 = [cv2.KeyPoint(float(k1[i, 0]), float(k1[i, 1]), 4) for i in range(len(k1))]
    dm = [cv2.DMatch(i, int(matches[i] + 0.5), float(mscores[i]))
          for i in range(len(matches)) if matches[i] >= 0]
    canvas = cv2.drawMatches(
        cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2BGR), kp0,
        cv2.cvtColor(other_rgb, cv2.COLOR_RGB2BGR), kp1, dm, None,
        matchColor=(0, 220, 0), singlePointColor=(0, 0, 255),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.putText(canvas, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 255), 2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    return len(dm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default=str(_REPO / "image"))
    ap.add_argument("--out", default="sequence_out")
    ap.add_argument("--ext", default=".png")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--no-vis", action="store_true")
    ap.add_argument("--precision", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--json", default=str(_REPO / "reports" / "sequence_trt.json"),
                    help="default: the repository's reports/ directory")
    args = ap.parse_args()

    from run_accuracy import TrtPipeline, trt_engine_paths

    root = Path(args.images)
    images = sorted(p for p in root.iterdir()
                    if p.is_file() and p.suffix == args.ext)
    if len(images) < 2:
        raise SystemExit(f"need >=2 {args.ext} files in {root}, found {len(images)}")

    class Cfg:
        s1_engine = s2_engine = None
        trt_precision = args.precision
    s1, s2 = trt_engine_paths(Cfg())
    pipe = TrtPipeline(s1, s2, name=f"trt-{args.precision}",
                       precision=args.precision)
    print(f"[seq] {len(images)} images from {root} | engines: "
          f"{Path(s1).name}, {Path(s2).name}")

    # Reference: extracted ONCE, reused for every other frame - the whole point of
    # the per-image stage 1, and the same order `infer_sequence.cpp` uses.
    t0 = time.perf_counter()
    ref_rgb = load_image(images[0])
    if ref_rgb is None:
        raise SystemExit(f"cannot read reference image {images[0]}")
    k0, d0 = pipe.extract(ref_rgb)
    ref_ms = (time.perf_counter() - t0) * 1000
    # This first call also pays engine deserialisation (tens of seconds), so it is
    # NOT a per-image number: the per-frame timings below are, and they exclude it.
    print(f"[seq] reference {images[0].name}: extracted once in {ref_ms:.1f} ms "
          f"(includes first-call engine load; per-frame numbers below exclude it)")

    rows, done = [], 0
    for path in images[1:]:
        other = load_image(path)
        if other is None:
            print(f"[seq] skip unreadable {path.name}")
            continue
        t1 = time.perf_counter()
        k1, d1 = pipe.extract(other)
        extract_ms = (time.perf_counter() - t1) * 1000
        t2 = time.perf_counter()
        matches, mscores = pipe.match((k0, d0), (k1, d1))
        match_ms = (time.perf_counter() - t2) * 1000
        m = np.asarray(matches[0]).reshape(-1)
        s = np.asarray(mscores[0], dtype=np.float64).reshape(-1)
        # `k0` is (1, K, 2): the keypoint COUNT is shape[1], and reading shape[0]
        # here classified every index above 0 as out of range.
        n_kpts = np.asarray(k0).reshape(-1, 2).shape[0]
        valid = np.isfinite(m) & (m >= 0)
        n = int(valid.sum())
        mean_score = float(s[valid].mean()) if n else 0.0
        # A match index must be usable as a keypoint index; anything else is a
        # defect in the pipeline, not a weak match.
        bad = int((~valid & (m != -1)).sum()) + int((m > n_kpts - 1).sum())
        rows.append({"image": path.name, "matches": n, "mean_score": mean_score,
                     "extract_ms": round(extract_ms, 2),
                     "match_ms": round(match_ms, 2),
                     "invalid_entries": bad})
        print(f"[seq] {path.name:32s} extract {extract_ms:6.1f} ms  "
              f"match {match_ms:5.1f} ms  matches {n:3d}  mean {mean_score:.3f}"
              + (f"  !! {bad} invalid entries" if bad else ""))
        if not args.no_vis:
            drew = draw(ref_rgb, (k0, d0), other, (k1, d1), m, s,
                        f"{path.name}  {n} matches  {match_ms:.1f} ms",
                        Path(args.out) / f"{path.stem}.matches.png")
            assert drew == n, f"drew {drew} lines for {n} matches"
        done += 1
        if args.limit and done >= args.limit:
            break

    tot_extract = sum(r["extract_ms"] for r in rows)
    tot_match = sum(r["match_ms"] for r in rows)
    n = max(1, len(rows))
    print(f"\n[seq] {len(rows)} frames | mean extract {tot_extract / n:.1f} ms  "
          f"mean match {tot_match / n:.1f} ms | matches total "
          f"{sum(r['matches'] for r in rows)}")
    print(f"[seq] invalid entries across all frames: "
          f"{sum(r['invalid_entries'] for r in rows)}")

    if not args.no_vis:
        print(f"[seq] visualisations -> {args.out}/")
    if args.json:
        payload = {"images": str(root), "precision": args.precision,
                   "reference": images[0].name, "reference_ms": round(ref_ms, 2),
                   "engines": {"stage1": Path(s1).name, "stage2": Path(s2).name},
                   "frames": rows}
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2))
        print(f"[seq] -> {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
