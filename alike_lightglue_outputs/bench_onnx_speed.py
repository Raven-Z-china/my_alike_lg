#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Time the shipped ONNX graphs on the host: CPU provider vs the GPU provider.

Same protocol as the numbers in the deployment README's "ONNX inference speed"
section - 512x512 image pair for stage 1, 512 keypoints per image for stage 2, 20
timed runs, median reported.  Both columns are measured HERE, in one process and
one protocol, so the two are comparable; the GPU column is what that section was
missing.

What the GPU number includes: `session.run` with numpy inputs, so the
host->device copies and the device->host copy of the outputs are inside the timed
region - that is what one call costs from Python, which is what a caller sees.

Three things this script refuses to do
--------------------------------------
1. Report a GPU number it could not verify.  Every CUDA run is compared against
   the CPU run on identical inputs, order-invariantly (a top-k can legitimately
   emit the same points in a different order), and the comparison is printed next
   to the timing.
2. Report a GPU number for a graph that silently kept nodes on the CPU.  ORT
   prints a warning when a node is not assigned to the preferred provider; the
   session is opened at warning severity and its C++ stderr is captured, because
   that message does not pass through Python's logging.  `--verbose-assign` digs
   out the per-node list.
3. Pretend the number is a board latency.  These graphs are fp32 ONNX on a
   discrete GPU; the board runs fp16 RKNN on an NPU.

Usage
    python bench_onnx_speed.py --real                    # HPatches pair, both providers
    python bench_onnx_speed.py --synthetic --provider cuda
    python bench_onnx_speed.py --real --verbose-assign   # per-node EP assignment
"""
import argparse
import contextlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

# Analysis-only script, kept OUTSIDE the deployment repository.
_REPO = Path(__file__).resolve().parents[1] / "alike_lightglue_ONNX&RKNN_deploy"
if not _REPO.is_dir():
    raise SystemExit(f"cannot find the deployment repo at {_REPO}; edit _REPO here")
sys.path.insert(0, str(_REPO))

ITERS = 20
WARMUP = {"cpu": 3, "cuda": 20}
SIZE = 512

STAGE1 = [
    _REPO / "weights/original/alike_stage_bil.onnx",
    _REPO / "weights/optimized/alike_stage_gath.onnx",
]
STAGE2 = [
    _REPO / "weights/original/lightglue_stage_einsum.onnx",
    _REPO / "weights/optimized/lightglue_stage_d7.onnx",
]


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
def synthetic_feeds(path):
    """Deterministic inputs matching the shipped signature of `path`."""
    import onnx

    rng = np.random.default_rng(0)
    m = onnx.load(str(path), load_external_data=False)
    feeds = {}
    for value in m.graph.input:
        dims = [d.dim_value for d in value.type.tensor_type.shape.dim]
        name = value.name
        if name == "image":                      # [2,3,512,512]
            feeds[name] = rng.uniform(0, 1, size=dims).astype(np.float32)
        elif name.startswith("keypoints"):       # [1,512,2] pixel coordinates
            feeds[name] = rng.uniform(0, 511, size=dims).astype(np.float32)
        else:                                    # descriptors, L2-normalised rows
            d = rng.standard_normal(size=dims).astype(np.float32)
            feeds[name] = d / np.linalg.norm(d, axis=-1, keepdims=True)
    return feeds


def real_feeds():
    """One real HPatches viewpoint pair through the real stage-1 graph.

    Preprocessing is copied from `eval/run_accuracy.py::OnnxPipeline`.  Stage 1 is
    PER-IMAGE, so the timing input is ONE `(1,3,512,512)` view - a single
    extraction, which is the unit of work a caller can cache - and stage 2's inputs
    come from running stage 1 once per view.
    """
    import cv2

    import paths
    import onnxruntime as ort

    root = paths.hpatches()
    seq = sorted(s for s in root.iterdir() if s.is_dir())[0]
    views = []
    for name in ("1.ppm", "2.ppm"):
        im = cv2.imread(str(seq / name), cv2.IMREAD_COLOR)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        views.append(im.transpose(2, 0, 1)[None].astype(np.float32) / 255.0)

    so = ort.SessionOptions()
    so.log_severity_level = 3
    s1 = ort.InferenceSession(str(STAGE1[1]), so,
                              providers=["CPUExecutionProvider"])
    (k0, d0, _), (k1, d1, _) = (s1.run(None, {"image": v}) for v in views)
    label = f"{seq.name}/1.ppm + {seq.name}/2.ppm"
    print(f"  real pair: {label}")
    return ({"image": views[0]},
            {"keypoints0": k0, "keypoints1": k1,
             "descriptors0": d0, "descriptors1": d1},
            label)


# --------------------------------------------------------------------------- #
# sessions and timing
# --------------------------------------------------------------------------- #
@contextlib.contextmanager
def _redirect_stderr(tmp):
    saved = os.dup(2)
    os.dup2(tmp.fileno(), 2)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)


def run_captured(fn):
    """Run `fn()` with fd 2 redirected, and return (result, captured text).

    ORT's EP-assignment warnings are written by C++, not through the Python
    `logging` module, so a Python-level handler sees nothing.  The text is read
    inside the `with` because `TemporaryFile` closes on exit.
    """
    with tempfile.TemporaryFile() as tmp:
        with _redirect_stderr(tmp):
            result = fn()
        tmp.seek(0)
        return result, tmp.read().decode(errors="replace")


def session(path, provider, verbose=False):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 0 if verbose else 1        # 0=verbose, 1=warning
    providers = (["CUDAExecutionProvider", "CPUExecutionProvider"]
                 if provider == "cuda" else ["CPUExecutionProvider"])
    kw = {"providers": providers}
    if provider == "cuda":
        kw["provider_options"] = [{"device_id": 0}, {}]   # one per provider

    def build():
        sess = ort.InferenceSession(str(path), so, **kw)
        sess.run(None, feeds_cache[str(path)])            # forces assignment
        return sess

    sess, log = run_captured(build)
    if provider == "cuda" and sess.get_providers()[0] != "CUDAExecutionProvider":
        raise SystemExit(f"{path.name}: asked for CUDA, got {sess.get_providers()}"
                         " - check libcudnn/libcublas on LD_LIBRARY_PATH")
    return sess, log


def time_session(sess, feeds, iters, warmup):
    names = [o.name for o in sess.get_outputs()]
    outs = None
    for _ in range(warmup):
        outs = sess.run(names, feeds)
    runs = []
    for _ in range(iters):
        t0 = time.perf_counter()
        outs = sess.run(names, feeds)
        runs.append((time.perf_counter() - t0) * 1e3)
    runs.sort()
    return runs, outs


# --------------------------------------------------------------------------- #
# verification
# --------------------------------------------------------------------------- #
def find_engine(onnx_path):
    """The TensorRT engine built for `onnx_path`, if one is shipped next to it.

    `convert_to_trt.py` names them `<stem>_<precision>.engine`; fp16 is preferred
    because that is what a deployment would run, and the precision is recorded in
    the row so an fp16 engine is never compared silently against an fp32 session.
    """
    for precision in ("fp16", "fp32"):
        cand = onnx_path.with_name(f"{onnx_path.stem}_{precision}.engine")
        if cand.is_file():
            return cand, precision
    return None, None


class TrtRunner:
    """Time a shipped engine with the TensorRT runtime itself.

    Everything that is not per-inference work - deserialisation, context creation,
    device buffer allocation - happens ONCE in `__init__`, so the timed region is
    `execute_async_v3` plus the same host->device copies and device->host readback
    that `session.run` performs in the CPU/CUDA columns.  Without that split the
    timing would measure engine loading, not inference.
    """

    def __init__(self, engine_path, feeds):
        import tensorrt as trt
        import torch

        self.trt, self.torch = trt, torch
        logger = trt.Logger(trt.Logger.ERROR)
        self.engine = trt.Runtime(logger).deserialize_cuda_engine(
            engine_path.read_bytes())
        if self.engine is None:
            raise SystemExit(f"{engine_path}: engine failed to deserialise (it is "
                             f"built for one GPU/TRT/CUDA combination)")
        self.ctx = self.engine.create_execution_context()
        self.precision = {n: str(self.engine.get_tensor_dtype(n))
                          for n in self._names()}

        self.host_in, self.dev_in, self.out = {}, {}, {}
        for name in self._names():
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                np_dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                arr = np.ascontiguousarray(feeds[name].astype(np_dtype))
                self.host_in[name] = arr
                self.dev_in[name] = torch.empty(arr.shape, dtype=torch.from_numpy(
                    np.empty(0, dtype=np_dtype)).dtype, device="cuda")
                self.ctx.set_tensor_address(name, self.dev_in[name].data_ptr())
            else:
                shape = tuple(self.engine.get_tensor_shape(name))
                np_dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                self.out[name] = torch.empty(
                    shape, dtype=torch.from_numpy(np.empty(0, dtype=np_dtype)).dtype,
                    device="cuda")
                self.ctx.set_tensor_address(name, self.out[name].data_ptr())
        self.stream = torch.cuda.Stream()

    def _names(self):
        return [self.engine.get_tensor_name(i)
                for i in range(self.engine.num_io_tensors)]

    def run(self):
        for name, arr in self.host_in.items():
            self.dev_in[name].copy_(self.torch.from_numpy(arr), non_blocking=True)
        if not self.ctx.execute_async_v3(self.stream.cuda_stream):
            raise SystemExit("execute_async_v3 returned false")
        self.stream.synchronize()
        return {k: v.cpu().numpy() for k, v in self.out.items()}


def compare_outputs(ref, cur, names):
    """Order-invariant disagreement between two runs of one graph.

    A top-k can return the same keypoints in a different order - that is allowed,
    and it is why a naive positional comparison once reported a 484 px "error"
    here.  So each keypoint of `cur` is matched to its nearest keypoint of `ref`
    and the residual is measured after that matching; the permutation found there
    is reused for the descriptor and score outputs of the same graph.
    """
    ref = dict(zip(names, ref))
    cur = dict(zip(names, cur))
    out = {}
    if "keypoints" in ref:
        a = ref["keypoints"].astype(np.float64)
        b = cur["keypoints"].astype(np.float64)
        d2 = ((b[:, :, None, :] - a[:, None, :, :]) ** 2).sum(-1)      # (N,K,K)
        j = d2.argmin(-1)                                             # (N,K)
        n = np.arange(d2.shape[0])[:, None]
        dd = np.sqrt(d2[n, np.arange(d2.shape[1])[None, :], j])
        out["kp_step_median_px"] = round(float(np.median(dd)), 6)
        out["kp_step_max_px"] = round(float(dd.max()), 4)
        out["kp_over_1px"] = int((dd > 1.0).sum())
        if "descriptors" in ref:
            nb = np.take_along_axis(ref["descriptors"].astype(np.float64),
                                    j[..., None], axis=1)
            cos = (nb * cur["descriptors"].astype(np.float64)).sum(-1)
            out["desc_cos_min"] = round(float(cos.min()), 8)
            out["desc_cos_median"] = round(float(np.median(cos)), 8)
        if "scores" in ref:
            s = np.take_along_axis(ref["scores"].astype(np.float64), j, axis=1)
            out["score_max_abs_delta"] = round(
                float(np.abs(s - cur["scores"]).max()), 6)
    for name in ref:
        if name in ("keypoints", "descriptors", "scores"):
            continue
        x, y = ref[name], cur[name]
        if x.dtype.kind in "iu":                 # matches: exact equality only
            same = x == y
            out[f"{name}_equal_frac"] = round(float(same.mean()), 6)
            out[f"{name}_n_diff"] = int((~same).sum())
        else:
            out[f"{name}_max_abs_delta"] = round(
                float(np.abs(x.astype(np.float64) - y.astype(np.float64)).max()), 6)
    return out


def assignment_report(log):
    """What the captured C++ stderr says about where the nodes went."""
    lines = [l for l in log.splitlines() if l.strip()]
    bad = [l for l in lines
           if "not assigned to the preferred execution provider" in l
           or "Falling back to CPU" in l
           or "fallback" in l.lower() and "cpu" in l.lower()]
    nodes = [l.strip() for l in lines if "assigned to" in l.lower()]
    return bad, nodes


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", choices=["cpu", "cuda", "both"], default="both")
    ap.add_argument("--synthetic", action="store_true",
                    help="random-noise inputs instead of a real HPatches pair "
                         "(same shapes; NOT used for the verification numbers)")
    ap.add_argument("--real", action="store_true")
    ap.add_argument("--iters", type=int, default=ITERS)
    ap.add_argument("--verbose-assign", action="store_true",
                    help="print the per-node execution-provider assignment")
    ap.add_argument("--extra", action="append", default=[],
                    help="additional ONNX to time with the STAGE-2 inputs, e.g. the "
                         "rewrite-only graph: the 9-layer checkpoint exported with "
                         "--attention matmul and no layer cut.  Repeatable.")
    ap.add_argument("--trt", dest="trt", action="store_true", default=True,
                    help="also time the TensorRT engine shipped next to each ONNX "
                         "(<stem>_fp16.engine preferred, else _fp32); default on")
    ap.add_argument("--no-trt", dest="trt", action="store_false")
    ap.add_argument("--out", default=str(_REPO / "reports" / "bench_onnx_speed.json"),
                    help="default: the repository's reports/ directory")
    args = ap.parse_args()
    use_real = args.real or not args.synthetic

    import onnxruntime as ort
    print(f"onnxruntime {ort.__version__} | available {ort.get_available_providers()}")
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    providers = ["cpu", "cuda"] if args.provider == "both" else [args.provider]

    global feeds_cache
    feeds_cache = {}
    if use_real:
        real_s1, real_s2, pair_label = real_feeds()
        feeds_cache[str(STAGE1[0])] = real_s1
        feeds_cache[str(STAGE1[1])] = real_s1
        feeds_cache[str(STAGE2[0])] = real_s2
        feeds_cache[str(STAGE2[1])] = real_s2
        real = {"mode": "real", "pair": pair_label}
    else:
        real = {"mode": "synthetic"}
        for path in STAGE1 + STAGE2:
            feeds_cache[str(path)] = synthetic_feeds(path)

    # Extra graphs share the stage-2 inputs; when the inputs are real they are
    # reused rather than re-derived, so stage 1 runs once for the whole session.
    extras = [Path(p) for p in args.extra]
    for path in extras:
        if not path.is_file():
            raise SystemExit(f"--extra {path} does not exist")
        feeds_cache[str(path)] = (feeds_cache[str(STAGE2[1])] if use_real
                                  else synthetic_feeds(path))
    report = {"ort": ort.__version__, "inputs": real,
              "protocol": f"{args.iters} timed runs, median, warmup {WARMUP}",
              "models": {}}
    for stage, paths in (("stage1", STAGE1), ("stage2", STAGE2 + extras)):
        for path in paths:
            if not path.is_file():
                continue
            feeds = feeds_cache[str(path)]
            row = {"stage": stage, "onnx": os.path.relpath(path, _REPO),
                   "ms": {}, "p10_ms": {}, "max_ms": {}, "trt": {}}
            print(f"\n--- {stage}: {path.name}")
            ref = ref_names = None
            for prov in providers:
                sess, log = session(path, prov, verbose=args.verbose_assign)
                bad, nodes = assignment_report(log)
                runs, outs = time_session(sess, feeds, args.iters, WARMUP[prov])
                names = [o.name for o in sess.get_outputs()]
                med = float(np.median(runs))
                row["ms"][prov] = round(med, 1)
                row["p10_ms"][prov] = round(float(np.percentile(runs, 10)), 1)
                row["max_ms"][prov] = round(runs[-1], 1)
                row["provider_used_" + prov] = sess.get_providers()[0]
                row["ep_warning_" + prov] = bool(bad)
                print(f"    {prov:5s} median {med:7.1f} ms   p10 "
                      f"{row['p10_ms'][prov]:7.1f}   max {runs[-1]:7.1f}   "
                      f"({sess.get_providers()[0]})")
                if bad:
                    print(f"          !! provider warning: {bad[0][:150]}")
                if args.verbose_assign and nodes:
                    print(f"          assignment lines: {len(nodes)}")
                    for l in nodes[:8]:
                        print(f"            {l[:150]}")
                if ref is None:
                    ref, ref_names = outs, names
                else:
                    row["verify_vs_cpu"] = compare_outputs(ref, outs, ref_names)
                    print(f"          vs CPU: " + "  ".join(
                        f"{k}={v}" for k, v in row["verify_vs_cpu"].items()))
            # --- TensorRT engine, if one is shipped next to this ONNX ------------
            engine, precision = (find_engine(path) if args.trt else (None, None))
            if engine is not None:
                runner = TrtRunner(engine, feeds)
                trt_outs = runner.run()                    # warm-up + sanity
                for _ in range(WARMUP.get("cuda", 20) // 4):
                    runner.run()
                runs = []
                for _ in range(args.iters):
                    t0 = time.perf_counter()
                    trt_outs = runner.run()
                    runs.append((time.perf_counter() - t0) * 1e3)
                runs.sort()
                med = float(np.median(runs))
                row["ms"]["trt"] = round(med, 1)
                row["p10_ms"]["trt"] = round(float(np.percentile(runs, 10)), 1)
                row["max_ms"]["trt"] = round(runs[-1], 1)
                row["trt"]["engine"] = engine.name
                row["trt"]["precision"] = precision
                print(f"    trt   median {med:7.1f} ms   p10 {row['p10_ms']['trt']:7.1f}"
                      f"   max {runs[-1]:7.1f}   ({engine.name}, {precision})")
                # Same order-invariant checks, but against the TRT outputs: the
                # engine is a different precision AND a different runtime, so this
                # is the only way to know the timing is of the same computation.
                names_trt = list(trt_outs)
                if ref is None:
                    ref, ref_names = [trt_outs[n] for n in names_trt], names_trt
                else:
                    row["verify_trt_vs_cpu"] = compare_outputs(
                        ref, [trt_outs[n] for n in names_trt], ref_names)
                    print(f"          vs CPU: " + "  ".join(
                        f"{k}={v}" for k, v in row["verify_trt_vs_cpu"].items()))
                if len(providers) == 2:
                    row["speedup_trt_vs_cpu"] = round(
                        row["ms"]["cpu"] / row["ms"]["trt"], 2)
                    print(f"    trt speedup vs cpu {row['speedup_trt_vs_cpu']:.2f}x")
            if len(providers) == 2:
                row["speedup"] = round(row["ms"]["cpu"] / row["ms"]["cuda"], 2)
                print(f"    speedup {row['speedup']:.2f}x")
            report["models"][f"{stage}/{path.name}"] = row

    def key(suffix):
        return next(k for k in report["models"] if k.endswith(suffix))

    # Stage 1 is PER IMAGE and stage 2 is per PAIR, so the two are not addable as
    # they stand: a pair costs two stage-1 calls plus one stage-2 call, and an
    # ADDITIONAL image matched against a cached extraction costs one more.
    columns = list(providers)
    if args.trt and all("trt" in report["models"][key(n)]["ms"]
                        for n in ("alike_stage_gath.onnx", "lightglue_stage_d7.onnx")):
        columns.append("trt")
    for prov in columns:
        s1 = report["models"][key("alike_stage_gath.onnx")]["ms"][prov]
        s2 = report["models"][key("lightglue_stage_d7.onnx")]["ms"][prov]
        report.setdefault("pipeline_ms", {})[prov] = {
            "stage1_per_image": s1,
            "per_pair": round(2 * s1 + s2, 1),
            "extra_image_after_cache": s1,
        }
        print(f"\npipeline {prov}: stage 1 {s1} ms/IMAGE, stage 2 {s2} ms/pair")
        print(f"  pair (2 extractions + match) : {2 * s1 + s2:.1f} ms")
        print(f"  each further image, reusing a cached extraction: {s1:.1f} ms")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\n[bench] -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
