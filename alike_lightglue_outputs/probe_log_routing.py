#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Where does the RKNN build log actually go?  One build per routing, measured.

THE PROBLEM
-----------
`capture_build_log` redirects fd 1/2 to a temp file.  That catches the
`Transpose will fallback to CPU` warnings (36 of them for the Einsum graph) but
NOT the `No lowering found ... use CustomOperatorLower` lines, which escape to
the terminal.  Two different emitters, two different sinks - so an audit that
watches only one of them is half-blind, and "0 fallbacks" can mean "the other
sink was never read".

WHY IT ESCAPES
--------------
`RKNN.__init__` calls `set_log_level_and_file_path(verbose, verbose_file)`.  That
happens at CONSTRUCTION, which is before the redirect is installed, so a stream
or fd captured there keeps pointing at the real stdout no matter what the process
does to fd 1 afterwards.

WHAT THIS PROBE COMPARES
------------------------
    A  fd redirect, RKNN built BEFORE the redirect   (the current harness)
    B  fd redirect, RKNN built INSIDE the redirect   (does construction order matter?)
    C  the toolkit's own `verbose_file=` argument    (the intended API)
    D  no redirect, `verbose_file=`                  (C's control)

For each: how many bytes land in the capture, and how many of the two message
kinds are visible in it.  The point is to find a routing that sees BOTH, since
each kind alone can report a clean graph.

Usage
    python probe_log_routing.py --onnx weights/original/lightglue_stage_einsum.onnx
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

# Analysis-only script, kept OUTSIDE the deployment repository: nothing in
# alike_lightglue_ONNX&RKNN_deploy/ imports or reads any output from it.
_REPO = Path(__file__).resolve().parents[1] / "alike_lightglue_ONNX&RKNN_deploy"
if not _REPO.is_dir():
    raise SystemExit(f"cannot find the deployment repo at {_REPO}; edit _REPO here")
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "scripts"))
HERE = Path(__file__).resolve().parent        # outputs/ - where reports go

from convert_to_rknn import capture_build_log  # noqa: E402

NO_LOWERING = "No lowering found"
WILL_FB = "will fallback to CPU"


def tally(text):
    return {NO_LOWERING: text.count(NO_LOWERING),
            WILL_FB: text.count(WILL_FB),
            "bytes": len(text)}


def configure(rk, stage):
    cfg = {"target_platform": "rk3588", "float_dtype": "float16"}
    if stage == "alike":
        cfg["mean_values"] = [[0, 0, 0]]
        cfg["std_values"] = [[255, 255, 255]]
    rk.config(**cfg)
    rk.load_onnx(model=ARGS.onnx)


def case_a(onnx, stage):
    """Current harness: construct first, then install the redirect."""
    from rknn.api import RKNN
    rk = RKNN(verbose=True)
    configure(rk, stage)
    cap = capture_build_log()
    try:
        with cap:
            rk.build(do_quantization=False)
        text = cap.read()
    finally:
        cap.cleanup()
    rk.release()
    return text


def case_b(onnx, stage):
    """Construct INSIDE the redirect, so any fd captured at construction is ours."""
    from rknn.api import RKNN
    cap = capture_build_log()
    try:
        with cap:
            rk = RKNN(verbose=True)
            configure(rk, stage)
            rk.build(do_quantization=False)
            rk.release()
        text = cap.read()
    finally:
        cap.cleanup()
    return text


def case_c(onnx, stage, also_redirect=True):
    """The toolkit's own `verbose_file` - the intended log-to-file API."""
    from rknn.api import RKNN
    fd, path = tempfile.mkstemp(prefix="rknn_vfile_", suffix=".log")
    os.close(fd)
    try:
        rk = RKNN(verbose=True, verbose_file=path)
        configure(rk, stage)
        if also_redirect:
            cap = capture_build_log()
            try:
                with cap:
                    rk.build(do_quantization=False)
                text = cap.read()
            finally:
                cap.cleanup()
        else:
            rk.build(do_quantization=False)
            text = ""
        rk.release()
        file_text = Path(path).read_text(errors="replace")
        return text, file_text
    finally:
        if os.path.exists(path):
            os.unlink(path)


def main():
    print("=== A: fd redirect, construct BEFORE redirect (current harness) ===")
    t = case_a(ARGS.onnx, ARGS.stage)
    print("   ", tally(t))

    print("=== B: fd redirect, construct INSIDE redirect ===")
    t = case_b(ARGS.onnx, ARGS.stage)
    print("   ", tally(t))

    print("=== C: verbose_file= + fd redirect ===")
    cap_text, file_text = case_c(ARGS.onnx, ARGS.stage, also_redirect=True)
    print("    captured-by-fd :", tally(cap_text))
    print("    verbose_file   :", tally(file_text))

    print("=== D: verbose_file= only, no redirect ===")
    _, file_text = case_c(ARGS.onnx, ARGS.stage, also_redirect=False)
    print("    verbose_file   :", tally(file_text))
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--stage", default="lightglue", choices=["alike", "lightglue"])
    ARGS = ap.parse_args()
    sys.exit(main())
