#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Which of the two rewrites makes `load_onnx` fail?

THE FAILURE
-----------
Adding the `ScatterElements` -> `Concat` rewrite makes the toolkit refuse the
graph outright, before any conversion:

    File "rknn/api/ir_graph.py", line 2131, in IRGraph.convert_to_fp32
    AttributeError: 'numpy.ndarray' object has no attribute 'data_type'

The attention rewrite alone converts and runs.  So one of the two is responsible
and the fix depends on which: if it is the `Concat` form, the substitution needs
a different shape; if it is something the EXPORT emits around it (an extra
initializer, a differently-typed constant), the fix may be in the export rather
than in the rewrite.

This exports each variant in turn and asks the toolkit to load it.  Three cases,
so the result is attributable to one patch:

    base    neither patch                       (expected: loads)
    attn    attention only                      (expected: loads)
    assign  assignment only                     (the suspect)
    both    both patches                        (the deployed form)

Usage
    python probe_load_bisect.py --checkpoint <ckpt> --out-dir /tmp/bisect
"""
import argparse
import sys
from pathlib import Path

import numpy as np

# Analysis-only script, kept OUTSIDE the deployment repository: nothing in
# alike_lightglue_ONNX&RKNN_deploy/ imports or reads any output from it.
_REPO = Path(__file__).resolve().parents[1] / "alike_lightglue_ONNX&RKNN_deploy"
if not _REPO.is_dir():
    raise SystemExit(f"cannot find the deployment repo at {_REPO}; edit _REPO here")
sys.path.insert(0, str(_REPO))
HERE = Path(__file__).resolve().parent        # outputs/ - where reports go
REPO = _REPO


# (label, attention patch, assignment patch, matmul sub, concat sub).  The last
# two switch the assignment rewrite's two changes INDEPENDENTLY: applying them
# together makes `load_onnx` fail, so the two must be separable to find out which
# one the toolkit cannot parse.
VARIANTS = [
    ("base", False, False, False, False, "derive"),
    ("attn", True, False, False, False, "derive"),
    ("assign", False, True, True, True, "declare"),
    ("assign-mm", False, True, True, False, "derive"),
    ("assign-cc", False, True, False, True, "declare"),
    ("assign-cc2", False, True, False, True, "derive"),
    ("both", True, True, True, True, "derive"),
    ("both-decl", True, True, True, True, "declare"),
]


def phase_export(args, out_dir):
    """Export every variant.  Needs torch + glue-factory (the `alike` env)."""
    import collections

    import onnx
    import torch

    from model import load_lightglue_stage

    # Reference dumps are scratch artifacts (refs/ is git-ignored, not shipped).
    # Regenerate with:  python scripts/export_onnx.py --checkpoint <ckpt> \
    #     --no-gathered-head --out weights/original/alike_stage_bil.onnx \
    #     --ref refs/alike_stage_bil_ref.npz
    ref = REPO / "refs/alike_stage_bil_ref.npz"
    if not ref.is_file():
        raise SystemExit(f"{ref} missing - regenerate it with export_onnx.py "
                         f"(see the comment above this check)")
    z = np.load(ref)
    k = torch.from_numpy(z["keypoints"]).float()
    d = torch.from_numpy(z["descriptors"]).float()
    k0, k1, d0, d1 = k[0:1], k[1:2], d[0:1], d[1:2]

    for name, do_attn, do_assign, sub_mm, sub_cc, zmode in VARIANTS:
        stage = load_lightglue_stage(args.checkpoint,
                                     image_size=(args.size, args.size)).eval()
        matcher = stage.matcher
        if do_attn:
            from ablate_attention import patch_cross_block
            for block in matcher.transformers:
                patch_cross_block(block.cross_attn)
        if do_assign:
            import model.matchers.lightglue as lg
            from ablate_attention import patch_match_assignment
            for la in matcher.log_assignment:
                patch_match_assignment(la, orig_fn=lg.sigmoid_log_double_softmax,
                                       matmul=sub_mm, concat=sub_cc,
                                       zero_mode=zmode)

        path = out_dir / f"lg_{name}.onnx"
        # Export the STAGE, not the matcher.  `LightGlue.forward` takes a dict, so
        # handing it four positional tensors fails with
        # `TypeError: too many positional arguments` from inside torch.export -
        # an error that names neither the model nor the arity mismatch.  The
        # stage wrapper exists precisely to adapt tensor I/O to that dict, and
        # `export_onnx.py` exports it for the same reason.
        with torch.no_grad():
            torch.onnx.export(
                stage, (k0, k1, d0, d1), str(path),
                input_names=["keypoints0", "keypoints1",
                             "descriptors0", "descriptors1"],
                output_names=["matches0", "mscores0"],
                opset_version=args.opset, do_constant_folding=True,
                dynamic_axes=None, training=torch.onnx.TrainingMode.EVAL)
        m = onnx.load(str(path))
        c = collections.Counter(n.op_type for n in m.graph.node)
        print(f"{name:10s} {len(m.graph.node):6d} {c.get('Einsum',0):7d} "
              f"{c.get('ScatterElements',0)+c.get('ScatterND',0):8d} "
              f"{c.get('Concat',0):7d}  exported")
    return 0


def phase_load(args, out_dir):
    """Try to load every variant.  Needs the toolkit only (the `rknn` env).

    Kept separate because the two halves live in different environments:
    exporting needs torch and glue-factory, loading needs rknn-toolkit2, and
    neither env should be forced to carry the other's dependencies.  That is the
    same reason `run_accuracy.py` has `--ref-from`.
    """
    from rknn.api import RKNN

    print(f"\n{'variant':10s}  load_onnx")
    for name, _, _, _, _, _ in VARIANTS:
        path = out_dir / f"lg_{name}.onnx"
        rk = RKNN(verbose=False)
        rk.config(target_platform="rk3588", float_dtype="float16")
        try:
            rc = rk.load_onnx(model=str(path))
            status = "OK" if rc == 0 else f"rc={rc}"
        except Exception as exc:                       # noqa: BLE001
            # The toolkit wraps its real error in a traceback string; the LAST
            # line is the one that names the actual fault.
            status = f"{type(exc).__name__}: {str(exc).strip().splitlines()[-1][:70]}"
        finally:
            rk.release()
        print(f"{name:10s}  {status}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["export", "load"], required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out-dir", default="/tmp/bisect")
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--opset", type=int, default=13)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.phase == "export":
        if not args.checkpoint:
            ap.error("--phase export needs --checkpoint")
        return phase_export(args, out_dir)
    return phase_load(args, out_dir)


if __name__ == "__main__":
    sys.exit(main())
