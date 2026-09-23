"""Probe: does truncating LightGlue's layer lists actually change the output?

This exists because the ablation sweep reported `layers 9->8` and `layers 9->7`
with byte-identical matches, and there are two possible reasons:

  (a) the model changed and is insensitive to depth, or
  (b) the model did not change, and the sweep was measuring the baseline twice.

(b) is far more likely given identical output, and it would invalidate every
depth row. This prints the raw index and score deltas - before any thresholding -
so the two can be told apart. Run it directly:

    python probe_depth.py
"""
import copy
import sys
from pathlib import Path

import torch

# This script lives OUTSIDE the deployment repository: it is analysis only and
# nothing in alike_lightglue_ONNX&RKNN_deploy/ imports it.  It needs that repository
# on sys.path for `model` and `paths`, and it writes its output here, next to
# itself, so the repository never depends on any report it produces.
_REPO = Path(__file__).resolve().parents[1] / "alike_lightglue_ONNX&RKNN_deploy"
if not _REPO.is_dir():
    raise SystemExit(f"cannot find the deployment repo at {_REPO}; edit _REPO at the top of this file")
sys.path.insert(0, str(_REPO))
HERE = Path(__file__).resolve().parent        # outputs/ - where reports go
import paths  # noqa: E402
from model import load_lightglue_stage  # noqa: E402

CKPT = str(paths.TRAINED_CKPT)


def main():
    base = load_lightglue_stage(CKPT).eval()
    torch.manual_seed(0)
    for nk in (96, 512):
        k = torch.rand(1, nk, 2) * 512
        d = torch.randn(1, nk, 128)
        d = d / d.norm(dim=-1, keepdim=True)
        with torch.no_grad():
            lb, sb = base(k, k.clone(), d, d.clone())
        print(f"--- {nk} keypoints (baseline matches={int((lb >= 0).sum())}) ---")
        for depth in (9, 8, 7, 6):
            mm = copy.deepcopy(base)
            lists = (mm.matcher.transformers, mm.matcher.log_assignment,
                     mm.matcher.token_confidence)
            before = [len(x) for x in lists]
            for lst in lists:
                del lst[depth:]
            mm.matcher.conf.n_layers = depth
            after = [len(x) for x in lists]
            with torch.no_grad():
                lo, so = mm(k, k.clone(), d, d.clone())
            print(f"  depth={depth}: list_len {before}->{after}  "
                  f"idx_diff={int((lo != lb).sum()):5d}/{lo.numel():5d}  "
                  f"max_abs_dscore={(so - sb).abs().max():.6f}  "
                  f"matches={int((lo >= 0).sum())}")


if __name__ == "__main__":
    main()
