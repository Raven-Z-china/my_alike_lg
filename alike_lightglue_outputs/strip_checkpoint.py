#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Drop the parts of a training checkpoint that no path in the deployment reads.

Maintenance-only script, kept OUTSIDE the deployment repository: the repository
holds the lean files and nothing that refers to this step.

WHAT IS DEAD, AND HOW THAT WAS ESTABLISHED
------------------------------------------
`weights/checkpoints/*.tar` are training checkpoints, so they carry everything a
training run would need in order to continue.  Two groups in them cannot be read
by anything in the deployment:

* `extractor.desc_head.*` - ALIKE's SDDH descriptor head, 6 tensors / 316,480
  params.  Dead twice over: the configs this lineage descends from set
  `descriptor_source_policy: alike`, which makes the extractor resolve every view
  to ALIKE's own `convhead2` descriptors so `desc_head` is never called at all;
  and stage 1 is exported from `extractor.net.*` only.
* `optimizer` / `lr_scheduler` - resume state.  `gluefactory.train` reads
  `init_cp["optimizer"]` under `--restore` only, a flag no launch script in either
  repository passes - and a restore from these files could not work anyway, since
  the recovery runs start from a model whose parameter count differs.

Measured on the two shipped checkpoints: 90.7 MB and 70.6 MB of Adam moments plus
1.2 MB of SDDH each - 66% of the 9-layer file and 66% of the 7-layer one.

WHAT IS KEPT
------------
`model` minus SDDH, `conf` (re-checked against the architecture by
`make_pruned_ckpt.py` and `finetune.py`), and the `epoch` / `eval` metadata, which
costs a few hundred bytes and is genuinely read: `train.py` looks up
`best_cp["eval"][conf.train.best_key]` when a run resumes.

WHY IT REFUSES TO GUESS
-----------------------
`model` must consist of `extractor.net.*` and `matcher.*` and nothing else, so a
head added by a future training run cannot be dropped silently.  The stripped file
is written to a temporary path, re-loaded, compared tensor by tensor against the
original, and pushed through BOTH production loaders before it replaces anything.

Usage (run from this directory)
    python strip_checkpoint.py --checkpoint <repo>/weights/checkpoints/<f>.tar --in-place
    python strip_checkpoint.py --checkpoint <ckpt> --out <new>.tar
    ... --dry-run                 # report only
    ... --keep-training-state     # drop only the dead model heads
"""
import argparse
import os
import sys
from pathlib import Path

import torch

# Analysis-only script, kept OUTSIDE the deployment repository: nothing in
# alike_lightglue_ONNX&RKNN_deploy/ imports or reads any output from it.  It needs
# that repository on sys.path for the two checkpoint loaders.  Paths given on the
# command line may be relative to the repo root or to the current directory.
_REPO = Path(__file__).resolve().parents[1] / "alike_lightglue_ONNX&RKNN_deploy"
if not _REPO.is_dir():
    raise SystemExit(f"cannot find the deployment repo at {_REPO}; edit _REPO here")
sys.path.insert(0, str(_REPO))

# Read by a production loader: `extractor.net.*` by model/alike_stage.py,
# `matcher.*` by model/lightglue_stage.py.  Anything else in `model` is refused.
KEEP_MODEL_PREFIXES = ("extractor.net.", "matcher.")
DROP_MODEL_PREFIXES = ("extractor.desc_head.",)
DROP_TOP_LEVEL = ("optimizer", "lr_scheduler")


def resolve(path):
    """Accept a path relative to the CWD or to the deployment repo root."""
    p = Path(path)
    if p.exists():
        return p
    alt = _REPO / path
    if alt.exists():
        return alt
    raise SystemExit(f"no such file: {path} (also tried {alt})")


def _bytes(obj):
    """Serialised size of a nested checkpoint entry, in bytes."""
    if torch.is_tensor(obj):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_bytes(v) for v in obj)
    return 0


def _tensors(obj):
    if torch.is_tensor(obj):
        return 1
    if isinstance(obj, dict):
        return sum(_tensors(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_tensors(v) for v in obj)
    return 0


def _mb(n):
    return n / 2 ** 20


def split(ck, keep_training_state=False):
    """Return (stripped, dropped_model, dropped_top_level)."""
    state = ck["model"]
    known = KEEP_MODEL_PREFIXES + DROP_MODEL_PREFIXES
    unknown = sorted(k for k in state if not k.startswith(known))
    if unknown:
        raise SystemExit(
            "refusing to write: `model` holds tensors outside the prefixes the "
            "production loaders read, so dropping them would be a guess.  Teach "
            "this script about them first:\n  " + "\n  ".join(unknown[:12]))

    dropped_model = {k: v for k, v in state.items()
                     if k.startswith(DROP_MODEL_PREFIXES)}
    stripped = dict(ck)
    stripped["model"] = {k: v for k, v in state.items() if k not in dropped_model}

    dropped_top = {}
    if not keep_training_state:
        for key in DROP_TOP_LEVEL:
            if key in stripped:
                dropped_top[key] = stripped.pop(key)
    return stripped, dropped_model, dropped_top


def report(ck, keep_training_state):
    rows = []
    for prefix in KEEP_MODEL_PREFIXES + DROP_MODEL_PREFIXES:
        part = {k: v for k, v in ck["model"].items() if k.startswith(prefix)}
        action = "DROP" if prefix in DROP_MODEL_PREFIXES else "keep"
        rows.append((prefix, _tensors(part), sum(v.numel() for v in part.values()),
                     _bytes(part), action))
    for key in DROP_TOP_LEVEL:
        if key in ck:
            action = "keep" if keep_training_state else "DROP"
            rows.append((key, _tensors(ck[key]), 0, _bytes(ck[key]), action))
    named = [r[0] for r in rows]
    rows.append((", ".join(sorted(k for k in ck if k not in named)), 0, 0, 0,
                 "keep"))
    print(f"  {'group':22s} {'tensors':>8s} {'params':>13s} {'MB':>8s}   action")
    for name, n, params, nbytes, action in rows:
        print(f"  {name:22s} {n:>8d} {params:>13,} {_mb(nbytes):>8.2f}   {action}")


def verify(tmp, original, stripped):
    """Re-load the written file and re-check it three ways."""
    again = torch.load(tmp, map_location="cpu", weights_only=False)
    if sorted(again) != sorted(stripped):
        raise SystemExit("top-level keys changed across re-serialisation")
    differing = [k for k in again["model"]
                 if not torch.equal(again["model"][k], original["model"][k])]
    if differing:
        raise SystemExit(f"{len(differing)} kept tensors differ after reload: "
                         f"{differing[:5]}")
    print(f"  [verify] {len(again['model'])} kept tensors reload bit-identical")

    from model import load_alike_stage, load_lightglue_stage  # noqa: E402
    load_alike_stage(str(tmp), verbose=True)
    load_lightglue_stage(str(tmp), verbose=True)
    print("  [verify] both production loaders read the stripped file")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--in-place", action="store_true",
                    help="replace the checkpoint; the bytes dropped here are not "
                         "recoverable from inside the deployment repository")
    ap.add_argument("--out", default=None,
                    help="write the stripped checkpoint to this path instead")
    ap.add_argument("--keep-training-state", action="store_true",
                    help="keep optimizer / lr_scheduler; drop only dead heads")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.in_place == (args.out is not None):
        ap.error("pass exactly one of --in-place / --out")

    src = resolve(args.checkpoint)
    dst = src if args.in_place else Path(args.out)
    ck = torch.load(src, map_location="cpu", weights_only=False)
    print(f"{src}  ({os.path.getsize(src) / 2 ** 20:.1f} MB)")
    report(ck, args.keep_training_state)

    stripped, dropped_model, dropped_top = split(ck, args.keep_training_state)
    freed = _bytes(dropped_model) + sum(_bytes(v) for v in dropped_top.values())
    dropped_n = _tensors(dropped_model) + sum(_tensors(v)
                                              for v in dropped_top.values())
    print(f"  -> dropping {_mb(freed):.1f} MB ({dropped_n} tensors, "
          f"{sum(v.numel() for v in dropped_model.values()):,} params)")
    if args.dry_run:
        print("  --dry-run: nothing written")
        return

    tmp = dst.with_name(dst.name + ".tmp")
    torch.save(stripped, tmp)
    verify(tmp, ck, stripped)
    os.replace(tmp, dst)
    print(f"{dst}  ({os.path.getsize(dst) / 2 ** 20:.1f} MB)")


if __name__ == "__main__":
    main()
