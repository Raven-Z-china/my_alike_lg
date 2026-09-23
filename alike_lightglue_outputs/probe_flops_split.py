#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Does channel-pruning the backbone even matter?  Count FLOPs, not parameters.

WHY THIS IS ASKED BEFORE ANY PRUNING
------------------------------------
The backbone is 330,136 parameters out of roughly 11.95 M for the two stages -
**2.8 %**.  On a parameter basis, pruning it is pointless.  But parameters are not
work: the backbone is fully convolutional at 512x512, while the matcher is 512
keypoints' worth of attention, and the two scale in completely different ways.  A
33 % cut of a 2.8 % parameter share could still be a large fraction of the compute,
or a negligible one, and no parameter count can distinguish those.

This project has already been bitten by exactly that reasoning: the `heads 4->2`
ablation's parameter count looked like a real reduction (82.6 %) while the change
saved 32 parameters of 11.9 M and altered no tensor shape that mattered.  So the
question "is the backbone worth pruning" gets answered on FLOPs.

WHAT IS COUNTED
---------------
MACs (multiply-accumulate), per module, by instrumenting the module tree with
forward hooks.  MACs are exact for conv (k^2 * c_in * c_out * H_out * W_out) and
matmul (m * n * k) and are the right proxy for what an NPU spends its time on.
They are NOT a latency claim - see the README's note on the simulator - but a
ratio between two parts of the same graph is informative.

Usage
    python probe_flops_split.py --checkpoint <ckpt>
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

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

#: Modules that cost work, with the rule for counting them.
_CONV = (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)


class MacCounter:
    """Accumulate MACs per named module, by subtree prefix."""

    def __init__(self):
        self.per_module = {}
        self._handles = []

    def _count(self, module):
        if isinstance(module, nn.Conv2d):
            k = module.kernel_size[0] * module.kernel_size[1]
            return k * module.in_channels * module.out_channels \
                * module.output_hw[0] * module.output_hw[1] // module.groups
        if isinstance(module, nn.Linear):
            return module.in_features * module.out_features
        return 0

    def attach(self, model):
        for name, mod in model.named_modules():
            if isinstance(mod, _CONV):
                self._handles.append(
                    mod.register_forward_hook(self._make_hook(name)))
        return self

    def _make_hook(self, name):
        self.acc = getattr(self, "acc", {})

        def hook(mod, inp, out):
            if isinstance(mod, nn.Conv2d):
                # out is (N, C, H, W); cache the spatial size for _count
                mod.output_hw = (out.shape[-2], out.shape[-1])
            n_extra = 1
            if isinstance(mod, nn.Linear) and out.dim() > 2:
                # (B, K, C) -> B*K rows, not one
                for d in out.shape[:-1]:
                    n_extra *= int(d)
            self.acc[name] = self.acc.get(name, 0) + n_extra * self._count(mod)
        return hook

    def detach(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        return getattr(self, "acc", {})


def human(n):
    for unit in ("", "K", "M", "G"):
        if abs(n) < 1000:
            return f"{n:.1f}{unit}"
        n /= 1000.0
    return f"{n:.1f}T"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default=str(paths.TRAINED_CKPT))
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--keypoints", type=int, default=512)
    args = ap.parse_args()

    from model import load_alike_stage, load_lightglue_stage

    s1 = load_alike_stage(args.checkpoint, top_k=args.keypoints).eval()
    s2 = load_lightglue_stage(args.checkpoint,
                              image_size=(args.size, args.size)).eval()

    img = torch.rand(2, 3, args.size, args.size)

    with torch.no_grad():
        c1 = MacCounter().attach(s1)
        k, d, sc = s1(img)
        acc1 = c1.detach()

        c2 = MacCounter().attach(s2)
        s2(k[0:1], k[1:2], d[0:1], d[1:2])
        acc2 = c2.detach()

    tot1, tot2 = sum(acc1.values()), sum(acc2.values())
    print(f"=== MACs for one frame pair at {args.size}x{args.size}, "
          f"{args.keypoints} kpts ===")
    print(f"  stage 1 (backbone + DKD) : {human(tot1):>10s}   "
          f"{100*tot1/(tot1+tot2):5.1f} %")
    print(f"  stage 2 (matcher)        : {human(tot2):>10s}   "
          f"{100*tot2/(tot1+tot2):5.1f} %")
    print()
    print("=== stage 1, by top-level component ===")
    groups = {}
    for name, v in acc1.items():
        top = name.split(".")[0]
        groups[top] = groups.get(top, 0) + v
    for name, v in sorted(groups.items(), key=lambda kv: -kv[1]):
        print(f"  {name:22s} {human(v):>10s}   {100*v/tot1:5.1f} % of stage 1"
              f"   {100*v/(tot1+tot2):5.1f} % of total")
    print()
    print("=== stage 1 backbone, per conv (top 14) ===")
    convs = [(n, v) for n, v in acc1.items() if n.startswith("net.")]
    for name, v in sorted(convs, key=lambda kv: -kv[1])[:14]:
        print(f"  {name:38s} {human(v):>10s}  {100*v/tot1:5.1f} %")
    print()
    print("=== stage 2, by component ===")
    g2 = {}
    for name, v in acc2.items():
        parts = name.split(".")
        top = ".".join(parts[:2]) if len(parts) > 1 else parts[0]
        g2[top] = g2.get(top, 0) + v
    for name, v in sorted(g2.items(), key=lambda kv: -kv[1])[:8]:
        print(f"  {name:30s} {human(v):>10s}   {100*v/tot2:5.1f} % of stage 2")
    return 0


if __name__ == "__main__":
    sys.exit(main())
