# ALIKE + LightGlue — RKNN Deployment (RK3588)

Keypoint detection and matching on the Rockchip RK3588 NPU.
The full pipeline — ALIKE backbone, NMS, top-k, soft-argmax, descriptor
sampling, and the LightGlue matcher — runs **in-graph**, with **0 nodes
falling back to the CPU** in the shipped configuration.

This README covers the practical path: what ships, how to convert, the
measured accuracy on the open-source HPatches benchmark, ONNX inference
speed, and how to run each tool.

> **No latency claims.** All RKNN numbers come from the PC simulator
> (no board attached), which validates numerics only. On-device latency is
> still unmeasured.

## Repository layout

| path | contents |
|---|---|
| `weights/` | **All deployable artifacts** — checkpoints, ONNX, RKNN (see below) |
| `model/` | Torch modules used for ONNX export (the source of truth) |
| `model/matchers/` | LightGlue, vendored — see *Self-contained* below |
| `scripts/` | Conversion, auditing, benchmarking, ablation, and probe tools |
| `pruning/` | Checkpoint truncation + accuracy-recovery fine-tuning |
| `eval/` | End-to-end accuracy harness (HPatches) |
| `refs/` | Scratch (git-ignored): reference dumps + calibration lists |
| `paths.py` | every out-of-repo path, in one place |
| `cpp/` | On-board C++ inference sample (RKNN C API) |

Two things deliberately live **outside** this repository:

| location | contents |
|---|---|
| `../alike_lightglue_outputs/` | every tool's output, plus the analysis-only scripts |
| the paths in `HPATCHES_ROOT` / `GLUEFACTORY_ROOT` | the benchmark and the training checkout |

No code here reads anything from the outputs directory. Reports are written with
`--json` / `--report` / `--out` (defaulting to `paths.outputs()`), and the tools
never consume a previous run's report — so that directory can be moved, deleted
or archived without touching this code. `OUTPUTS_ROOT` relocates it.

### Self-contained

Everything on the deployable path — export, refine, convert, evaluate — runs from
this repository plus PyPI packages. The ALIKE backbone, the DKD detector and the
LightGlue matcher are all vendored under `model/`, so **no training checkout is
needed** to rebuild any shipped artifact, and a clone works from any directory:
checkpoint defaults resolve to `weights/checkpoints/`, not to one machine's
absolute path.

Two things genuinely live outside the repository, and `paths.py` resolves each
from an environment variable first, then a location relative to the checkout
(`data/<name>/` or a sibling directory), and otherwise refuses with a message
naming the variable — never by guessing an absolute path that happens to exist on
one machine:

| variable | needed for |
|---|---|
| `HPATCHES_ROOT` | the open-source HPatches benchmark; every documented number |
| `GLUEFACTORY_ROOT` | the two tools below, and nothing else |

No absolute path is stored anywhere in this repository. Reports name a file by
role rather than by location, so a checkpoint appears as
`<checkpoint:alike_native_gl_s1_d7.tar>` and the reader supplies
`GLUEFACTORY_ROOT` themselves. `scripts/portable_path.py` enforces that when a
report is written.

`pruning/finetune.py` is the one script that cannot be self-contained: it
*launches* `python -m gluefactory.train`, so it needs the training framework by
definition. You never invoke it to deploy — it reproduces the 9→7 recovery.
`scripts/verify_vendored_matcher.py` also wants the checkout, because certifying
the port means comparing against the original implementation; without it the
script skips with a clear message and the last recorded result stands (in the
outputs directory).

`weights/` is the one place to look for anything you can actually run:

```
weights/
├── checkpoints/
│   ├── alike_native_gl_s1_9L.tar    trained 9-layer baseline  -> weights/original/
│   └── alike_native_gl_s1_d7.tar    fine-tuned 7-layer matcher -> weights/optimized/
├── original/    un-optimised, deployable as-is
│   ├── alike_stage_bil.onnx          (+ .onnx.data)  stage 1
│   ├── alike_stage_bil_fp.rknn                       stage 1, fp16
│   ├── lightglue_stage_einsum.onnx   (+ .onnx.data)  stage 2
│   └── lightglue_stage_einsum_fp.rknn                stage 2, fp16
└── optimized/   the shipped pair
    ├── alike_stage_gath.onnx         (+ .onnx.data)  stage 1
    ├── alike_stage_gath_fp.rknn                      stage 1, fp16
    ├── lightglue_stage_d7.onnx       (+ .onnx.data)  stage 2
    └── lightglue_stage_d7_fp.rknn                    stage 2, fp16
```

Each `.onnx` has a sibling `.onnx.data` external-weight file — the two must
be moved and converted together. The `.rknn` files are the fp16 builds of
the same graphs. Nothing in `weights/` is disposable: every file is either a
shipped model or the exact checkpoint that produced one.

## Environments

Two conda environments. They never need to be active at the same time; which one
a step belongs to is stated at every step of the conversion workflow below.

| env | python | purpose | spec |
|---|---|---|---|
| `alike` | 3.10 | export ONNX, accuracy, pruning, ablation, fine-tuning | [`requirements-alike.txt`](requirements-alike.txt) |
| `rknn` | 3.8 | ONNX → RKNN conversion, simulator validation | [`requirements-rknn.txt`](requirements-rknn.txt) |

```bash
conda activate alike   # export + validate
conda activate rknn    # convert + simulator
```

### Setting them up

x86-64 Linux with a CUDA driver. Both torch builds cover Turing through Ampere
(`sm_75`…`sm_86` is inside their range), so either environment can use a
current-generation GPU; nothing here requires it to.

```bash
# --- alike: everything that builds or scores a graph ---
conda create -n alike python=3.10 -y && conda activate alike
pip install torch==2.11.0 torchvision==0.26.0 \
    --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-alike.txt

# --- rknn: conversion and the simulator, nothing else ---
conda create -n rknn python=3.8 -y && conda activate rknn
# torch FIRST, and from the CUDA index: rknn-toolkit2 pins `torch<=2.4.0`, and a
# bare `pip install rknn-toolkit2` satisfies that pin from the CPU-only PyPI wheel.
pip install torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu121
pip install rknn-toolkit2==2.3.2 omegaconf==2.3.1
```

Both spec files carry the reason for each pin, and which packages are optional.
`omegaconf` in the `rknn` environment is not a toolkit dependency: two
checkpoint-inspecting probes import it at module level.

Two constraints are hard rather than conventional. `rknn-toolkit2` 2.3.2 ships
only a **cp38** wheel, so the `rknn` environment's Python version is not a choice;
and the `torch<=2.4.0` pin is what forces that toolkit version's torch. The rest
of the pins are simply the combination this was tested on — drift of a minor
version is unlikely to matter.

Nothing else needs installing — no training checkout. Sanity check, which takes
no arguments and needs no data:

```bash
conda activate alike && python scripts/check_selfcontained.py   # expect 0 blocked
```

### Why two

`rknn-toolkit2` 2.3.2 declares `torch<=2.4.0, numpy<=1.26.4`. Meeting that
everywhere is not possible — three measured reasons:

* **Stage 2 cannot be exported by torch 2.4 at any opset.** At 13 it raises
  `UnsupportedOperatorError: aten::scaled_dot_product_attention` (the op is opset
  14+). At 14–18 it raises `IndexError` inside torch's own `aten::transpose`
  symbolic, which infers rank 4 for a rank-5 value and does not normalise negative
  dims. The same Python code exports correctly on 2.11, so this is a torch defect,
  not something this repository can rewrite around.
* **Stage 1 exports there, but ships 22 % larger.** At the same `--opset 18`:
  529 nodes / 12.32 MB from torch 2.4, against 254 nodes / 10.07 MB from 2.11.
  torch 2.4 inlines 172 constants that 2.11 hoists into initializers. Converting
  both against the same reference yields identical keypoints, descriptors and
  scores, so it is a packing difference, not an accuracy one — and both reach
  0 off-NPU nodes.
* **`numpy 2.x`, which `alike` uses, is excluded by the `numpy<=1.26.4` pin.**

So: `rknn` converts and simulates; `alike` builds and scores. The `rknn` arm of
the accuracy harness runs in `rknn` (with `--ref-from`) because it only loads the
simulator. Collapsing the split would mean making stage 2 exportable on torch 2.4
— removing the `F.scaled_dot_product_attention` call from `SelfBlock` (reached
unconditionally on CPU whenever `FLASH_AVAILABLE`, regardless of the `flash` flag)
*and* finding an opset-18-clean formulation of the rank-5 attention path. That has
no guarantee and would change the shipped graph. (Isolating the offending call
meant re-registering torch's own `aten::transpose` symbolic — decorator stack
reproduced exactly — to print the rank it had inferred: 4, for a rank-5 value.)

## What ships

Two stages per directory, run back to back. Static shapes everywhere;
512×512 input, 512 keypoints, both views in one call (batch 2).

| directory | stage | model | RKNN fp16 | off-NPU nodes |
|---|---|---|---|---|
| `weights/original/` | 1 | `alike_stage_bil.onnx` | 12.13 MB | **0** |
| | 2 | `lightglue_stage_einsum.onnx` | 42.93 MB | **65** |
| `weights/optimized/` | 1 | `alike_stage_gath.onnx` | 10.07 MB | **0** |
| | 2 | `lightglue_stage_d7.onnx` | 33.54 MB | **0** |

`original` is the un-optimised baseline: dense descriptor head, upstream
`Einsum` attention, 9 matcher layers. `optimized` is the shipped version:
keypoint-gathered descriptor head, `MatMul`/`Concat` attention, and the
9→7-layer matcher fine-tuned back above the 9-layer baseline.

The 65 off-NPU nodes in the baseline matcher are 27 `Einsum` (no RKNN
lowering), 2 `ScatterElements`, and 36 `Transpose` (they lower fine, then
get handed back: 512×512 exceeds the NPU transpose limit of 8192 by 32×).

## ONNX → RKNN conversion workflow

```bash
conda activate alike
# 1) Stage 1 (ALIKE). --batch 2 is required: the matcher consumes both
#    views in ONE call, so the reference dump must hold two feature sets.
python scripts/export_onnx.py --checkpoint weights/checkpoints/alike_native_gl_s1_9L.tar \
    --size 512 --keypoints 512 --batch 2 \
    --out weights/optimized/alike_stage_gath.onnx --ref refs/s1.npz
#    Baseline instead: add --no-gathered-head, output to weights/original/.

# 2) Stage 2 (matcher). --stage1-ref makes the export run on REAL stage-1
#    output rather than random tensors, so the reference dump is reachable.
#    Baseline instead: --attention einsum + the 9-layer checkpoint.
python scripts/export_onnx.py --stage lightglue --attention matmul \
    --checkpoint weights/checkpoints/alike_native_gl_s1_d7.tar --keypoints 512 \
    --stage1-ref refs/s1.npz --ref refs/s2.npz \
    --out weights/optimized/lightglue_stage_d7.onnx

conda activate rknn
# 3) Convert to RKNN fp16. --norm differs per stage: stage 1 normalises on
#    the NPU (C++ hands over raw uint8); the matcher must be left alone.
python scripts/convert_to_rknn.py weights/optimized/alike_stage_gath.onnx \
    --stage alike --norm 0,0,0:255,255,255 --ref refs/s1.npz \
    --report <reports>/rknn_alike_gath.json
python scripts/convert_to_rknn.py weights/optimized/lightglue_stage_d7.onnx \
    --stage lightglue --norm none --ref refs/s2.npz \
    --report <reports>/rknn_lg_d7.json

# 4) Off-NPU audit — the authoritative count: ONE build per process, both
#    log sinks captured. The audit inside convert_to_rknn.py is convenient,
#    not authoritative (it misses the C++ stream: it reports 36 where the
#    truth is 65 on the Einsum graph).
python scripts/build_log_probe.py --onnx weights/optimized/lightglue_stage_d7.onnx \
    --stage lightglue --dump /tmp/d7.log

# 5) End-to-end accuracy (the two arms run in separate envs).  The reference
#    checkpoint must be the one the matcher weights came from.
conda activate alike
python eval/run_accuracy.py --checkpoint weights/checkpoints/alike_native_gl_s1_d7.tar \
    --pipeline torch onnx --json <reports>/e2e.json
conda activate rknn
python eval/run_accuracy.py --pipeline rknn-fp16 \
    --ref-from <reports>/e2e.json --json <reports>/rknn.json
```

## Accuracy (HPatches)

Open-source HPatches viewpoint sequences, 512×512, 512 keypoints, seed 0.
Every number compares one variable at a time — `L2` = torch → ONNX (fp32),
`L3` = ONNX → RKNN fp16. The raw per-run reports are in the outputs directory.

Shipped model (`weights/optimized/`), 59 viewpoint pairs:

| metric | torch | ONNX (L2) | RKNN fp16 (L3) |
|---|---|---|---|
| `n_inl_gt` (deterministic inliers) | 225.5932 | **225.5932** | 225.2203 (−0.17 %) |
| `mAA` | 0.7476 | 0.7485 | 0.7343 |
| `@3px` | 0.8136 | 0.8136 | 0.7966 |
| `@5px` | 0.8136 | 0.8136 | 0.8136 |
| `mprec@3px` | 0.6535 | 0.6536 | **0.7335** |
| `n_match` | 329.54 | 329.51 | 295.24 (−10.4 %) |

Reading the table:

* **The export is exact.** `n_inl_gt` matches torch to four decimals on
  the deterministic metric; `mAA` differs by 0.0009.
* **The fp16 conversion keeps 99.83 % of the correct correspondences**
  while emitting ~10 % fewer matches — the matches it declines are the
  ones whose float32 confidence was already below 1e-4, which is why
  precision *rises* by 0.08 while the inlier count barely moves.
* Per-output alignment (8 real image pairs): ONNX keypoints are
  **bit-exact (0.000 px)**; RKNN sits at a median **0.043 px** (max 4.02 px,
  from 1–3 near-tie rank flips out of 512); descriptor cosine median
  **0.99971**. The matcher, fed identical inputs, agrees on 502/512 match
  indices.
* Compare arms on `n_inl_gt`, never on `n_inl`: the RANSAC inlier count is
  order-dependent (the same graph scored 258.46 alone and 245.68 after
  another pipeline ran), and `mprec@3px × n_match == n_inl_gt` holds by
  construction.

## ONNX inference speed

onnxruntime 1.23.2, `CPUExecutionProvider`, 512×512 / 512 keypoints, one
image pair, median of 20 runs. This is host-CPU ONNX time — **not** board
latency, and simulator wall-clock must not be used either (it rewards the
wrong direction: CPU-fallback `Transpose` runs faster on the host than the
simulated NPU `MatMul`).

| model | median ms | relative |
|---|---|---|
| stage 1, original (dense head) | 115 | 1.00× |
| stage 1, optimized (gathered head) | **46** | **2.5×** |
| stage 2, original (`Einsum`, 9 layers) | 210 | 1.00× |
| stage 2, `MatMul` 9 layers (graph rewrite only) | 95 | 2.2× |
| stage 2, optimized (`MatMul`, 7 layers) | **74** | **2.8×** |
| **full pipeline** | 324 → **120** | **2.7×** |

Of stage 2's 2.8×, 2.2× comes from the `Einsum`→`MatMul` graph rewrite and
the rest from the 9→7 layer cut. In MACs, the gathered head removes 23.5 %
of the whole pipeline (18.10 → 13.84 G).

## Tools and optimization scripts

| goal | command |
|---|---|
| graph-rewrite switches (defaults = optimized) | `export_onnx.py ... --attention matmul\|einsum`, `--gathered-head\|--no-gathered-head`, `--descriptor-interp bilinear\|floor` |
| prove the rewrite exact | `ablate_attention.py --checkpoint <ckpt> --verify` (asserts `max\|Δscore\| = 0`) |
| enumerate valid `MatMul` substitutions | `probe_einsum.py` (no args) |
| structural ablation (truncate weights, no retrain) | `ablate_matcher.py --checkpoint <9L ckpt> --only depth-8 depth-7 depth-6 agg-uniform --json <out>`; configs whose output does not change are rejected |
| truncation → loadable ckpt + config | `pruning/make_pruned_ckpt.py --checkpoint <9L ckpt> --depth 7 --out <exp-dir>`; `--heads` is refused (head surgery is impossible for this matcher) |
| accuracy-recovery fine-tune | `pruning/finetune.py --conf <pruned>/config.yaml --experiment <name> --resume-pruned <pruned> --extra train.max_steps=13611`; verifies conf matches the checkpoint first, exits on mismatch; `--dry-run` prints the command |
| backbone channel pruning (measured, rejected) | `probe_flops_split.py` (counts MACs), `scan_alike_channels.py`, `scan_channel_fractions.py`, all with `--checkpoint <ckpt>` |
| off-NPU node count (authoritative) | `build_log_probe.py --onnx <onnx> --stage <stage>` |
| model size + simulator ratio | `bench_stages.py --stage1 <onnx> --stage2 <onnx>...` |
| int8 (measured, not viable on RK3588) | `make_calib.py`, `sweep_int8.py` — stage 1 degrades to a 0.85 px median offset while 4 host nodes come back; stage 2 carries `-inf`, which min-max calibration cannot digest |
| certify the vendored matcher (wants the training repo) | `verify_vendored_matcher.py --checkpoint weights/checkpoints/alike_native_gl_s1_9L.tar` |
| re-assert self-containment | `check_selfcontained.py` |

Two rows name analysis-only tools that are **not** in `scripts/`: `probe_einsum.py`,
`probe_flops_split.py`, `scan_alike_channels.py` and `scan_channel_fractions.py`
live in the outputs directory alongside the reports. They produced an explanatory
or a rejected result and are not on the deploy path.

`pruning/structured.py` is a library, not a CLI: `channel_sensitivity` is
the one function the scan scripts call.

## C++ sample (on board)

```bash
cd cpp && mkdir build && cd build
cmake -DCMAKE_TOOLCHAIN_FILE=<aarch64-toolchain> \
      -DRKNN_API_ROOT=<path-to-librknn_api> \
      -DOPENCV_ROOT=<opencv-for-board> ..
make
./infer ../weights/optimized/alike_stage_gath_fp.rknn \
        ../weights/optimized/lightglue_stage_d7_fp.rknn \
        left.jpg right.jpg --out matches.txt
```

The sample reads uint8 colour images, lets the NPU normalise (mean 0 / std 255 — no
float preprocessing in C++), runs both stages, and writes one match per
line: `idx0 idx1 score`.
