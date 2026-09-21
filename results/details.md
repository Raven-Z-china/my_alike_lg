# Engineering details — ALIKE + LightGlue RKNN deployment

This is the long-form engineering log: every measurement, every failed
approach, and the reasoning behind each decision in the project. The
practical path — conversion commands, the headline numbers, and tool usage —
is in **[`README.md`](README.md)**; this file is what you read when you want
to know *why* those numbers are trustworthy (or why a plausible-looking
measurement was not).

Cross-modal feature extraction and matching on Rockchip NPUs (RK3588).
The keypoint detector (NMS, top-k, soft-argmax) is **in-graph**, and there is
**no `grid_sample` anywhere**: score and descriptor lookups use a
**round + clip + gather** pipeline that is NPU-safe.

**Status in one line: both stages are fully NPU-resident (0 off-NPU nodes, down
from 65 for the matcher), the converted pipeline keeps 99.83 % of the correct
correspondences, and the matcher has been pruned 9→7 layers and fine-tuned —
−22 % parameters with `mAA` +0.011 and no accuracy loss.** On-device latency is
still unmeasured; that needs the board.

Nothing above is a latency claim. What is measured is **NPU residency**: how many
nodes in each converted graph are executed by the host instead of the NPU, which is
a count and therefore exact. Getting that count right took four attempts — each one
reported a confident number, and the first three were wrong in the direction of
*looking clean*. See finding 3 under *What was measured* for the full sequence,
because the failure modes are more transferable than the number.

The deployed weights are the cross-modal model trained in
[glue-factory](https://github.com/cvg/glue-factory) at *512×512 input, 512
keypoints*: ALIKE backbone (fine-tuned) + LightGlue matcher (fine-tuned).
The SDDH descriptor head is **not** part of this deployment — the matcher reads
ALIKE's own `convhead2` descriptors.

### Self-contained

The backbone, the DKD detector and the LightGlue matcher are all vendored under
`model/`, so rebuilding every shipped artifact needs this repository and PyPI
packages and nothing else. The port was not taken on trust: the matcher used to
be imported from the training framework, and removing that import is exactly the
kind of change that can shift a number without raising anything, so
`scripts/verify_vendored_matcher.py` builds both implementations from the same
checkpoint and asserts the state dicts and all nine forward outputs are
**bit-identical** (recorded in `results/vendored_matcher.json`). It also proves
the one behaviour dropped in the port — construction-time loading of
`aliked_lightglue.pth`, a 47 MB file that every checkpoint's config points at —
was redundant, because the caller's checkpoint load overwrites every tensor it
touched.

Two things still want the training repo, and neither is on the deploy path:
`pruning/finetune.py`, which *launches* `python -m gluefactory.train` and cannot
be self-contained for that reason, and `verify_vendored_matcher.py`, which needs
both sides to compare. The latter skips with a clear message when the checkout is
absent.

Portability came with the same change. Paths used to be absolute literals in
fifteen files, so a clone anywhere but one machine resolved checkpoints and
datasets to nothing - and a wrong path does not fail where the mistake is made,
it fails inside a conversion, or quietly hands a directory listing to something
expecting tensors. `paths.py` now owns all of them: in-repo locations are derived
from the checkout's own position, and the three outside it (`HPATCHES_ROOT`,
`BIMODAL_ROOT`, `GLUEFACTORY_ROOT`) are resolved from the environment, then a
sibling directory, then the historical location, raising a message that names the
variable to set.

---

## Pipeline

Two RKNN models, run back to back. Static shapes everywhere.

```
        uint8 RGB  (2, 512, 512, 3)          both views, one call
                     │                        [NPU normalizes x/255]
        ┌────────────▼─────────────────┐
        │ STAGE 1 · alike (NPU)         │
        │   backbone → convhead2       │   129 ch = 128 desc + 1 score
        │   NMS × 7 (maxpool/mask)     │   static unrolled rounds
        │   TopK (K = 512)             │
        │   5×5 soft-argmax            │   sub-pixel refinement
        │   score  ← ROUND·CLIP·GATHER │   replaces grid_sample
        │   desc   ← ROUND·CLIP·GATHER │   replaces grid_sample
        └────────────┬─────────────────┘
        kpts (2, 512, 2)      desc (2, 512, 128)
                     │
        ┌────────────▼─────────────────┐
        │ STAGE 2 · lightglue (NPU)    │
        │   input_proj (128 → 256)     │
        │   9 × transformer, 4 heads   │
        │   filter_matches (static)    │   -1 padded, no dynamic shapes
        └────────────┬─────────────────┘
        matches0 (512,)   mscores0 (512,)
```

**Why two models, not one fused graph.** Every accuracy claim in this repo is
attributable: stage 1 and stage 2 are validated *independently* against the torch
reference, so a regression localises to a stage rather than a 300-op graph. A
fused single-graph export is also provided (`--fuse`); it shares the same weights
and the same validation protocol.

**Why round+clip+gather instead of `grid_sample`.** `grid_sample` is not
NPU-executable and falls back to CPU at runtime, stalling the whole pipeline. The
replacement changes exactly one thing numerically: the keypoint *score* lookup
moves from bilinear-at-subpixel to nearest-integer. Keypoint positions,
descriptors and matches are unaffected, and LightGlue does not consume scores at
all (`required_data_keys = keypoints, descriptors`). The change is measured, not
assumed — see the accuracy protocol.

---

## Repository layout

```
paths.py          every out-of-repo path in one place (datasets, training repo)
model/            torch modules used for export (source of truth)
  alike_stage.py          backbone + DKD, rewritten export-safe
  lightglue_stage.py      matcher wrapper, static filter_matches
  matchers/lightglue.py   LightGlue itself, VENDORED from the training framework
  gather_ops.py           round+clip+gather sampling (shared by both lookups)
  dkd.py                  NMS + top-k + soft-argmax, NPU-native
  gathered_head.py        descriptor computed only at the keypoints (exact, -23.5 % MACs)
scripts/
  export_onnx.py          torch → ONNX, static shapes + torch reference dump
  convert_to_rknn.py      ONNX → RKNN; fp16 primary, int8 optional; op audit
  make_calib.py           int8 calibration sets (images for stage 1, tensors for 2)
  sweep_int8.py           int8 dtype/algorithm sweep with accuracy + residency
  probe_flops_split.py    MACs per stage/conv - why the backbone is not 2.8 % of the work
  scan_alike_channels.py  whole-block sensitivity of the ALIKE backbone
  scan_channel_fractions.py  fractional channel cuts, measured on BOTH outputs
  probe_head_restructure.py  verify the gathered-descriptor head is exact
  verify_gathered_stage.py   wired-in stage vs dense stage, through the real forward
  diag_restructure_gap.py    locate a residual rather than guess at its cause
  probe_load_bisect.py    which rewrite breaks load_onnx (see finding 9)
  verify_vendored_matcher.py  certifies the vendored LightGlue against the
                          training framework's copy (the only script that wants
                          the training repo, and it skips without it)
  check_selfcontained.py  re-asserts that nothing on the deploy path imports the
                          training framework - runs every entry point with
                          `gluefactory` blocked at the interpreter level
  build_log_probe.py      ONE build per process; the trusted fallback count
  bench_stages.py         residency + size + simulator ratio per stage
  ablate_matcher.py       truncate layers / aggregation, score both judges
  ablate_attention.py     Einsum → MatMul rewrite, with --verify against upstream
  probe_einsum.py         enumerates MatMul substitutions; which ones are valid
  probe_depth.py          is a truncation actually changing the output?
  probe_log_routing.py    which sink the build log reaches (see finding 3)
pruning/
  make_pruned_ckpt.py     turn a truncation into a loadable checkpoint + config
                          (reconciles every list length and tensor shape against
                           the framework's own build - see the n-1 token_confidence
                           trap in the ablation section)
  structured.py           the channel-sensitivity gate the backbone scans call
                          (the layer/head surgery helpers that used to live here
                          were dead code and were removed - see the ablation
                          section for why head surgery is impossible anyway)
  finetune.py             accuracy recovery; verifies the conf matches the
                          checkpoint before launching (see below)
cpp/
  CMakeLists.txt          parameterised board paths
  infer.cpp               RKNN C API sample: image pair → matches
eval/
  run_accuracy.py         end-to-end: HPatches + registered RGB/TIR judge
weights/
  checkpoints/            the two trained checkpoints (9-layer, and 9->7 recovered)
  original/               the un-optimised baseline, both stages, ONNX + RKNN
  optimized/              the shipped pair: gathered head + MatMul/Concat + 9->7
refs/                     scratch (git-ignored): reference dumps, calib lists
results/                  accuracy, ablation and residency reports (committed)
```

`README.md` is the short one: the conversion commands, the accuracy and ONNX
speed tables, and how to invoke each optimisation script. Everything below is the
long-form reasoning behind those numbers.

The three `probe_*` scripts are not product code. They exist because each one
answers a question that a plausible-looking measurement got wrong, and keeping
them means the answer can be re-derived instead of re-remembered.

## Environments

Setup commands, version pins and the summary of why there are two are in
`README.md`. What follows is the measurement behind that summary, kept here
because it is diagnosis rather than setup.

`rknn-toolkit2` 2.3.2 declares `torch<=2.4.0, numpy<=1.26.4`. The tempting move
is to run torch 2.4 everywhere and delete one environment. It was tested, and it
does not work.

### torch 2.4 cannot export stage 2, at any opset

| opset | result |
|---|---|
| 13 | `UnsupportedOperatorError: aten::scaled_dot_product_attention` (opset 14+) |
| 14–18 | `IndexError` inside torch's own `aten::transpose` symbolic |

The first is a version fact. The second needed instrumenting, because the
traceback pointed into torch rather than into anything here:

```
*** CRASH  rank=4 (axes = [0, 1, 2, 3])  dim0=-2 dim1=4
```

torch 2.4's symbolic is

```python
axes = list(range(rank))
axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
```

with **no normalisation of negative dims**, and it is being called with
`dim1=4` on a value it believes is rank 4. `dim1=4` is only legal for rank ≥ 5,
and the graph does contain rank-5 tensors — the `unflatten` → `Reshape` path in
the attention blocks — so what is wrong is the **rank inference**, not the call.
The identical symbolic in torch 2.11 produces a correct graph from the same
Python, which is what makes this a torch defect rather than something to
work around here.

Two false leads on the way there, recorded because both looked convincing:

* The producer of the failing value is an already-translated `onnx::Mul`, i.e.
  the SDPA `query_scaled`. That does **not** mean SDPA's own symbolic built the
  bad transpose — that symbolic emits `Transpose` nodes directly, never through
  `aten::transpose`.
* Re-registering `aten::transpose` without its `parse_args("v", "i", "i")`
  decorator changes the call signature, so the first instrumentation attempt saw
  a raw `torch._C.Value` and raised a `TypeError` — a new bug wearing the same
  clothes. The decorator stack has to be reproduced exactly, which is why the
  crash above shows ints.

### torch 2.4 exports stage 1, identically but 22 % larger

Same `--opset 18`, `--gathered-head`:

| | nodes | initializers | `.rknn` |
|---|---|---|---|
| torch 2.4 | 529 | 30 | **12.32 MB** |
| torch 2.11 | 254 | 64 | **10.07 MB** |

torch 2.4 leaves 172 constants as `Constant` nodes (plus 8 `ConstantOfShape`,
5 `Shape`, 8 `Where`) where 2.11 hoists them into initializers. That is the whole
of the difference. Converting both against the same reference gives **identical**
keypoint distances, descriptor cosines and match scores, and both reach
0 off-NPU nodes — so it is a packing difference, not an accuracy one.

Worth recording because the first reading of it was wrong: the extra
`ConstantOfShape` nodes looked like the thing that had previously crashed
`load_onnx`, and the assumption was that this graph would fail to convert. It
converts fine. The earlier crash needed a shape driven by a tensor *input*; these
do not.

### What a merged environment would require

Making stage 2 exportable on torch 2.4 means removing the
`F.scaled_dot_product_attention` call from `SelfBlock` — reached unconditionally
on CPU whenever `FLASH_AVAILABLE`, regardless of the `flash` flag, which is
itself an upstream latent bug — *and* finding an opset-18-clean formulation of
the rank-5 attention path. No guarantee, and it would change the shipped graph.
Two environments are the cheaper answer.

An earlier note in this file claimed `rknn` "cannot use the GPU at all". That was
a property of one development machine (a GPU newer than the `sm_90` ceiling of
torch 2.4's build) and not of the environment. torch 2.4+cu121 carries
`sm_50`…`sm_90`, so it normally runs CUDA on anything up to Ampere, including
the RTX 3090 the two-environment setup is documented against.

---

## Quickstart

```bash
# 0. stage 1 (ALIKE backbone + DKD).  --batch 2 because the matcher consumes both
#    views in ONE call, so the reference dump must contain two sets of features.
#    `--out` names the shifted build; add --no-gathered-head / --attention einsum
#    (step 1) to reproduce the `original/` baseline instead.
python scripts/export_onnx.py \
    --checkpoint weights/checkpoints/alike_native_gl_s1_9L.tar \
    --size 512 --keypoints 512 --batch 2 \
    --out weights/optimized/alike_stage_gath.onnx --ref refs/s1.npz

# 1. stage 2 (matcher).  --attention matmul is the rewrite that takes it from 65
#    off-NPU nodes to 0; --stage1-ref makes the export run on REAL stage-1 output
#    rather than random tensors, so the reference dump is a reachable input.
python scripts/export_onnx.py --stage lightglue --attention matmul \
    --checkpoint weights/checkpoints/alike_native_gl_s1_d7.tar \
    --keypoints 512 --stage1-ref refs/s1.npz \
    --out weights/optimized/lightglue_stage_d7.onnx --ref refs/s2.npz

# 2. convert (fp16 primary path).  `--norm` differs per stage: stage 1 normalises
#    on the NPU so C++ can hand over uint8; the matcher must be left alone.
conda activate rknn
python scripts/convert_to_rknn.py weights/optimized/alike_stage_gath.onnx \
    --stage alike --norm 0,0,0:255,255,255 --ref refs/s1.npz \
    --report results/rknn_alike_gath.json
python scripts/convert_to_rknn.py weights/optimized/lightglue_stage_d7.onnx \
    --stage lightglue --norm none --ref refs/s2.npz \
    --report results/rknn_lg_d7.json

# 3. residency — ONE build per process, both log streams.  This is the number to
#    trust; the audit inside convert_to_rknn.py is convenient, not authoritative.
python scripts/build_log_probe.py --onnx weights/optimized/lightglue_stage_d7.onnx \
    --stage lightglue --dump refs/d7.log

# 4. end-to-end accuracy, both stages chained, against the torch reference.
#    torch and the simulator live in different envs, so the arms are separate runs.
conda activate alike
python eval/run_accuracy.py \
    --checkpoint weights/checkpoints/alike_native_gl_s1_d7.tar \
    --pipeline torch onnx --json results/e2e_d7_ta.json
conda activate rknn
python eval/run_accuracy.py --pipeline rknn-fp16 \
    --ref-from results/e2e_d7_ta.json --json results/e2e_d7_rknn.json
```

int8 is a separate, optional path and is reported on its own — and the report is
that it does not work here, for two independent reasons. See
`results/int8_report.md`: stage 1's accuracy collapses while 4 host nodes come
back, and stage 2's graph contains `-inf` which min-max calibration cannot digest.
The stage-2 calibration set must be **stage-1 outputs**, not images — the matcher's
input space is descriptors, and calibrating it on image statistics would quantise
the wrong distribution.

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

The sample reads uint8 RGB, lets the NPU normalise (mean 0 / std 255 — no float
preprocessing in C++), runs both stages, and writes one match per line:
`idx0 idx1 score`. A `--dump` flag writes raw stage outputs for cross-checking
against the Python parity harness.

---

## Accuracy protocol

Every claim is measured against the **torch reference** (the original
checkpoint, unmodified) and attributed to exactly one source of change:

| layer | what it isolates | comparison |
|---|---|---|
| **L0** | reference | torch checkpoint, as trained |
| **L1** | *algorithm* — round/clip/gather vs grid_sample | L0 vs export-module in fp32 torch |
| **L2** | *export* — torch → ONNX | L1 vs onnxruntime (fp32) |
| **L3** | *backend* — ONNX → RKNN fp16 | L2 vs RKNN simulator |
| **L4** | *quantisation* (optional) — fp16 → int8 | L3 vs RKNN int8 simulator |

Metrics and acceptance for the fp16 path ("lossless" in the measurable sense —
bitwise identity is impossible on an fp16 NPU and is not claimed):

| metric | acceptance |
|---|---|
| scores_map / descriptor_map max abs diff | < 0.005 |
| descriptor per-point cosine (512×128) | ≥ 0.9999 |
| keypoint integer positions | 100 % identical |
| keypoint sub-pixel offset | < 0.05 px |
| match index set (`-1` padding included) | identical |
| match score max abs diff | < 1e-3 |
| end-to-end HPatches mAA | within 0.001 of L0 |
| end-to-end RGB/TIR judge, inliers@3px | identical (± 1, seed noise) |

Evaluation sets (fixed seeds, committed lists):
- dense / keypoint / descriptor parity — 32 images, mixed modality
- matcher parity — 64 constructed pairs (same-image warped) + 64 registered pairs
- end-to-end — HPatches (540 pairs) and the registered RGB/TIR judge (600 pairs)

The int8 path reports the same table with no pass/fail line — it characterises
the trade-off, and per-layer quantisation sensitivity plus a skip-quant list are
generated alongside it.

## Tuning & recovery

If a layer fails, the fix is determined by *which* layer failed:

- **L1** — the only possible L1 regression is the keypoint-score lookup. It is
  bounded and LightGlue never reads scores; if the end-to-end numbers hold, the
  delta is documented and accepted.
- **L2** — op-level export bug; fix the export (opset, op reordering), never the
  weights.
- **L3** — per-stage localisation first, then mixed-precision config or an
  op rewrite.
- **L4** — re-calibrate with a larger mixed-modality set, extend the skip-quant
  list, or fine-tune with quantisation awareness.
- **pruning** — recovery fine-tune uses the validated glue-factory recipe:
  matcher-only, frozen extractor, lr 1e-4, 1 epoch (13,612 steps). Acceptance
  after recovery: the full L0→L3 protocol re-run, no metric below its line.

## Pruning

Structured only (whole channels / whole layers — no sparse masks, which NPUs
cannot exploit):

- **ALIKE backbone** — channel pruning by L1 ranking, gated by a per-block
  sensitivity scan (prune one block at a time, measure end-to-end before
  committing).
- **LightGlue** — layer pruning 9→{6,7,8} combined with `aggregation: uniform`
  (layer-count changes require score averaging, not last-layer-only), and head
  pruning 4→2. Internal width is **not** reduced: at 512 keypoints the matcher
  is launch-overhead-bound, and halving its width saves only ~3.5 % latency
  while costing 75 % of its parameters' capacity.

## Results

Measured on 2026-09-20, 512×512 / 512 keypoints, RK3588 toolchain 2.3.2,
`alike_native_gl_s1`. Artifacts under `results/`; the shipped pair under
`weights/optimized/`, the baseline it is measured against under
`weights/original/`.

### Conversion

| stage | RKNN fp16 | off-NPU nodes | breakdown |
|---|---|---|---|
| alike (backbone + DKD) | 12.13 MB | **0** | — |
| lightglue, upstream | 42.93 MB | **65** | 27 `Einsum`, 2 `ScatterElements`, 36 `Transpose` |
| lightglue, `--attention matmul` (as it was) | 44.37 MB | **2** | 2 `ScatterElements` |
| **lightglue, `--attention matmul` (current)** | **42.78 MB** | **0** | — |

**Both stages are now fully NPU-resident.** Reproduce any row with
`scripts/build_log_probe.py`, which builds one graph in a fresh process and counts
both warning streams (`results/probe_*.json`); the current matcher reports
`off-NPU nodes: 0` with `parser_consistent: true`, and a plain `grep` of the dumped
log agrees on all four patterns.

The whole point of replacing `grid_sample` and `Unfold` was a graph with nothing
falling back to the CPU. Stage 1 achieves it outright. Stage 2 did not, and both
earlier versions of this table were wrong about it in different ways — see
findings 3 and 5, which are about the *audit*. The rewrite that fixes it is
`--attention matmul`: `Einsum` has no RKNN lowering, and expressing the same
contraction as a batched `MatMul` is verified exact torch-to-torch at
`max |Δscore| = 0.000e+00` (`scripts/ablate_attention.py --verify`). The *exported*
graph is a fraction of a percent away from bit-identical rather than exactly
identical — see the end-to-end table below, which is the number that matters.

The 36 `Transpose` nodes are the surprise, and they are why the rewrite pays even
though `Einsum` is also on the list. They do not fail to lower — they lower, then
get handed back:

```
Transpose will fallback to CPU, because input shape has exceeded the max limit,
height(512) * width(512) = 262144, required product no larger than 8192!
```

The matcher moves `(B, H, N, N)` tensors around and 512×512 is **32× over** the
NPU's transpose limit, so every one of those moves to the host. They disappear
entirely under the `MatMul` form, because the head/sequence permutation the
upstream code needs is absorbed into the contraction's operand layout.

The last 2 were the corner assignments in `sigmoid_log_double_softmax`, and they
are now gone too — see finding 9 for the toolkit bug that had to be worked around.

#### What the rewrites cost, measured end to end

Both rewrites are exact **torch-to-torch** (`max |Δscore| = 0.000e+00` for the
attention and the assignment together). Same 59 HPatches pairs, same 200 RGB/TIR
pairs, same stage 1 (`results/e2e_final.json`):

| metric | torch | ONNX | Δ |
|---|---|---|---|
| HPatches `n_inl_gt` (deterministic) | 225.5593 | **225.5593** | **0.0000** |
| HPatches `mAA` | 0.7365 | 0.7355 | −0.0010 |
| HPatches `n_match` | 326.017 | 326.051 | +0.034 |
| RGB/TIR `rep@3` | 0.6218 | 0.6218 | 0.0000 |
| RGB/TIR `n_inl@3` | 313.995 | 313.995 | 0.0000 |
| RGB/TIR `matches_mean` | 390.520 | 390.560 | +0.040 |

On the deterministic inlier count the export is **identical to torch to four
decimals**. The `mAA` residual of −0.001 is not a model difference — it comes from
the RANSAC estimator, and the section below shows how large that artefact actually
is.

An earlier revision of this file reported **−0.4 % inliers** for the `MatMul`
rewrite and called it "a trade, not a free lunch". That conclusion was wrong, and
the reason is a measurement defect, not a model one: the inlier count it used came
from RANSAC, which is order-dependent. See *the RANSAC artefact* below.

#### The RANSAC artefact, which produced a wrong conclusion

`cv2.findHomography(..., cv2.RANSAC)` draws from OpenCV's **global** RNG, so its
inlier count for a given pair depends on how many such calls happened before it.
Measured on this same 59-pair set with the **same ONNX graph**:

| invocation | `n_inl` |
|---|---|
| `--pipeline onnx` alone | **258.4576** |
| `--pipeline torch onnx`, onnx arm | **245.6780** |

A 13-inlier swing — 5 % of the total — from call order alone, and
`cv2.setRNGSeed` does **not** remove it. Every two-arm comparison in this project
was therefore measuring its arms on different footings, and the first ONNX arm
that ran was always the favoured one.

The fix is to take the estimator out of the measurement: **`n_inl_gt`** counts the
matches consistent with the *known* homography, which is a pure function of the
matches and the ground truth. It is reproducible across processes and orders, and
it is the better question anyway — "how many correspondences are correct" rather
than "how many a particular RANSAC run agreed with". `n_inl` and the `H_error`/
`mAA` derived from the estimate are still reported, because they are the protocol
the training-side tables use, but they are read with this caveat attached.

Consequences, both applied: the end-to-end comparison above now uses `n_inl_gt`,
and the ablation sweep was re-run on it (the previous numbers were inside the
noise — see the ablation section).

**On-device latency has not been measured** — that needs the board. The PC
simulator validates numerics only, and its wall-clock is host CPU emulation. No
number in this README is a latency claim.

### End-to-end, both stages chained

Independently reproducible with:

```bash
python eval/run_accuracy.py \
    --checkpoint weights/checkpoints/alike_native_gl_s1_9L.tar \
    --pipeline torch onnx rknn-fp16 --hpatches 540 --crossmodal 600
```

The ONNX arm is **bit-identical to torch on both judges** - `mAA` 0.6228 vs
0.6228, `n_inl` 279.83 vs 279.83, `rep@3` 0.6355 vs 0.6355. The only non-zero
cell anywhere is `prec@3` at +0.0003 on 8 pairs, which is one match out of four
hundred and is well inside the seed-to-seed spread of that judge. That is what
"the export is exact" means end to end, not just per tensor.

The RKNN arm runs the **PC simulator**, so the numbers below validate numerics
and say nothing about latency.

### Accuracy protocol, on real images

8 real HPatches viewpoint pairs at 512×512, 512 keypoints, RKNN fp16 simulator
vs the torch reference (`results/accuracy.json`):

| metric | ONNX | RKNN fp16 |
|---|---|---|
| keypoints, median offset | **0.000000 px** (bit-exact) | **0.043 px** |
| keypoints, max offset | 0.000 px | 4.02 px (a few near-tie flips) |
| descriptors, sampling operator alone | — | **median cos 0.99998**, p1 0.9918 |
| descriptors, incl. keypoint movement | — | **median cos 0.99971**, p1 0.9908 |

The matcher stage, fed identical inputs:

| metric | result |
|---|---|
| matches | 253 valid in both, **502 / 512 indices identical** |
| match scores | max abs diff 4.2e-02 |

**Both stages meet the bar in substance.** ONNX is bit-exact; RKNN's residual is
0.043 px of keypoint jitter from fp16 NPU arithmetic, which costs 0.0003 of
descriptor cosine. The matcher is essentially exact.

### The converted pipeline, end to end, against torch

Both stages chained on the RKNN simulator, 59 HPatches pairs + 200 RGB/TIR pairs
(`results/e2e_rknn.json`, `results/e2e_final.json`). **This table is the 9-layer
`MatMul` graph** — the depth cut has since been re-measured on the shipped 7-layer
graph and it changes nothing structurally: `n_inl_gt` 225.5932 → 225.2203 and
`n_inl@3` 313.95 → 313.55, i.e. **99.83 % / 99.87 % of the correct correspondences
retained**, with 10–14 % fewer matches instead of 8–11 % (`results/e2e_d7_ta.json`,
`results/e2e_d7_rknn.json`; see the *Accuracy* section of `README.md`):

| metric | torch | RKNN fp16 | Δ |
|---|---|---|---|
| HPatches `rep@3` (detector) | 0.6218 | 0.6221 | +0.0003 |
| HPatches `n_inl_gt` (deterministic) | 225.5593 | 225.2203 | **−0.339 (−0.15 %)** |
| HPatches `n_match` | 326.02 | 300.10 | **−25.9 (−8 %)** |
| HPatches `mprec@3px` | 0.6594 | 0.7175 | **+0.058** |
| RGB/TIR `rec@3` | 0.9862 | 0.9845 | −0.0017 |
| RGB/TIR `n_inl@3` | 313.99 | 313.59 | **−0.41 (−0.13 %)** |
| RGB/TIR `matches_mean` | 390.52 | 346.40 | **−44.1 (−11 %)** |
| RGB/TIR `prec@3` | 0.7998 | 0.9013 | **+0.102** |

**The converted model keeps 99.85 % of the correct correspondences** (`n_inl_gt`)
and 99.87 % on the cross-modal judge, while emitting 8–11 % fewer matches. The
matches it declines to emit are the ones it is least sure about, so precision rises
by 0.06–0.10. For a downstream RANSAC or pose solver this is close to the ideal
shape of a quantisation effect: same geometry, fewer outliers.

The deployment emits ~11 % fewer matches and keeps essentially all the inliers, so
precision rises sharply. That is a good outcome — the dropped matches were junk —
but "the matcher is fine" does not follow from it, so it was localised with a
stage-crossing probe (`scripts/probe_match_drop.py`, 24 pairs):

| crossed pipeline | matches/pair | contribution |
|---|---|---|
| A — ONNX stage 1 → ONNX stage 2 (reference) | 325.67 | — |
| B — **RKNN stage 1** → ONNX stage 2 | 324.92 | **−0.75** (detector) |
| D — ONNX stage 1 → **RKNN stage 2** | 302.04 | **−23.62** (matcher) |
| C — RKNN stage 1 → RKNN stage 2 (shipped) | 302.21 | −23.46 |

The detector is exonerated: order-matched, its keypoints sit **0.045 px** from the
float32 ones and its descriptors at cosine **0.999689**. The whole difference is in
the matcher, and the mechanism is the `filter_matches` threshold:

* the matcher drops **24.4** matches/pair and adds **0.8**;
* of the dropped, **23.9 had a float32 score below 1e-4** — and fp16's smallest
  normal is 6.1e-5.

`filter_matches` keeps a match when `mscores0 > 0`, with `mscores0 = exp(max0)`.
A log-assignment maximum near −11 gives `exp(−11) ≈ 1.7e-5`, which float16 rounds
to the edge of its range and across that `> 0` test. So fp16 discards matches whose
confidence is already ~1e-5 — which is why the inlier count does not move. The
build log says the same thing in its own words:
`range [-inf, -2.38e-07] ... out of the float16`.

An earlier, much smaller probe (4 dropped, 3 added of 512) suggested this was
negligible. It used a **random-noise image** for the reference dump, whose matching
problem is degenerate — the same trap as the ablation probe below, and the reason
both probes now run on real image pairs.

### Structural ablations: what depth, heads and aggregation actually cost

The matcher is 95 % of the parameters (11.36 M of 11.95 M), so every size lever is
in the matcher. Measured by **truncating the trained weights** rather than
retraining — `scripts/ablate_matcher.py`, 59 HPatches pairs + 150 registered
RGB/TIR pairs, `results/ablation_matcher.json`.

Deltas against the untruncated baseline, on the **deterministic** inlier count.
`n_inl_gt` and `n_inl@3` are counts, so read them before the ratios:

| config | params | HPatches `mAA` | HPatches `n_inl_gt` | RGB/TIR `rec@3` | RGB/TIR `n_inl@3` |
|---|---|---|---|---|---|
| baseline (9 layers, 4 heads) | 100 % | — | (225.56) | — | (314.0) |
| layers 9→8 | 88.9 % | +0.011 | **+0.37** | −0.0001 | −0.03 |
| **layers 9→7** | **77.8 %** | −0.010 | **+0.10** | −0.0002 | **−0.05** |
| layers 9→6 | 66.8 % | +0.012 | −3.10 | −0.0008 | −0.23 |
| aggregation → uniform | 100 % | +0.007 | +0.41 | −0.0008 | −0.21 |
| layers 9→7 + aggregation uniform | 77.8 % | +0.011 | −0.29 | −0.0023 | −0.64 |
| ~~heads 4→2~~ | ~~82.6 %~~ | *unreproducible* | ~~−36.42~~ | *unreproducible* | ~~−47.4~~ |
| ~~layers 9→8 + heads 4→2~~ | ~~73.4 %~~ | *unreproducible* | ~~−37.54~~ | *unreproducible* | ~~−48.5~~ |

`heads 4→2` was measured with a weight truncation that turned out **not to be
valid**, so those rows cannot be reproduced and are struck through rather than
deleted — the numbers were real outputs of real code, and hiding that would be
worse than labelling it. See *why the head rows are invalid* after the list.

**`mAA` cannot tell these levers apart; the inlier counts can.**

* **Removing layers is nearly free.** 9→7 *adds* 0.1 inliers to a 225.6 baseline
  (+0.05 %) while cutting 22 % of the parameters, and costs 0.0002 of `rec@3` on
  the cross-modal judge. The 9→8 row is likewise positive. Both are within a
  fraction of an inlier of the baseline, and that is now a statement the metric can
  actually support.
* **The head rows suggested removing heads is expensive** — 4→2 costing 36.4
  HPatches inliers (−16.1 %) and **47 of 314** cross-modal inliers — 15 % of all
  usable correspondences — for a 17 % cut. The ratio columns mostly *improve*
  (`prec@3` +0.062), because dropping heads drops weak matches and precision is a
  mean of ratios. **That conclusion is not trustworthy** — see below — but the
  *shape* of the warning stands on its own: an ablation can improve every
  precision-like column while destroying the match count, and precision is a mean of
  ratios so it goes up when weak matches are removed.
* The gap between the layer and head effects is ~360× on the deterministic metric
  and was ~30× on the RANSAC one, where it sat inside the noise band. That is why
  the sweep was re-run: the earlier table reported the 9→7 cost as −0.02 inliers, a
  number that metric could not resolve.

#### Why the head rows are invalid

Truncating the trained model to 2 heads is **not possible by slicing weights**, and
the reason is that the two attention blocks have incompatible layouts:

| block | projection | shape at 4 heads | shape at 2 heads | sliceable? |
|---|---|---|---|---|
| `SelfBlock` | `Wqkv` | `(768, 256)` | `(768, 256)` | **no — identical** |
| `CrossBlock` | `to_qk`, `to_v`, `to_out` | `(256, 256)` | `(256, 256)` | **no — identical** |
| `posenc` | `Wr` | `(32, 2)` | `(64, 2)` | **yes** |

`SelfBlock.Wqkv` is `nn.Linear(embed_dim, 3 * embed_dim)` — its width does **not
depend on `num_heads`** at all. The heads are recovered in `forward` by
`qkv.unflatten(-1, (num_heads, -1, 3))`, so each head's q, k and v are the three
*interleaved* values of a contiguous `head_dim × 3` run. Keeping a prefix of
`3·heads·head_dim` rows does not keep a set of whole heads, and no reordering of
that buffer makes it one. `CrossBlock.to_qk` is
`nn.Linear(embed_dim, dim_head * num_heads)` where `dim_head * num_heads ==
embed_dim` for **every** valid head count — so its shape is unchanged and slicing it
is what *creates* a mismatch rather than removing one.

**The parameter count could not reveal this, because the two errors cancel.** The
slicer reported "45 projections across 18 modules, ~2.07 M removed, 82.6 %" — a
number consistent with a real ablation, produced by cutting `SelfBlock` the wrong
way and `CrossBlock` when it should not have been cut at all.

It surfaced at fine-tune launch, and only there:

```
size mismatch for matcher.posenc.Wr.weight: copying a param with shape [32, 2]
  from checkpoint, the shape in current model is [64, 2]
size mismatch for matcher.transformers.0.self_attn.Wqkv.weight:
  ... [384, 256] ... [768, 256]
size mismatch for matcher.transformers.0.cross_attn.to_qk.weight:
  ... [128, 256] ... [256, 256]
```

The training-time `LightGlue.__init__` reads `conf.num_heads` and builds
`head_dim = descriptor_dim // num_heads`, so it constructs the correct 2-head model
on its own. **Changing the head count therefore needs no weight surgery at all** —
it is a retrain with `matcher.num_heads: 2`, and the config produces the right
shapes. Both `slice_attention_heads` and `pruning/make_pruned_ckpt.py` now raise
instead of acting, so the invalid path cannot be taken again.

What survives from those rows: **heads were never a viable size lever anyway.**
`num_heads` changes `head_dim`, not any parameter shape except `posenc.Wr`, so a
4→2 head change saves **32 parameters of 11.9 M** — nothing. It is an
architecture choice to retrain for accuracy, not a compression technique. That is
a stronger and more useful statement than the −36-inlier figure, and it is the one
the shape table proves.

The two *layer* levers compose additively in cost, so there is no cheap
combination there: the best size/accuracy point is layers 9→7 alone, then 9→6.

**These are lower bounds, not the retrained numbers.** The surviving layers were
trained with the removed ones present, so a config that scores well truncated is a
safe choice and a config that scores badly may recover after fine-tuning. The
reverse reading does not hold, which is why the ablations only promote candidates
and never reject them.

#### `layers 9→7`, fine-tuned and adopted

The truncation said 9→7 was nearly free, so it was trained
(`results/ablation_matcher_d7_ft.json`). 13,611 steps, lr 1e-4, 1 epoch, seed 0,
extractor frozen — the recipe in `pruning/finetune.py`, launched from the pruned
checkpoint:

| metric | 9 layers (baseline) | **9→7 fine-tuned** | Δ |
|---|---|---|---|
| params | 11.885 M | **9.251 M** | **−22.2 %** |
| HPatches `n_inl_gt` | 225.5593 | **225.5932** | **+0.034** |
| HPatches `mAA` | 0.7365 | **0.7476** | **+0.011** |
| HPatches `n_match` | 326.02 | 329.54 | +3.5 |
| HPatches `mprec@3px` | 0.6594 | 0.6535 | −0.006 |
| RGB/TIR `n_inl@3` | 313.995 | **315.833** | **+1.84** |
| RGB/TIR `rec@3` | 0.9862 | **0.9866** | +0.0004 |
| RGB/TIR `prec@3` | 0.7998 | 0.7920 | −0.008 |
| RGB/TIR `matches_mean` | 390.52 | 397.05 | +6.5 |

**The 7-layer model beats the 9-layer one on both inlier counts and on `mAA`,
while carrying 22 % fewer parameters.** That is not a win for pruning as such — it
is one epoch of fine-tuning at lr 1e-4 applied to a model that had been trained on
a different schedule, and the extra optimisation is worth more than the two layers
cost. The honest reading is that **layers 9→7 is free after a short recovery, so
the smaller model should be adopted**; it is not that depth is harmful.

The exported 7-layer matcher is also **0 off-NPU nodes**, same as the 9-layer
`MatMul` graph — the rewrite composes with the pruning.

#### `layers 9→6`: where the curve turns

The 9→7 recovery **beat** the 9-layer baseline, so 9→6 was run the same way
(`results/ablation_matcher_d6_ft.json`). It is the first point of the three where
accuracy actually gives way:

| model | params | HPatches `n_inl_gt` | HPatches `mAA` | RGB/TIR `n_inl@3` |
|---|---|---|---|---|
| 9 layers (baseline) | 11.885 M | 225.5593 | 0.7365 | 313.9950 |
| **9→7 fine-tuned** | **9.251 M** | **225.5932** | **0.7476** | **315.8330** |
| 9→6 fine-tuned | 7.934 M | 223.6271 | 0.7269 | 315.7133 |
| 9→6 vs baseline | −33.2 % | **−1.93** | **−0.0096** | +1.72 |

**9→7 is the optimum, and it is a genuine interior point rather than the end of a
trend.** 9→6 removes another 1.32 M parameters — 14 % of what remains — and pays
`mAA` −0.021 relative to 9→7 with no cross-modal gain (+1.72 against +1.84). So the
two judges stop agreeing past depth 7: the cross-modal number still likes the
smaller model while HPatches has turned down, which is the signal that the shallow
end is now losing capacity rather than shedding redundancy.

The 6-layer graph is still **0 off-NPU nodes** and 30.41 MB against 42.78 MB for the
9-layer `MatMul` graph.

**Why the two judges disagree is worth noting.** HPatches is visible-only viewpoint
pairs and stresses geometry; the RGB/TIR judge is cross-modal and stresses the
descriptor space the model was trained for. A depth cut that helps one and hurts the
other is consistent with the shallower matcher becoming more reliant on descriptor
similarity and less on the iterative geometric refinement the later layers provide.
That reading is a hypothesis, not a measurement — but it is the reason a single
number should not decide this, and it is why both judges are reported.

#### `finetune.py` is verified, and the check it was missing

`pruning/finetune.py` had never been executed — only `--dry-run`ed. Running it
found two defects, and both are fixed:

**It did not `chdir`.** The subprocess is `python -m gluefactory.train`, which only
resolves from the framework's root, so the documented usage (`python
pruning/finetune.py ...` from this project) died with a bare
`No module named 'gluefactory'`. It now resolves the checkout the same way the
exporter does (`$GLUEFACTORY_ROOT`, then siblings) and runs with `cwd=`.

**It did not check that `--conf` matches the checkpoint**, which is the failure that
cost a whole training run earlier. `train.load_experiment` merges
`conf.model = merge(ckpt_conf.model, cli_conf)` — **the CLI config wins** — so
passing the original config next to a pruned checkpoint builds the *original*
architecture and then loads the pruned weights into it with `strict=False`. PyTorch
tolerates MISSING keys under `strict=False`, so the extra layers stay at their
random initialisation and the run trains a model with random layers while the loss
curve looks healthy.

It now refuses, before touching anything:

```
architecture mismatch: the checkpoint at ... is n_layers=7 but --conf says
n_layers=9.  `--conf` WINS over the checkpoint's stored config, so this run would
build the 9-layer model and load the 7-layer weights into it with `strict=False`,
silently leaving the extra layers at their random initialisation.
```

**Why the guard is not redundant, given that the crash exists.** The silent version
was caught at all only because `matcher.confidence_thresholds` is a length-`n_layers`
buffer and a *shape* mismatch does raise even under `strict=False`. That is a lucky
tripwire — it works because a parameter happens to be shaped by the layer count —
not a check. Before that buffer was being truncated (see the `token_confidence` trap
above) the checkpoint's copy was still `[9]`, it matched a 9-layer model, and the run
went through. `check_architecture` is the actual check.

Verified by three runs (`--extra train.max_steps=20`):

| test | result |
|---|---|
| correct conf, from this project's root | completes; `Finished training on process 0.` |
| checkpoint and conf both 7 layers | `architecture check OK: checkpoint has 7 transformer layer(s) (indices 0..6)` |
| **original conf against the 7-layer checkpoint** | **refused, exit 1** |
| output checkpoint | `n_layers=7`, lists **7/7/6**, `confidence_thresholds (7,)`, matcher **9.250900 M** — identical to the adopted 9→7 model |

#### A latent bug that only bites at depth 6

`LightGlue.__init__` builds the three per-layer `ModuleList`s with **different**
lengths:

```python
self.transformers     = [TransformerLayer(d, h) for _ in range(n)]      # n
self.log_assignment   = [MatchAssignment(d)   for _ in range(n)]        # n
self.token_confidence = [TokenConfidence(d)   for _ in range(n - 1)]    # n - 1
```

Truncating all three with a single `del lst[depth:]` leaves `token_confidence` one
layer too long. The failure is silent: the fine-tune loads with `strict=False`, so
the surplus tensor is reported as `unexpected` and the run proceeds — with a
checkpoint whose parameter count is not the model's.

**It was latent at depth 7 and only became a real defect at depth 6.** The framework
itself builds `n - 1` token-confidence heads, so at `n=7` my slice kept 6 and the
framework built 6 — they coincided by accident. At `n=6` my slice kept 6 and the
framework built 5. Both the published 9→7 run and the 9→6 run were re-checked against
this: 9→7 is clean (framework 7/7/6, checkpoint 7/7/6, 9.897516 M both, zero
`unexpected`), and after the fix 9→6 matches exactly (8.580649 M both, lists 6/6/5,
zero `unexpected`).

The check that catches this in one step is to build the model from the config and
compare its `state_dict()` against the checkpoint's — **parameter totals alone are
not enough, which is how the same class of error survived an earlier round.**

Two things the recovery run exposed, both checkpoint-shape problems rather than
model ones:

* `train.load_experiment` reads `checkpoint["conf"]`, so a checkpoint packed with
  weights alone dies at startup with a bare `KeyError: 'conf'`.
* `matcher.confidence_thresholds` is a length-`n_layers` parameter that carries no
  layer index in its name, so cutting the three `ModuleList`s left it at `[9]`
  against a 7-layer model and the framework refused the load with
  `size mismatch ... shape [9] ... shape in model is [7]`. `pruning/make_pruned_ckpt.py`
  now reconciles every packed tensor against the truncated model's expected shapes
  rather than patching this one name, and reports anything it truncates.

Every row is verified to have actually taken effect before it is scored: a
truncation that leaves the output bit-identical is rejected rather than reported
as a 0.0000 delta. That check exists because a first attempt at this sweep keyed
off attribute names that do not exist on this matcher, sliced nothing, and
reported a head count that had never been applied.

### A measurement bug worth recording, because it produced the opposite conclusion

An earlier revision of this file reported the stage-1 descriptors as **FAILING at
median cosine 0.88**. That number was an artefact. The comparison matched tensors
by POSITION, and `topk` returns its selection in score order, which is
backend-specific — so `d_sim[i]` and `d_ref[i]` were descriptors of two different
points. Matching the correspondence first (nearest neighbour, then compare) gives
median **0.9994** on the same data.

Two lessons, both already paid for once in this project:

* **An order-mismatched comparison of a set-valued output is meaningless**, and it
  fails in the direction that looks like a real defect. The same mistake produced a
  "484 px keypoint error" that was the backend returning the same points in a
  different order. Every comparison in `scripts/check_accuracy.py` and
  `eval/run_accuracy.py` now
  matches the correspondence explicitly, and the isolation metric (sample the
  torch map at the BACKEND's keypoints) exists to tell "wrong operator" apart from
  "keypoints moved".
* **Measure the operator before blaming it.** With the correspondence matched, the
  sampling operator's own error is 2e-05, i.e. exact. The interesting quantity was
  never the operator; it was the 0.043 px keypoint jitter an fp16 NPU introduces,
  which is invisible in the operator and unavoidable in the arithmetic.

## What was measured, and what is still open

Ten findings came out of building this, and most are counter-intuitive. They are
recorded because each one cost real time and each one would be hit again. The
first three are about the algorithms; the last two are about the *measurement*,
and both of those produced a false clean bill of health.

**1. `descriptor_dim` vs `input_dim`.** The released ALIKED+LightGlue already
consumes 128-d descriptors (`input_proj` is `Linear(256, 128)`); `descriptor_dim`
is the *internal* width and is a different knob. Worth stating because the two
names invite exactly that confusion.

**2. `round + clip + gather` avoids the CPU fallback but is not lossless.** The
keypoint score map is extremely sharp at full resolution — adjacent pixels differ
by up to 0.99 — so moving the sample from the sub-pixel position to the nearest
integer shifts the score by up to **0.66** (mean 0.065) and changes a
threshold-filtered keypoint set by **2–5 %**. Bilinear interpolation is a weighted
sum of four integer neighbours and is *equally* gather-friendly, so the graph can
have both properties at once: `GatherSampleBilinear` reproduces `grid_sample` to
**2.4e-07** with no `grid_sample` in the graph. It is the default; `score_mode=
"nearest"` keeps the literal round+clip+gather variant available and its cost is
now measured instead of assumed.

**3. The fallback detector was itself broken, and it kept failing in the direction
that looks like success.** This one took four rounds to get right, and each round
reported a confident number that was wrong in a different way. It is worth
recording in full because the failure mode is an audit that *agrees with you*.

The matcher had **65** nodes off the NPU, and successive versions of this script
reported **0**, then **29**, then **36**:

| version | what it did | reported | truth |
|---|---|---|---|
| 1 | `redirect_stdout`, prose-only pattern | 0 | 65 |
| 2 | + `node type = <Op>` pattern, unique-line count | "29 lines" | 65 |
| 3 | + `os.dup2` capture, `max()` across categories | 36 | 65 |
| 4 | + process-level capture, categories summed | **65** | 65 |

The four causes, each of which is independently sufficient to produce a false
clean result:

* **`redirect_stdout` cannot see the log at all.** It replaces `sys.stdout`, a
  Python object; the toolkit writes from C++ to fd 1. Measured: **0 bytes**
  captured while the build emitted 1.2 MB. The audit was grepping an empty string.
* **Counting unique LINES bounds the report at the number of op types.** One
  `node type = Einsum` line is deduplicated against 27 of them, so the number stops
  moving no matter how bad the graph gets.
* **`verbose` decides whether the warnings exist, and it must be set at
  construction.** Three settings, same graph:

  ```
  RKNN(verbose=False)                        480,129 bytes    0 warnings
  RKNN(verbose=False) + later set_log_level  866,686 bytes    0 warnings
  RKNN(verbose=True)                       1,216,961 bytes   36 warnings
  ```

  The post-hoc setter is the trap: it visibly grows the log while still omitting
  the warnings, so a fix that "looks like it worked" does not. `verbose` has to
  reach the CONSTRUCTOR, because `RKNN.__init__` is where
  `set_log_level_and_file_path` is called.
* **The log is split across two sinks with different flush behaviour, and no
  in-process capture sees both.** The `No lowering found` lines reach the terminal
  through a Python-level buffer that flushes *after* any fd redirect has been
  undone, so per-method captures returned 0 of them while the terminal received 76.
  Only capturing at the **process** level — build in a child, read its stdout and
  stderr — sees everything. `scripts/build_log_probe.py` does that, and every count
  is produced twice (parser and plain `str.count`) with a mismatch reported as
  failure rather than as a number.

Two counting errors were separate from the capture, and both made the total too
small. `max()` across categories assumed the categories were two views of the same
nodes; they are not — `No lowering found` names `Einsum`/`ScatterElements`, while
`will fallback to CPU` is all `Transpose`, so the total is their **sum**, 29 + 36.
And `sum(ops.values())` is a *breakdown* of the no-lowering count, not a third
category, so adding it double-counts (measured: 29 became 58).

What the 65 nodes are, and why they are two different problems:

* 27 `Einsum` are attention (`q·k` and `attn·v`) plus the soft-argmax weighting.
  RKNN has no `Einsum`, so they must be expressed as `MatMul`. This is a **pure
  graph rewrite with identical arithmetic** — verified at `max |Δscore| = 0`, see
  finding 6 — and it is the single highest-value change in the project, because
  attention is the whole matcher.
* 36 `Transpose` **lower fine and are then handed back**, because 512×512 exceeds
  the NPU's 8192-element transpose limit by 32×. They are a *warning*, not an
  error, which is exactly why a detector watching only for hard failures misses
  them. They vanish under the `MatMul` form.
* 2 `ScatterElements` are the corner assignments in `sigmoid_log_double_softmax`
  (`scores[:, :m, :n] = ...`, `scores[:, :-1, -1] = ...`). Same fix shape as the
  border mask: build the corner terms separately and `Concat`. This is the last
  item, and it is why the finished matcher is at **2**, not 0.

The `Einsum` nodes were missed once *before* any of this, too: an early stage-2
conversion reported "none", and it was believed because the *loader* printed an
error while the *audit* printed nothing. That combination is worse than a failure —
a failure stops you.

**4. Boolean operators and `nn.Unfold` do not reach the NPU.** Upstream ALIKE's
`simple_nms` is written with `|`, `&` and `~`, and the detector used
`nn.Unfold`. RKNN reports `No lowering found ... node type = Or` for the first and
cannot represent `Im2Col` at all; both silently degrade to a custom operator, and
the simulator then disagrees with the reference by 484 px. `simple_nms` is
therefore rewritten in float arithmetic (`or -> a+b-a*b`, `and -> a*b`, `>=`
instead of `==`, which is equivalent because a max-pool result is always one of
its inputs), verified **bit-identical** to the boolean original, and the 5×5
neighbour window is gathered by index arithmetic instead of unfolding. The flat
index of a neighbour is `idx + (dy*W + dx)` — an addition, not a division — which
also removes the integer `//`/`%` whose ONNX lowering needs boolean ops.

**6. `Einsum` → `MatMul` is exact, and it is worth 63 of the 65 nodes.** The
rewrite is in `scripts/ablate_attention.py` and it changes the *contraction*, not
the attention:

```python
sim    = torch.einsum("bhid,bhjd->bhij", qk0, qk1)   # no RKNN lowering
sim    = torch.matmul(qk0, qk1.transpose(-2, -1))    # same value, native op
m0     = torch.einsum("bhij,bhjd->bhid", attn01, v1)
m0     = torch.matmul(attn01, v1)
m1     = torch.einsum("bhji,bhjd->bhid", attn10.transpose(-2,-1), v0)
m1     = torch.matmul(attn10, v0)
```

`--verify` compares the patched matcher against the untouched class implementation
on real keypoints and asserts the tolerance: **`max |Δscore| = 0.000e+00`**. Every
projection, both softmaxes, the scale and the output projection are left as
upstream wrote them, so a discrepancy can only come from the contraction — which
is what makes the verification mean something rather than being a smoke test.

**The last line is the one that had to be DERIVED rather than read.** Upstream's
`m1` is `einsum("bhji, bhjd -> bhid", attn10.transpose(-2, -1), v0)` with
`attn10` of shape `(B, H, N0, N1)`. A `transpose(-2, -1)` swaps the last two axes,
giving `(B, H, N1, N0)` — both non-`D` axes — so the subscript `i` it is supposed
to contract against `d` is not there, and the call cannot be well-formed. It is
dead code on the deployed path (`bias=True` always selects the `flash` branch on
CUDA), which is why nobody has hit it.

So the substitution was determined by shapes instead of by reading the subscripts.
`scripts/probe_einsum.py` enumerates every plausible variant and prints which ones
survive; exactly **one** per direction is shape-valid, and it is a plain `matmul`
with no transpose on the attention matrix at all. Every transposed variant — the
one upstream writes included — fails. That is a stronger argument than either
reading the intent or trusting the original, and it is re-runnable.

The first attempt at this got it wrong in a way that only the numeric check caught:
it paired a `(H, N0, N1)` softmax with a value tensor whose last axis was 128
rather than 64 (`to_qk` and `to_v` are separate projections that happen to agree
today), and the shape error surfaced only because `--verify` ran on real tensors
rather than asserting the two forms "look equivalent".

**7. `n_heads` and `qkv` do not exist on this matcher, and getting that wrong
produces a sweep full of baselines.** The first version of the head ablation keyed
off `m.n_heads` and `m.qkv`. The actual attributes are `SelfBlock.num_heads` /
`SelfBlock.Wqkv` and `CrossBlock.heads` / `CrossBlock.to_qk` / `CrossBlock.to_v`,
and the head dimension is the INNERMOST one because `unflatten(-1, (heads, -1))`
splits by the outer factor. So `getattr` returned `None`, the slice was skipped,
and the sweep reported `heads 4->2` for eight rows that were all the baseline —
with parameter counts that dropped, because `conf.num_heads` had been changed even
though no tensor was. Two guards now make that impossible: the slicer returns
`(n_sliced, n_matched)` and the caller rejects a zero, and every config must
perturb the output on a non-degenerate probe before it is scored.

That last clause is also its own lesson: the first probe used the same random
descriptors for both images, which makes the true correspondence the identity and
therefore a fixed point of any layer count — so it reported "no change" for
*correct* ablations too. A probe that cannot fail is not a check.

**9. `ConstantOfShape` makes `load_onnx` fail, and the fix is to derive the zero
instead of declaring it.** Removing the last 2 `ScatterElements` was supposed to be
mechanical — replace a buffer-and-write with a `Concat` of the same pieces — and it
broke the conversion outright:

```
AttributeError: 'numpy.ndarray' object has no attribute 'data_type'
  in rknn/api/ir_graph.py, IRGraph.convert_to_fp32
```

Isolating it took a four-way bisect over the two rewrites and both halves of the
second (`scripts/probe_load_bisect.py`):

| variant | change | `load_onnx` |
|---|---|---|
| `base` | none | OK |
| `attn` | attention → `MatMul` | OK |
| `assign-mm` | + `MatchAssignment` einsum → `matmul` | OK |
| `assign-cc` | + `sigmoid_log_double_softmax` → `Concat` | **FAIL** |
| `assign-cc2` | same, but the zero is DERIVED | OK |
| `both` | everything, zero derived | OK |

The culprit is one line: `sim.new_zeros((b, 1, 1))`, the padding cell of the
assignment matrix. It exports as a **`ConstantOfShape`** node — which takes its
shape from an int64 *tensor input* rather than from an attribute, exactly the kind
of input a weight-walking pass misinterprets.

The replacement is arithmetically the same and introduces no new node type:
`sim[:, :1, :1] * 0.0`, a `Slice` of a tensor already in the graph times a scalar.
The failure was caused by a new KIND of node, so the fix is to introduce none.

This is worth recording because the fix is not "use `Concat`" or "avoid `Concat`" —
it is that **a rewrite can be numerically exact and still be un-convertible**, and
the only way to know is to run the conversion. `--verify` at `Δ = 0` says nothing
about whether the toolkit can parse the result; those are two different gates and
both have to pass.

**10. `GatherElements` is mishandled by the simulator.** `torch.gather(x, dim,
idx)` with a rank-2 index exports as `GatherElements`, and the RKNN simulator
returns plausible-looking but wrong values for it (descriptors at cosine 0.88
instead of 1.0 — the model still runs and the output ranges still look sane).
Only `index_select` on a flattened tensor with a rank-1 index emits a plain ONNX
`Gather`, which is what `gather_ops._flatten_gather` does now.

### Why descriptors are sampled bilinearly here (a change relative to upstream)

Upstream ALIKE samples descriptors at the *floor* of the sub-pixel keypoint. The
descriptor map is extremely high-frequency — cosine between **adjacent pixels** of
the normalised map is **mean 0.859, min 0.556** — so `floor` makes the sampled
descriptor a STEP function of the keypoint position: a 0.043 px jitter that
crosses an integer boundary substitutes a neighbouring pixel's descriptor.

`descriptor_interp="bilinear"` (the default here, `"floor"` kept for comparison)
makes the descriptor continuous in the keypoint, which is what makes the model
portable across backends that disagree in the last bits. Measured on the noise
reference dump, where the effect is worst, the minimum cosine went from **0.54 to
0.90**; on real images the descriptor agreement is 0.9997 either way, because real
imagery jitters less.

Why the shift rather than the offset: with `floor` the error is *zeroth order* in
the keypoint displacement (the sampled pixel changes discontinuously), with
bilinear it is *first order* (the descriptor changes by the sub-pixel weight
times the local gradient). Only the second is bounded by the keypoint accuracy,
and the keypoint accuracy is set by the NPU's fp16 arithmetic, which is not
something this project can change.

This is a deliberate deviation from the reference algorithm and it is measured as
such: both variants are exportable and the flag selects which. End-to-end numbers
in `results/accuracy.json` are for the bilinear default.

## Status

| deliverable | state |
|---|---|
| ONNX export scripts | done |
| RKNN conversion scripts (fp16 + int8 option, 3-layer op audit) | done |
| RKNN-deployable model structure files | done, verified |
| weights | documented in `Weights provenance`; not committed |
| structured pruning code | done (`pruning/structured.py`) |
| accuracy-recovery fine-tuning | **verified end to end** (`pruning/finetune.py`) |
| C++ inference sample + CMake | done, not compiled here (no aarch64 toolchain) |
| README | this file |
| **fp16 conversion, both stages** | **done, accuracy bar met** |
| **RKNN end-to-end vs torch** | **inliers identical**; 11 % fewer matches, higher precision |
| **matcher 9→7, fine-tuned** | **adopted** — −22 % params, `mAA` +0.011, 0 off-NPU nodes |
| **stage-1 graph, NPU-resident** | **done — 0 nodes fall back** |
| **stage-2 graph, NPU-resident** | **done — 0 nodes fall back** (was 65) |
| fallback audit itself | corrected 4×; only the process-level capture is complete |
| `Einsum` → `MatMul` + `Scatter*` → `Concat` | done, **verified exact** (`max |Δscore| = 0`) |
| structural ablation sweep | done, **results are lower bounds** (not retrained) |
| int8 path | **attempted, NOT viable on RK3588** — `results/int8_report.md` |
| ALIKE channel pruning | **measured, REJECTED** — no viable cut at these widths |
| **ALIKE head restructure** | **wired in — 0 off-NPU nodes, −23.5 % MACs, `n_inl_gt` unchanged** |
| on-device latency | **not measured** — needs the board |

### What is not done, and what it would take

* **int8 does not work on this target, and that is now measured rather than
  assumed** (`results/int8_report.md`). Stage 1 (`w8a8`) converts but the accuracy
  collapses — keypoint median offset 0.043 → **0.85 px**, descriptor cosine
  0.9997 → **0.896** — *and* it reintroduces **4 host-resident `Transpose` nodes**
  that the fp16 graph does not have, so residency has to be re-audited per
  precision. `w8a16`, the obvious fix, is rejected outright:
  `quantized_dtype = 'w8a16' not support in 'rk3588'`. Stage 2 fails to build at
  all: `cannot convert float NaN to integer` in min-max calibration, because the
  graph contains `-inf` from `F.logsigmoid` — the same `-inf` the fp16 path merely
  warns about, 369 times per run. Making int8 work would require changing the model
  (a finite floor for `logsigmoid`), not just a build flag.
* **Changing `num_heads` is not a compression lever at all**, so there is nothing
  to retrain *for*: `head_dim` is the only thing that changes, and the only
  parameter whose shape depends on the head count is `posenc.Wr` — **32 values out
  of 11.9 M**. It is an architecture hyperparameter to retrain for accuracy if
  someone wants to spend a run on it, not a way to make the model smaller. This
  retires what was previously listed here as the highest-value open experiment.
* **Depth is settled at 9→7** — 9→6 was run and is worse (`mAA` −0.021 against 9→7,
  no cross-modal gain), so there is nothing further to explore downward without a
  change of architecture. See *layers 9→6: where the curve turns* above.
* **The 9→7 model has now been re-exported end to end through RKNN** — this was the
  last open measurement and it is closed. `weights/optimized/` is the combined
  graph (gathered head + `MatMul`/`Concat` + 7 layers), verified at **0 off-NPU
  nodes**, and scored on both judges: `results/e2e_d7_ta.json` (torch vs ONNX, the
  export is exact — identical `n_inl_gt` / `n_inl@3`) and `results/e2e_d7_rknn.json`
  (ONNX vs RKNN fp16: `n_inl_gt` 225.5932 → 225.2203, i.e. **99.83 % of the correct
  correspondences retained**, `n_inl@3` 313.95 → 313.55, with 10–14 % fewer matches
  emitted and precision up 0.08 / 0.13). See the *Accuracy* section of `README.md`.
* **ALIKE channel pruning is measured and rejected**; the head restructure that
  replaces it is wired in and verified — see *Pruning the backbone: measured and
  rejected, and what to do instead* below.
* **On-device latency** needs the board. The simulator validates numerics only;
  no timing claim in this README comes from it.
* **The `n > 1 px` keypoint outliers** (1–3 of 512) come from near-ties at the
  rank-512 cut of `topk`: two candidates whose scores differ in the 4th decimal
  can swap rank between backends, and the loser is a genuinely different point.
  It is bounded (≤3 of 512 here) and does not compound, but it is the reason the
  max keypoint offset is 4 px rather than 0.05 px.

### Pruning the backbone: measured and rejected, and what to do instead

The backbone is **2.8 % of the parameters** (330 k of 11.95 M), which is why
channel pruning it looked like a non-starter and was left open for so long.
Measuring the compute instead reverses that: it is **35.9 % of the MACs**.

```
stage 1 (backbone + DKD)   6.5G MACs   35.9 %
stage 2 (matcher)         11.6G MACs   64.1 %
```

and one convolution inside it dominates everything else in the pipeline:

| component | MACs | share of pipeline |
|---|---|---|
| **`net.convhead2`** | **4.33 G** | **24.0 %** |
| `net.block1.conv2` | 0.60 G | 3.3 % |
| `net.block2.conv2` | 0.60 G | 3.3 % |
| all other convs | ~0.95 G | 5.2 % |
| matcher `transformers` | 11.5 G | 63.7 % |

#### The pruning result: no cut is viable

Two scans, both in `results/alike_channel_pruning.json`:

* **Whole-block zeroing** (`scripts/scan_alike_channels.py`) — every one of 16
  blocks is over the 0.02 tolerance, the best (`block4.conv2`) at 0.177 and the
  worst (`convhead2`) at 6.72. Zero within tolerance.
* **Fractional channel cuts** (`scripts/scan_channel_fractions.py`) — keep ratios
  0.90 / 0.75 / 0.50 on the eight highest-MAC blocks, measured on **both** outputs.
  **Zero viable configurations.** The best case, `block4.conv2` at keep 0.90, still
  moves the score map by L1 0.066 (3.3× the budget) *and* drops descriptor cosine
  to 0.9913 — where the entire fp16 NPU conversion costs only 0.9997.

The blocks are 16/32/64/128 wide, so "remove 10 % of the channels" means removing
1–12 channels out of a group that was trained jointly. There is no removable
redundancy at this scale, which is what the original note in `pruning/structured.py`
predicted; it is now measured rather than assumed.

#### The trap that scan had to avoid

`ALikeNet.forward` splits the head like this:

```python
descriptor_map = x[:, :-1, :, :]
scores_map     = torch.sigmoid(x[:, -1, :, :]).unsqueeze(1)
```

**The score is the LAST channel.** So pruning the 128 descriptor channels leaves the
score map **bit-identical** — measured `score L1 = 0.00000` at *every* keep ratio —
while the descriptor cosine collapses to 0.921 at keep 0.90. A sensitivity scan that
watched only the score map would report *zero damage* for a change that destroys the
quantity the matcher actually consumes.

This is the same class of error as the `heads 4→2` ablation and the order-unmatched
descriptor comparison: **a measurement that watches the wrong output reports
success.** Both outputs are therefore measured in every row, in the unit each is
consumed in — relative L1 for the score map (a raw value) and cosine for the
descriptor (used by angle, after L2 normalisation).

#### What to do instead: gather the descriptor instead of materialising it

`convhead2` is a **1×1 convolution** — a per-pixel linear map with no spatial
mixing — applied to all 512×512 pixels, producing 129 × 262,144 values. The detector
then reads **512 of those pixels**. So 33.6 M descriptor values are computed per
frame and **65,536 are used — 0.2 %**. Only the score channel genuinely needs to be
dense, because NMS and top-k act on the whole map.

Because a 1×1 conv is per-pixel linear, sampling commutes with it:

```
(W @ x)[:, P]  ==  W @ x[:, P]
```

so the descriptor rows can be applied to the *gathered features*:

```
before : convhead2(x) -> (B,129,H,W) -> normalise -> gather at K points
after  : score channel dense (1 of 129), gather x at K points,
         descriptor rows applied to the gathered tensor
```

**And the naive version of that idea is wrong.** Per-pixel normalisation is
non-linear, so it does *not* commute with bilinear interpolation:

```
normalize_perpixel(W @ bilinear(x))  !=  bilinear(normalize_perpixel(W @ x))
```

Three variants were therefore measured against the **real shipped computation**
(`GatherSampleBilinear` on the per-pixel-normalised dense map), at the sampler's
output — `scripts/probe_head_restructure.py`:

| variant | descriptor cos min | verdict |
|---|---|---|
| A — nearest gather, then the head | **0.905** | **wrong**: exact only against a *nearest* dense path, and this deployment ships `bilinear` |
| B — bilinear on raw features, then the head | 0.9997 | **an approximation**, not a substitution |
| **C — head + normalise PER CORNER, then combine** | **0.9999995** | **exact** |

Variant C is the shipped arithmetic restricted to the corners that are read: gather
the raw features at the four integer corners, apply the descriptor rows to each
corner, normalise each corner — which is what the dense path does per pixel — and
then combine with the bilinear weights.

| | result |
|---|---|
| sampler output, max &#124;Δ&#124; | **1.19e-07** — float32 associativity |
| descriptor cos min | 0.9999995 (the fp16 NPU conversion alone costs 0.9997) |
| `convhead2` cost | 4.329 G → **67.1 M MACs** — **64.5×** |
| whole pipeline | 18.10 G → **13.84 G MACs** — **−23.5 %** |

**This is a restructuring, not a truncation**, so it carries none of the "lower
bound" caveat that every pruning number in this project does.

There is a second benefit that MACs do not capture. The dense descriptor map is
`B × 128 × 512 × 512` = 67 M values — **134 MB at fp16**. Today it is materialised
on the NPU end to end even though 99.8 % of it is discarded. Removing it removes
that memory traffic too.

#### Wired in, and verified end to end

`model/gathered_head.py` implements variant C; `AlikeNet` gained `trunk()` and
`score_map()` so both paths share **one** copy of the backbone rather than two that
can drift; `AlikeStage(gathered_head=True)` is the new default
(`--no-gathered-head` reproduces the old graph).

| check | result |
|---|---|
| wired-in stage vs dense stage, real `forward`, 3 batches | keypoints **bit-identical**, scores **bit-identical**, descriptor cos min **0.9999998** |
| params | 329,168 both — the head weight is referenced, not copied |
| off-NPU nodes, converted graph | **0** (`no_lowering=0`, `will_fallback=0`) |
| op types introduced | **none** — `Gather`/`MatMul`/`Div`/`Mul`/`Add` were already in the graph |
| `.rknn` size | 12.13 MB → **10.07 MB** |
| ONNX end to end | **bit-identical** (`n_inl_gt` 225.5593, `n_inl` 245.6780, `mAA` 0.7355 — all unchanged) |
| RKNN end to end | HPatches `n_inl_gt` 225.2203 → **225.2373**; cross-modal `n_inl@3` 313.5900 → **313.5900** |

The bit-identical score path is the load-bearing check: it can only come out at
exactly zero if both paths really do run the same trunk. A re-derivation of the
trunk that was subtly wrong would agree on the descriptor and differ on the
keypoints.

This is also the one training-free change in the project that **makes the model
smaller, faster and no less accurate at the same time**, and the only reason it was
findable is that the MACs were counted instead of the parameters — on parameters the
backbone is 2.8 % and not worth a look.

## Speed

The conversion had to be *correct* before it was fast, and that is not a
formality here: of the four apparent problems found along the way, three were
measurement artefacts, and two of those pointed the WRONG WAY — one reported a
correct model as broken and one reported a broken graph as clean. A speed-first
pass would have optimised against the second one.

What has actually been changed, and what each is worth:

| change | status | measured effect |
|---|---|---|
| `grid_sample` → round/clip/gather | **done** | removes the only op that forced a full-size host round-trip per frame |
| `Unfold`/`Im2Col` → index arithmetic | **done** | removes an op RKNN cannot represent at all; the 5×5 window is one `Add` |
| boolean NMS → float arithmetic | **done** | `Or`/`And`/`Not` have no lowering; verified bit-identical |
| `Einsum` → `MatMul` (attention) | **done, opt-in** | 27 nodes removed, verified Δ=0 |
| `ScatterElements` → `Concat` (assignment) | **done, opt-in** | the other 2 removed, verified Δ=0 |
| **both together** | **done** | **65 → 0 off-NPU nodes**, `n_inl_gt` identical to torch |
| matcher layers 9→7 | **adopted** | −22 % params; `n_inl_gt` +0.03 HPatches / **+1.84** cross-modal, `mAA` +0.011 |
| matcher layers 9→6 | measured, rejected | −33 % params but `mAA` −0.021 vs 9→7 and no cross-modal gain |
| int8 | **measured, rejected** | stage 1 collapses (0.85 px) + 4 host nodes return; stage 2 will not build |
| **ALIKE head: dense descriptor map → gathered** | **done, wired in** | **−23.5 % total MACs** (18.10 → 13.84 G); `.rknn` 12.13 → 10.07 MB |
| ALIKE channel pruning | **measured, rejected** | no viable cut: every one breaks the score map or the descriptor |

The `Einsum` rewrite is the one that matters and it is **opt-in** rather than
default, for a specific reason: it is a hand-derived substitution for a line of
upstream code that is not well-formed, so it ships behind `--attention matmul`
alongside the `--verify` check that proves the substitution on real tensors. A
rewrite that has to be derived from shapes rather than read off the source should
be something you turn on deliberately.

Note what the residency numbers do *not* cover: 65 host nodes is a structural
cost, but how much wall-clock it costs depends on tensor sizes and how the runtime
schedules the copies. Removing them is unambiguously right; quantifying it needs
the board.

Every remaining lever changes the numerics, so each has to be re-run through
`eval/run_accuracy.py` (end to end, both judges) before it counts — and if it
changes the graph, through `scripts/build_log_probe.py` as well. That is the whole
point of having the protocol in the repository rather than in a notebook.

## Weights provenance

`alike_native_gl_s1/checkpoint_best.tar` — ALIKE backbone (fine-tuned, frozen
during matcher training) + LightGlue matcher fine-tuned for cross-modal matching
on 108,896 anchors with 16,650 registered thermal pairs; SDDH head frozen and
excluded from this deployment. See the training repository for the full lineage.

## Credits

- [glue-factory](https://github.com/cvg/glue-factory) — training and evaluation
- ALIKE / ALIKED — the backbone and DKD design
- [LightGlue](https://github.com/cvg/LightGlue) — the matcher
- Rockchip rknn-toolkit2
