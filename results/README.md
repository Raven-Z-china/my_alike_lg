# The measurement record (kept, but no longer used by the code)

Every number quoted in the top-level `README.md` traces to one of the files here.
They are the raw output of the tools, kept rather than regenerated on demand,
because the expensive ones need a GPU (accuracy runs), the `rknn` environment
(conversions), or both — and because several of them are **negative** results,
which are the hardest to re-derive and the easiest to lose.

> **This directory is detached from the code.** Tools now write to
> `outputs/reports/` (outside the repository — see `paths.outputs()`), and
> nothing in the repository reads anything back from either location. This folder
> is a frozen record: it may be archived or removed without affecting a single
> script or model. Do not add new output here.

`weights/` is what you need to run or deploy the models; neither this directory
nor the outputs directory is required.

## Read this before comparing anything here

### 1. The `crossmodal` sections are historical

Nineteen data files contain a `crossmodal` block. It came from a second judge that scored
the pipeline on a registered RGB/thermal dataset, and **the harness no longer has
that arm** — `eval/run_accuracy.py` now measures HPatches only, and
`paths.bimodal()` / `--crossmodal` are gone.

The data is left in place rather than stripped: it is part of the record of how
the conclusions were reached, and deleting it would make the accompanying notes
unverifiable. But be aware of what follows:

* Those blocks **cannot be reproduced** from the current code, and they need a
  private dataset that is not in this repository.
* The `hpatches` blocks in the same files **are** reproducible, and are what the
  top-level `README.md` quotes.
* Where the two disagree, the HPatches figure is the one this repository stands
  behind. The historical table that used the cross-modal column lives in
  `details.md`.

The affected files are the `e2e_*`, `ablation_matcher*` and `head_restructure`
entries listed below, plus `details.md`, which is where the historical tables
built on that column were written up.

### 2. Paths in these files are labels, not locations

Reports name a file by role — `<checkpoint:alike_native_gl_s1_d7.tar>`,
`<onnx:alike_stage_gath.onnx>` — rather than by an absolute path. That is
deliberate (`scripts/portable_path.py` enforces it): a committed report should not
publish the filesystem layout of the machine that produced it, and an absolute
path would be unresolvable anyway.

One limitation, stated rather than hidden: both the 9-layer model and its 9→7
recovery were trained from a file called `checkpoint_best.tar`, so the label alone
does not say which. The distinguishing information is the file's own name
(`ablation_matcher_d7_ft.json`) and its `what` field.

### 3. `accuracy.json`: the `summary` block is a mean over pairs

The seven values under `summary` are each the **mean over the 8 pairs** of that
pair's own statistic, the per-pair statistic being a median / max / 1st percentile
over that pair's keypoints. It is therefore *not* a percentile pooled over all
4096 keypoints, and `kp_rknn_max_px` is a mean of 8 per-pair maxima rather than a
global maximum — the file states this itself, in `summary_aggregation`, which
`scripts/check_accuracy.py` writes. For tails, read the per-pair `pairs` entries.
The "0.043 px" and "0.99971" quoted in the top-level `README.md` are these summary
values.

### 4. Some reports predate later reorganisations

Files were produced across several renames of the source directories. The
measurements are unaffected; only the labels would have changed. Nothing here was
retro-actively corrected, because guessing a rename mapping afterwards is how a
record stops being trustworthy.

## Regenerating a report

Tools write to whichever directory `paths.outputs()` resolves to
(`outputs/reports/` by default, `OUTPUTS_ROOT` to relocate), so substitute that
for `<reports>` below. The analysis-only scripts that produced several of these
now live in that same outputs directory rather than in `scripts/`.

| file(s) | produced by |
|---|---|
| `e2e_*.json`, `e2e_*.md` | `eval/run_accuracy.py --json <reports>/... --report ...` |
| `accuracy.json` | `scripts/check_accuracy.py --out <reports>/accuracy.json` |
| `ablation_matcher*.json`, `ablation_matcher.md` | `scripts/ablate_matcher.py --json ... --report ...` |
| `invalid_head_ablation.json` | `scripts/ablate_matcher.py` — the head-surgery arm, which is refused by design |
| `rknn_*.json` | `scripts/convert_to_rknn.py --report <reports>/...` |
| `probe_*.json` | the matching probe script (now in the outputs directory) |
| `bench_stages.json` | `scripts/bench_stages.py --out <reports>/bench_stages.json` |
| `int8_sweep.json` | `scripts/sweep_int8.py --out <reports>/int8_sweep.json` |
| `head_restructure.json` | `probe_head_restructure.py` (outputs directory) |
| `gathered_stage_verify.json` | `verify_gathered_stage.py` (outputs directory) |
| `alike_channel_scan.json` | `scan_alike_channels.py` (outputs directory) |
| `alike_channel_fractions.json` | `scan_channel_fractions.py` (outputs directory) |
| `vendored_matcher.json` | `scripts/verify_vendored_matcher.py --json ...` |
| `int8_report.md` | written by hand; the int8 investigation's conclusions |
| `details.md` | the long-form engineering log for the whole project |

`details.md` and `int8_report.md` are not tool output — they are written notes,
which is why they do not follow the pattern above.

## Index

**End-to-end accuracy** (`eval/run_accuracy.py`), one file per configuration
compared. `_ta` = torch vs ONNX; `_rknn` = ONNX vs the RKNN simulator.

| file | configuration |
|---|---|
| `e2e_final` | 9-layer baseline, all three backends |
| `e2e_einsum` | upstream `Einsum` matcher |
| `e2e_mm` | matcher after the `Einsum`→`MatMul` rewrite, still 9 layers |
| `e2e_gathered` | + the keypoint-gathered descriptor head |
| `e2e_d7_ta`, `e2e_d7_rknn` | the shipped pair: gathered head + `MatMul` + 7 layers |
| `e2e_rknn` | 9-layer `MatMul` graph on the simulator |

**Conversion and residency** — `rknn_alike_*` and `rknn_lg_*` are per-model
conversion audits (fallback node counts, numerics against the ONNX reference);
`probe_*` are the narrower questions, including `probe_match_drop` (which stage
loses the 8–11 % of matches) and `probe_einsum` (enumerating the valid `MatMul`
substitutions).

**Structure** — the depth ablation and its fine-tuned arms, the rejected
head-count surgery, the descriptor-head restructure, the backbone channel scans,
and the vendor certification.

**int8** — `int8_sweep.json` plus `int8_report.md`: measured, and not viable on
RK3588 for this model.
