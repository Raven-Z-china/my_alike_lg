# `alike_lightglue_outputs/` — analysis-only scripts

Lives **outside** the deployment repository
(`../alike_lightglue_ONNX&RKNN_deploy/`) on purpose. Nothing over there reads
anything from here, so this directory can be moved, deleted, archived or
regenerated at any time without touching a single script or model. That is the
point of the split: a report is a record of a measurement, not an input to one.

**Tool output no longer lives here.** Reports are written to the repository's own
`reports/` directory (`paths.outputs()`); this folder holds only the scripts that
produce them.

```
alike_lightglue_outputs/
├── probe_*.py                analysis-only scripts, moved out of scripts/
├── scan_*.py
├── verify_gathered_stage.py
├── diag_restructure_gap.py
├── bench_onnx_speed.py       host CPU/GPU/TRT timing of the shipped ONNX graphs
├── run_sequence_trt.py       reference-vs-folder matching demo (writes match images
│                             to sequence_out/ here, report into the repo's reports/)
└── strip_checkpoint.py       maintenance: drops checkpoint payloads no path reads
```

## Relationship to the repository

Each script here bootstraps the repository onto `sys.path` from an explicit
`_REPO` constant at the top of the file, and raises with the path it tried if
that directory is not there. Change `_REPO` if you move this folder.

They are analysis-only — none of them appears in the deployment path, and the
repository contains no import of any of them. The two things they do need from
the repository are `model/` (to build the networks) and `paths.py` (to find the
checkpoints and the benchmark), both of which are read-only here.

## Output location

Every `--json` / `--report` / `--out` defaults to the repository's `reports/`
directory (`paths.outputs()`), which is git-ignored — reports are measurement
artefacts, regenerable from the documented commands. Point `OUTPUTS_ROOT`
elsewhere to relocate it, for example to a scratch disk:

```bash
export OUTPUTS_ROOT=/scratch/reports
```

`run_sequence_trt.py --vis` match images are the one exception: they go to
`sequence_out/` next to the script, because they are demo output, not reports.

## Not included

Nothing here refers to a private dataset or to any absolute path from the machine
this was developed on. Two inputs do live outside both directories, and are
resolved through the repository's `paths.py`:

| variable | what it points at |
|---|---|
| `HPATCHES_ROOT` | the open-source HPatches benchmark |
| `GLUEFACTORY_ROOT` | the training checkout, for the two tools that need it |

## Running

The environment is the repository's `alike` one (see its README):

```bash
conda activate alike
cd alike_lightglue_outputs
python probe_flops_split.py --checkpoint ../alike_lightglue_ONNX\&RKNN_deploy/weights/checkpoints/alike_native_gl_s1_9L.tar
```

To fold the whole thing back together, move these scripts into the repository's
`scripts/`, revert their `_REPO` bootstrap, and drop the `OUTPUTS_ROOT` override.
None of that is required to run or deploy the models.
