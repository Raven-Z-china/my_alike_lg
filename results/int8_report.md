# int8: attempted, and not viable on RK3588 for this model

> **Historical record.** The paths below (`models/...`) name the graphs that were
> converted at the time. Those int8 builds and the intermediate `*_mm` graphs have
> since been deleted, and `models/` is now `weights/`. The numbers are unchanged;
> only the paths are stale. The current equivalent of the int8 stage-1 experiment
> is `weights/original/alike_stage_bil.onnx`.

The brief asked for int8 as an **optional** path reported separately. It was built
and measured. The answer is that neither stage can use it on this target, for two
independent reasons — and both are worth recording because one of them is a
property of the toolkit and the other is a property of the model.

## Summary

| stage | config | result |
|---|---|---|
| 1 (ALIKE) | `w8a8`, `normal` | converts, **accuracy collapses**, and **4 host nodes come back** |
| 1 (ALIKE) | `w8a16` | **rejected by the toolkit**: `quantized_dtype = 'w8a16' not support in 'rk3588'` |
| 2 (LightGlue) | `w8a8`, `normal` | **build fails**: `cannot convert float NaN to integer` |

## Stage 1: accuracy collapses, and the graph gets worse

Same reference dump as the fp16 path (`models/alike_stage_bil_ref.npz`), so the
columns are directly comparable with `results/accuracy.json`:

| | fp16 | int8 w8a8 | ratio |
|---|---|---|---|
| keypoint offset, median | 0.0430 px | **0.8488 px** | 19.7x worse |
| keypoint offset, max | 4.02 px | **29.00 px** | 7.2x worse |
| descriptor cosine, median | 0.9997 | **0.8964** | 300x the error |
| descriptor cosine, p1 | 0.9908 | **0.7090** | — |
| **off-NPU nodes** | **0** | **4** | int8 *added* host nodes |

Two things here, and the second is the one that would be missed by reading only
the accuracy table:

**The accuracy loss is not a small degradation, it is a different detector.** A
median keypoint shift of 0.85 px with a 29 px worst case is not a model that is
slightly less precise; it is one whose keypoints are in different places.

**int8 reintroduces host-resident nodes that fp16 does not have.** The fp16 graph
is fully NPU-resident (0 of 0), and the quantised build brings back 4
`Transpose will fallback to CPU` warnings. Quantisation changes the data types, and
the NPU's transpose limit is applied to the quantised tensors differently — so a
graph that is clean at fp16 is not automatically clean at int8. **Residency has to
be re-checked per precision, not assumed from the fp16 result.**

### Why the detector is the hard part

Stage 1 does not end in a classifier or a feature; it ends in a dense score map,
then NMS, then `topk`, then a **soft-argmax over a 5x5 window at temperature 0.1**.
That last step is the sensitive one: at T=0.1 the weights are essentially `argmax`
with slight smoothing, so perturbing the score map changes *which pixel wins* and
the keypoint jumps by a whole pixel instead of drifting. This is exactly the error
shape seen above — median 0.85 px, tail to 29 px.

It is also why `w8a16` was the obvious candidate: 16-bit *activations* are where
the score map lives. RK3588 does not support it, which closes that route.

## Stage 2: the build fails, because the graph contains `-inf`

```
File "rknn/api/quant_utils.py", line 84, in _cal_scale_zp_min_max
ValueError: cannot convert float NaN to integer
```

Min-max calibration takes the range of each tensor; a tensor containing `-inf`
gives `-inf` for the minimum, and `-inf` in a scale/zero-point computation is
`NaN`. The graph legitimately contains `-inf` on two paths:

* `F.logsigmoid(-z)` in `sigmoid_log_double_softmax`, the unmatched-padding column
  and row of the assignment matrix. For a large positive `z` this underflows.
* the `Concat` borders that assemble that matrix.

**This is not speculation — the fp16 path already reports it, 369 times per run:**

```
W inference: The range [-inf, 0.0] of 'log_sigmoid-rs' is out of the float16!
W inference: The range [-inf, 0.0] of 'log_sigmoid_1-rs' is out of the float16!
W inference: The range [-inf, 0.0] of 'cat_36-rs' is out of the float16!
W inference: The range [-inf, 0.0] of 'cat_38-rs' is out of the float16!
```

So the same `-inf` that the fp16 path merely *warns* about is fatal to int8. A
warning that can be ignored at one precision is a hard failure at another, and the
warning count was the only place that information existed.

A workaround exists in principle — clamp the `logsigmoid` output before the concat,
or replace `-inf` with a large finite negative — but it changes the model's values
in the region that decides whether a point is left unmatched, so it would need the
full accuracy protocol to accept. It is not attempted here; the honest state is
that int8 stage 2 requires a model change, not just a build flag.

## What would have to change for int8 to be usable

1. **Stage 2's `-inf`** must become a finite floor (`F.logsigmoid` clamped, or a
   `-1e4`-style constant). Exact for the fp16 path, and it makes the graph
   calibratable. Requires re-verification end to end.
2. **Stage 1's score map** would need either a target that supports 16-bit
   activations (not RK3588) or a detector whose keypoints are less sensitive to it.
   The soft-argmax temperature is the knob — but that is a trained parameter, not a
   build option.
3. **Residency must be re-audited after any precision change.** The 4 Transpose
   fallbacks appeared only at int8, and nothing in the fp16 work would have
   predicted them.

## Reproduce

```bash
# stage 1: calibration samples must be batch-2 (NCHW), matching the export
python scripts/make_calib.py --phase images --n 200 --out models/calib_stage1.txt
python scripts/sweep_int8.py --out results/int8_sweep.json

# stage 2: samples come from stage-1 OUTPUT, not from images
python scripts/make_calib.py --phase tensors --n 200 \
    --out-dir models/calib_stage2 --out models/calib_stage2.txt
python scripts/convert_to_rknn.py models/lightglue_stage_mm.onnx --stage lightglue \
    --norm none --dtype i8 --dataset models/calib_stage2.txt
```

Two calibration-set details that cost a build each to find:

* **A `.npy` is read as NCHW; an image path is read as NHWC.** The two inputs do
  not agree, and the second error only appears once the first is fixed.
* **Stage 1 is exported at batch 2**, so the quantiser rejects batch-1 samples with
  `expect 'nhwc' like (2, 512, 512, 3)`. The fix is to supply batch-2 arrays, not
  to export a batch-1 model for calibration — that would calibrate a different
  graph from the one that ships.
