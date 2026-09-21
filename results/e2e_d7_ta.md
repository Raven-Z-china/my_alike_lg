# End-to-end accuracy, converted pipeline vs torch

`512px`, `512` keypoints, `alike_native_gl_s1`, seed 0.
HPatches viewpoint pairs: 59; registered RGB/TIR: 200.

The RKNN arm runs the **PC simulator**. It validates numerics; it does not measure latency.

## hpatches

| metric | torch | onnx | delta |
|---|---|---|---|
| `mAA` | 0.7476 | 0.7485 | 0.0009 |
| `@3px` | 0.8136 | 0.8136 | 0.0000 |
| `@5px` | 0.8136 | 0.8136 | 0.0000 |
| `mprec@3px` | 0.6535 | 0.6536 | 0.0001 |
| `n_inl_gt` | 225.5932 | 225.5932 | 0.0000 |
| `n_inl` | 250.9492 | 251.2881 | 0.3390 |
| `H_error` | 22.7766 | 22.7675 | -0.0091 |
| `n_match` | 329.5424 | 329.5085 | -0.0339 |

## crossmodal

| metric | torch | onnx | delta |
|---|---|---|---|
| `rep@3` | 0.6218 | 0.6218 | 0.0000 |
| `prec@3` | 0.7910 | 0.7911 | 0.0001 |
| `rec@3` | 0.9860 | 0.9860 | -0.0000 |
| `n_inl@3` | 313.9500 | 313.9500 | 0.0000 |
| `matches_mean` | 394.7650 | 394.7100 | -0.0550 |

## How to read this

* `mAA` / `@3px` / `@5px` are HPatches homography accuracy; `mprec@3px` is the fraction of matches consistent with the ground-truth homography and `n_inl` the RANSAC inlier count. Precision alone is not enough: a matcher that emits fewer matches can win on precision and deliver fewer usable correspondences.
* `rep@k` is a detector property (the matcher is not involved); `rec@k` is computed over the repeatable subset only, which is what isolates the matcher.
* A delta is only meaningful next to the pair count. On 600 pairs of a cross-modal judge, a between-backend delta below ~0.005 is inside the run-to-run spread this judge shows.
