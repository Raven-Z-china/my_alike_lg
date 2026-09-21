# End-to-end accuracy, converted pipeline vs torch

`512px`, `512` keypoints, `alike_native_gl_s1`, seed 0.
HPatches viewpoint pairs: 59; registered RGB/TIR: 200.

The RKNN arm runs the **PC simulator**. It validates numerics; it does not measure latency.

## hpatches

| metric | torch | onnx | delta |
|---|---|---|---|
| `mAA` | 0.7365 | 0.7355 | -0.0010 |
| `@3px` | 0.7966 | 0.7966 | 0.0000 |
| `@5px` | 0.8136 | 0.8136 | 0.0000 |
| `mprec@3px` | 0.6594 | 0.6593 | -0.0001 |
| `n_inl` | 246.6949 | 245.6780 | -1.0169 |
| `H_error` | 20.0474 | 20.0576 | 0.0101 |
| `n_match` | 326.0169 | 326.0508 | 0.0339 |

## crossmodal

| metric | torch | onnx | delta |
|---|---|---|---|
| `rep@3` | 0.6218 | 0.6218 | 0.0000 |
| `prec@3` | 0.7998 | 0.7997 | -0.0001 |
| `rec@3` | 0.9862 | 0.9862 | -0.0000 |
| `n_inl@3` | 313.9950 | 313.9950 | 0.0000 |
| `matches_mean` | 390.5200 | 390.5600 | 0.0400 |

## How to read this

* `mAA` / `@3px` / `@5px` are HPatches homography accuracy; `mprec@3px` is the fraction of matches consistent with the ground-truth homography and `n_inl` the RANSAC inlier count. Precision alone is not enough: a matcher that emits fewer matches can win on precision and deliver fewer usable correspondences.
* `rep@k` is a detector property (the matcher is not involved); `rec@k` is computed over the repeatable subset only, which is what isolates the matcher.
* A delta is only meaningful next to the pair count. On 600 pairs of a cross-modal judge, a between-backend delta below ~0.005 is inside the run-to-run spread this judge shows.
