# End-to-end accuracy, converted pipeline vs torch

`512px`, `512` keypoints, `alike_native_gl_s1`, seed 0.
HPatches viewpoint pairs: 59; registered RGB/TIR: 200.

The RKNN arm runs the **PC simulator**. It validates numerics; it does not measure latency.

## hpatches

| metric | torch | rknn-fp16 | delta |
|---|---|---|---|
| `mAA` | 0.7365 | 0.7499 | 0.0134 |
| `@3px` | 0.7966 | 0.7797 | -0.0169 |
| `@5px` | 0.8136 | 0.8136 | 0.0000 |
| `mprec@3px` | 0.6594 | 0.7175 | 0.0581 |
| `n_inl_gt` | 225.5593 | 225.2203 | -0.3390 |
| `n_inl` | 246.6949 | 258.2712 | 11.5763 |
| `H_error` | 20.0474 | 19.5995 | -0.4479 |
| `n_match` | 326.0169 | 300.1017 | -25.9153 |

## crossmodal

| metric | torch | rknn-fp16 | delta |
|---|---|---|---|
| `rep@3` | 0.6218 | 0.6221 | 0.0003 |
| `prec@3` | 0.7998 | 0.9013 | 0.1015 |
| `rec@3` | 0.9862 | 0.9845 | -0.0017 |
| `n_inl@3` | 313.9950 | 313.5900 | -0.4050 |
| `matches_mean` | 390.5200 | 346.3950 | -44.1250 |

## How to read this

* `mAA` / `@3px` / `@5px` are HPatches homography accuracy; `mprec@3px` is the fraction of matches consistent with the ground-truth homography and `n_inl` the RANSAC inlier count. Precision alone is not enough: a matcher that emits fewer matches can win on precision and deliver fewer usable correspondences.
* `rep@k` is a detector property (the matcher is not involved); `rec@k` is computed over the repeatable subset only, which is what isolates the matcher.
* A delta is only meaningful next to the pair count. On 600 pairs of a cross-modal judge, a between-backend delta below ~0.005 is inside the run-to-run spread this judge shows.
