# End-to-end accuracy, converted pipeline vs torch

`512px`, `512` keypoints, `alike_native_gl_s1`, seed 0.
HPatches viewpoint pairs: 59; registered RGB/TIR: 200.

The RKNN arm runs the **PC simulator**. It validates numerics; it does not measure latency.

## hpatches

| metric | torch | rknn-fp16 | delta |
|---|---|---|---|
| `mAA` | 0.7476 | 0.7343 | -0.0133 |
| `@3px` | 0.8136 | 0.7966 | -0.0169 |
| `@5px` | 0.8136 | 0.8136 | 0.0000 |
| `mprec@3px` | 0.6535 | 0.7335 | 0.0800 |
| `n_inl_gt` | 225.5932 | 225.2203 | -0.3729 |
| `n_inl` | 250.9492 | 258.7288 | 7.7797 |
| `H_error` | 22.7766 | 33.8737 | 11.0971 |
| `n_match` | 329.5424 | 295.2373 | -34.3051 |

## crossmodal

| metric | torch | rknn-fp16 | delta |
|---|---|---|---|
| `rep@3` | 0.6218 | 0.6221 | 0.0003 |
| `prec@3` | 0.7910 | 0.9246 | 0.1336 |
| `rec@3` | 0.9860 | 0.9843 | -0.0017 |
| `n_inl@3` | 313.9500 | 313.5450 | -0.4050 |
| `matches_mean` | 394.7650 | 337.6650 | -57.1000 |

## How to read this

* `mAA` / `@3px` / `@5px` are HPatches homography accuracy; `mprec@3px` is the fraction of matches consistent with the ground-truth homography and `n_inl` the RANSAC inlier count. Precision alone is not enough: a matcher that emits fewer matches can win on precision and deliver fewer usable correspondences.
* `rep@k` is a detector property (the matcher is not involved); `rec@k` is computed over the repeatable subset only, which is what isolates the matcher.
* A delta is only meaningful next to the pair count. On 600 pairs of a cross-modal judge, a between-backend delta below ~0.005 is inside the run-to-run spread this judge shows.
