# Matcher structural ablations

Trained weights truncated, not retrained. `512px`, `512` keypoints, baseline matcher 11.885 M params.
HPatches 120 viewpoint pairs, registered RGB/TIR 150 pairs, seed 0.

**Every number is a lower bound.** The surviving layers were trained with the removed ones present, so a config that scores well truncated is a safe choice and a config that scores badly may still be fine after a retrain. The reverse reading is not valid.

## hpatches (delta vs baseline)

| config | params | `mAA` | `@3px` | `@5px` | `mprec@3px` | `n_inl_gt` | `n_inl` |
|---|---|---|---|---|---|---|---|
| `baseline` | 100.0% | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| `depth-8` | 88.9% | +0.0114 | -0.0339 | +0.0000 | +0.0020 | +0.3729 | +2.5593 |
| `depth-7` | 77.8% | -0.0097 | -0.0339 | -0.0169 | +0.0044 | +0.1017 | -0.0169 |
| `depth-6` | 66.8% | +0.0121 | +0.0000 | +0.0000 | +0.0116 | -3.1017 | -3.8644 |
| `heads-2` | 82.6% | +0.0096 | -0.0169 | +0.0000 | -0.0063 | -36.4237 | -37.9153 |
| `agg-uniform` | 100.0% | +0.0074 | -0.0169 | +0.0000 | +0.0126 | +0.4068 | +4.8136 |
| `depth-8+heads-2` | 73.4% | +0.0238 | +0.0000 | +0.0169 | -0.0041 | -37.5424 | -38.9831 |
| `depth-7+heads-2` | 64.3% | +0.0047 | +0.0000 | -0.0169 | -0.0030 | -37.2034 | -40.4915 |
| `depth-7+agg-uniform` | 77.8% | +0.0108 | +0.0000 | +0.0000 | +0.0130 | -0.2881 | +1.2712 |

## crossmodal (delta vs baseline)

| config | params | `rep@3` | `prec@3` | `rec@3` | `n_inl@3` | `matches_mean` |
|---|---|---|---|---|---|---|
| `baseline` | 100.0% | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| `depth-8` | 88.9% | +0.0000 | +0.0003 | -0.0001 | -0.0267 | -0.1933 |
| `depth-7` | 77.8% | +0.0000 | +0.0054 | -0.0002 | -0.0467 | -2.6600 |
| `depth-6` | 66.8% | +0.0000 | +0.0296 | -0.0008 | -0.2333 | -14.2800 |
| `heads-2` | 82.6% | +0.0000 | +0.0624 | -0.1595 | -47.4333 | -85.6000 |
| `agg-uniform` | 100.0% | +0.0000 | +0.0187 | -0.0008 | -0.2133 | -9.2333 |
| `depth-8+heads-2` | 73.4% | +0.0000 | +0.0684 | -0.1630 | -48.4800 | -88.8067 |
| `depth-7+heads-2` | 64.3% | +0.0000 | +0.0667 | -0.1643 | -48.8800 | -88.7933 |
| `depth-7+agg-uniform` | 77.8% | +0.0000 | +0.0253 | -0.0023 | -0.6400 | -12.8533 |

## Failures

* none

## Which lever this says to pull

Read `n_inl` before `mAA`. A matcher that keeps 95% of its inliers is still usable; one that drops to 60% is not, and `mAA` can stay flat through both because the surviving matches are still consistent with the ground truth. `n_inl@3` on the cross-modal judge is the sharpest signal in the table.
