# ELE680 SAC baseline — descriptive results

- Environment: HalfCheetah-v5; actor and both critics: 2×256 MLP
- Three training seeds (0, 1, 2), each trained for 1,000,000 steps
- Periodic: 5 evaluation episodes every 25,000 steps (40 checkpoints)
- Final: 10 deterministic evaluation episodes per trained seed
- Source Git revision for all runs: `1a9f2405b5fdb9cffa379b72f569a0a3a01515e4`

## Per-run metrics

|   seed |   final_mean_return |   final_episode_sd |   mean_periodic_return |   runtime_hours |
|-------:|--------------------:|-------------------:|-----------------------:|----------------:|
|      0 |             12250.5 |             133.93 |                9866.49 |            5.29 |
|      1 |             11335.9 |              66.36 |                9793.17 |            5.08 |
|      2 |             12132.4 |             117.24 |               10074.9  |           12.33 |

**Final evaluation across training seeds: 11,906.25 ± 497.45**
(mean ± sample SD of the *three per-seed final means*; n=3).
The per-seed `final_episode_sd` is within-seed episode variation, not between-seed variation.

## Interpretation boundaries

- This baseline alone does not establish architecture scaling effects.
- The shaded learning-curve band represents between-training-seed sample SD, not a confidence interval.
- The final returns use 10 episodes; the learning curves use periodic 5-episode evaluations.
- SPS is a cumulative throughput measure; the seed-2 slowdown cause remains unknown.
- The last periodic evaluation and the final evaluation reuse some deterministic episode seeds; do not treat those as independent samples.
- Diagnostics are moving averages for display; raw CSV files remain unchanged.

## Reproduce figures

```powershell
.\.venv\Scripts\python.exe .\analysis\plot_baseline.py
```
