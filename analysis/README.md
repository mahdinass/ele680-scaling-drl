# ELE680 Experiment Results

## Baseline: SAC on HalfCheetah-v5

Architecture: 2 hidden layers, 256 units per layer.
Training budget: 1,000,000 environment steps.
Seeds: 0, 1, 2.

Results were generated on UiS Gorina4 using the
frozen ELE680 experiment infrastructure.

## Data files

Each run includes:

- config.json: Exact training configuration.
- metadata.json: Software versions, Git commit, GPU and provenance.
- status.json: Completion status.
- train_metrics.csv: Optimization and training diagnostics.
- episode_metrics.csv: Training episode statistics.
- eval_metrics.csv: Deterministic evaluation returns.

Evaluation was performed every 25,000 steps using
5 episodes, with a final evaluation of 10 episodes.

## Baseline final evaluation

Seed 0: 12250.479377
Seed 1: 11335.909955
Seed 2: 12132.360888

Mean across seeds: 11906.25
Sample standard deviation across seeds: 497.45

These statistics describe variation between independent
training seeds, not variation between evaluation episodes.

## Provenance

Baseline Git commit:
1a9f2405b5fdb9cffa379b72f569a0a3a01515e4

The original complete run directories, including PyTorch
checkpoints and TensorBoard files, remain on Gorina4
under /bhome/u273406/ele680-scaling-drl/results/.

This directory contains copies of the scientific CSV/JSON
artifacts for analysis and reproducibility.

## Width comparison (2x256, 2x512, 2x1024)

Reproducible three-architecture comparison, generated from the tracked copies
under `analysis/data/` (`baseline/`, `width_2x512/`, `width_2x1024/`):

    .\.venv\Scripts\python.exe analysis\plot_width_comparison.py [OUT_DIR] [--no-figures]
    .\.venv\Scripts\python.exe -m unittest discover -s analysis/tests -v

Outputs are written to `analysis/figures/width_comparison/` (figures 01-08 as PNG
and PDF, five CSV/JSON/Markdown files); `width_comparison_summary.md` is the report.
The three architectures were trained at different Git commits (see the report).
All standard deviations across training seeds are sample SDs (ddof=1, n=3);
`final_within_seed_episode_sd` is the population SD over the evaluation episodes of
one run and is a different quantity.

### `width_comparison_stats.json` schema (version 2)

The file carries a `schema` block. Compared with the earlier two-width version:
`effect_width_vs_baseline` was removed and replaced by `<metric>.pairwise_differences`
(keys `2×512 - 2×256`, `2×1024 - 2×256`, `2×1024 - 2×512`, each wider minus narrower;
the `2×512 - 2×256` values are unchanged); per-architecture entries now contain
`n, mean, sd, min, max, range, cv, per_seed`; and
`runtime_flagged_slow_(>1.5x_median_of_six)` was renamed
`runtime_flagged_slow_(>1.5x_median_of_nine)`. No other file in the repository reads
the removed key.
