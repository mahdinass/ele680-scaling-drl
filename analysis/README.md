# ELE680 Experiment Results

## Complete 24-run study

The final study contains eight conditions and three seeds per condition. Its
authoritative summary data are in [`data/final_24run/`](data/final_24run/); the
[final scientific status](../docs/FINAL_STUDY_REPRODUCIBILITY.md) records provenance,
limitations, and the GPU-overlap qualification for two LayerNorm seeds. The baseline
and width sections below describe the earlier analysis stages.

From the repository root, with Python 3.12 in an activated virtual environment:

```bash
python -m pip install -r requirements/analysis.txt
python -m unittest discover -s analysis/tests -v
python analysis/final_24run_reproduce.py verify-bundled
python analysis/final_24run_reproduce.py plots --output-dir analysis/generated/final_24run
python analysis/plot_width_comparison.py analysis/generated/width_comparison
```

These commands use tracked CSV/JSON data and need no GPU or PyTorch. Generated
figures are written under the ignored `analysis/generated/` directory. The original
baseline script has a fixed output directory; to regenerate its figures without
overwriting the tracked originals, override that output in the current process:

```bash
python -c "from pathlib import Path; import analysis.plot_baseline as p; p.OUT=Path('analysis/generated/baseline'); p.main()"
```

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

The complete original run directories, including PyTorch checkpoints and
TensorBoard files, are archived separately by the group. They are available to
assessors on request; server access is not required for the tracked analyses.

This directory contains copies of the scientific CSV/JSON
artifacts for analysis and reproducibility.

## Width comparison (2x256, 2x512, 2x1024)

Reproducible three-architecture comparison, generated from the tracked copies
under `analysis/data/` (`baseline/`, `width_2x512/`, `width_2x1024/`):

    python analysis/plot_width_comparison.py analysis/generated/width_comparison [--no-figures]
    python -m unittest discover -s analysis/tests -v

The command above writes to `analysis/generated/width_comparison/` (figures 01-08 as PNG
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
