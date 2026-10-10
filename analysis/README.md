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
