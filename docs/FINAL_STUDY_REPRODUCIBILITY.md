# FINAL STUDY — 24-run reproducibility and scientific status (26 October 2026)

## Experiment matrix and count

Eight conditions, seeds (0,1,2), 1,000,000 environment steps each on HalfCheetah-v5 and SAC. See `README.md` for all eight per-condition JSON filenames and **the exact recorded scientific source SHAs**. The machine-readable source of truth for finished run identity and provenance is `analysis/data/final_24run/run_inventory.csv`. Reusing the 2x256 baseline in two comparison families does not create a ninth condition or duplicate its three runs.

The original eight architectures and isolated component semantics are reviewed in `docs/INPUTNORM_ABLATION.md`, `docs/RESIDUAL_ABLATION.md` and `docs/LAYERNORM_ABLATION.md`. Historical validation docs intentionally preserve their original candidate timestamps; they should not be mistaken for the final execution status described here. The three PRs were merged in dependency order to `main` preserving **original scientific commit objects**, not squashed (PR #1 InputNorm, PR #2 Residual, PR #3 LayerNorm).

## Exact scientific procedure

Shared algorithm: SAC actor, twin online/target critics, automatic entropy temperature tuning, 1M replay transitions, minibatch 256, warm-up 5000 steps, gamma=0.99, tau=0.005, actor LR 0.0003, critic/entropy LR 0.001, compensated actor update frequency 2. Deterministic periodic evaluation every 25000 environment steps for five episodes, plus an independent 10-episode final evaluation at 1000000. No hyperparameter retuning for a particular architecture.

Run data: one unique per-run folder with `config.json`, `metadata.json`, `status.json`, `checkpoint.pt`, `train_metrics.csv`, `episode_metrics.csv`, `eval_metrics.csv` and optional TensorBoard. Source commits are recorded independently per run, along with clean Git state, Python/Torch/Gymnasium/MuJoCo/GPU environment, all evaluation episode returns and full resolved configuration.

**All 24** trainings completed and the original independent artifact audit passed. LayerNorm seed 1 and 2 must *not* be described as satisfying the original exclusive GPU access gate, however. The allowed idle kernel and monitoring deviation were external runner operations; their SAC source and architectures were unchanged. The external runner finished with exit 1 due extra third-party CUDA processes, despite completed Python training exit 0 and later manual independent artifact audit passes. Additional CUDA PIDs were observed in 50/2903 and 242/2886 process samples. GPU-utilization levels increased when these processes were detected, but this cannot measure per-process compute or prove changed policy quality.

## Audit and reproducibility levels

1. Historical **execution**: exact code SHA, environment, full outputs, P100 assigned GPU 0. Separate exception documents and process telemetry qualify two final LayerNorm runs.
2. Original **artifact audit**: SHA/config/metadata/checkpoint/schema/completion/finite-value/evaluation/episode checks by the original frozen auditor. All 24 passed; see archived audit evidence. This does *not* assert GPU exclusivity for the two qualified runs.
3. Versioned **secondary aggregation** in `analysis/data/final_24run/`: `run_inventory.csv` (24 rows), `learning_curves.csv` (320 rows), `final_summary.csv` (8 rows); includes original per-seed values for independent recalculation. The script `analysis/final_24run_reproduce.py verify-bundled` checks group/seed/step grids, recomputes every across-seed sample SD and rejects unlisted deviations.
4. Optional **raw replication check**: `analysis/final_24run_reproduce.py verify-raw --results-root results` requires the complete raw output directories. It checks source SHA, completed status, configs, ten/five-episode evaluation arrays, full 1M step grid, parameter and checkpoint existence, and equality to published final values. The full historical auditors remain authoritative.

## Archive integrity and retrieval

The group archived original runs on Gorina4 at `/bhome/u273406/ELE680_SCIENCE_24RUNS_20261026.tar.gz` and transferred the archive to the student's Windows PC, verifying a SHA-256 match:
`9c1220e0e8097e38a4f89b2d047f9262b5fab7da78e2e1123bd9ae8cee715a54`

This archive is **not** committed to GitHub; it contains checkpoint binaries and other ~hundreds-MB original run artifacts. The group must make it available to assessors if requested; this README does not claim an examiner has access to the local server or private archive. Inspect and unpack it into an appropriate directory preserving relative paths before running the raw verifier. Use the provided versioned secondary CSVs for plots when raw archive is unavailable.

## Independent analysis outputs

`analysis/final_24run_reproduce.py plots` creates width, depth and isolated 8×256 ablation learning curves using all **40** periodic evaluation steps, with band = **sample standard deviation across the three training seed means** (ddof=1). The final ten-episode evaluation mean per training seed is a different measurement, stored in `run_inventory.csv`, and the within-evaluation episode SD is not a between-seed uncertainty estimate. The primary summary retains the low naive 8×256 seed 2 and LayerNorm seed 0. Small n=3 prohibits strong inferential claims; neither a significant difference nor a general scale law was established.

No new GPU jobs are required to regenerate the secondary curves from archived summaries; the script never launches training. For reproducibility, original scientific SHAs should be used (printed by the read-only `plan` command), and appropriate hardware scheduling/permissions must be obtained.
