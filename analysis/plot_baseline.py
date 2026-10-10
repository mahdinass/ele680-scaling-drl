"""Reproducible ELE680 SAC baseline figures from tracked per-run CSV/JSON files.

Run from repository root: .venv\\Scripts\\python.exe analysis\\plot_baseline.py
Original run files are not modified. No ML/GPU packages are required.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Headless: save files without requiring an interactive window.
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "data" / "baseline"
OUT = ROOT / "figures" / "baseline"
SEEDS = (0, 1, 2)
EXPECTED_COMMIT = "1a9f2405b5fdb9cffa379b72f569a0a3a01515e4"
EXPECTED_STEPS = np.arange(25_000, 1_000_001, 25_000, dtype=np.int64)


def run_dir(seed: int) -> Path:
    return DATA_ROOT / f"BASELINE_1M_seed{seed}_20261009"


def save_figure(fig: plt.Figure, filename: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{filename}.{ext}", dpi=210, bbox_inches="tight")
    plt.close(fig)


def finish_axes(ax: plt.Axes, ylabel: str) -> None:
    ax.set_xlabel("Environment steps")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.22)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(6, 6))


def load_run(seed: int):
    folder = run_dir(seed)
    required = (
        "config.json", "metadata.json", "status.json",
        "eval_metrics.csv", "train_metrics.csv", "episode_metrics.csv",
    )
    missing = [item for item in required if not (folder / item).is_file()]
    if missing:
        raise FileNotFoundError(f"Seed {seed}: missing {missing} in {folder}")

    config = json.loads((folder / "config.json").read_text(encoding="utf-8-sig"))
    meta = json.loads((folder / "metadata.json").read_text(encoding="utf-8-sig"))
    status = json.loads((folder / "status.json").read_text(encoding="utf-8-sig"))
    if status.get("status") != "completed" or status.get("steps_completed") != 1_000_000:
        raise ValueError(f"Seed {seed}: training status is not completed at 1M steps")
    if meta.get("seed") != seed or meta.get("git_commit") != EXPECTED_COMMIT or meta.get("git_dirty") is not False:
        raise ValueError(f"Seed {seed}: provenance/commit mismatch")
    if meta.get("parameter_counts", {}).get("total") != 217870:
        raise ValueError(f"Seed {seed}: unexpected parameter count")
    if config.get("seed") != seed or config.get("total_steps") != 1_000_000:
        raise ValueError(f"Seed {seed}: config seed/step mismatch")
    if meta.get("env_id") != "HalfCheetah-v5":
        raise ValueError(f"Seed {seed}: unexpected environment")

    evaluation = pd.read_csv(folder / "eval_metrics.csv")
    train = pd.read_csv(folder / "train_metrics.csv")
    needed_eval = {"step", "kind", "num_episodes", "mean_return", "std_return"}
    needed_train = {"step", "sps", "critic_loss", "alpha"}
    if not needed_eval <= set(evaluation.columns):
        raise ValueError(f"Seed {seed}: unexpected evaluation columns: {list(evaluation.columns)}")
    if not needed_train <= set(train.columns):
        raise ValueError(f"Seed {seed}: unexpected training columns: {list(train.columns)}")

    periodic = evaluation[evaluation["kind"] == "periodic"].sort_values("step").reset_index(drop=True)
    final = evaluation[evaluation["kind"] == "final"]
    if len(periodic) != 40 or len(final) != 1 or not np.array_equal(periodic["step"].to_numpy(), EXPECTED_STEPS):
        raise ValueError(f"Seed {seed}: expected 40 scheduled periodic evals and one final eval")
    if not (periodic["num_episodes"] == 5).all() or int(final.iloc[0]["num_episodes"]) != 10:
        raise ValueError(f"Seed {seed}: episode count mismatch")
    if int(final.iloc[0]["step"]) != 1_000_000:
        raise ValueError(f"Seed {seed}: unexpected final evaluation step")
    if not np.isfinite(periodic[["mean_return", "std_return"]].to_numpy(dtype=float)).all():
        raise ValueError(f"Seed {seed}: nonfinite evaluation metrics")
    if not np.isfinite(final[["mean_return", "std_return"]].to_numpy(dtype=float)).all():
        raise ValueError(f"Seed {seed}: nonfinite final metrics")
    if not math.isclose(float(meta["final_eval"]["mean_return"]), float(final.iloc[0]["mean_return"]), abs_tol=1e-5):
        raise ValueError(f"Seed {seed}: metadata and CSV final returns disagree")
    if train.empty or not train["step"].is_monotonic_increasing:
        raise ValueError(f"Seed {seed}: training log is empty or out of order")
    for metric in ("sps", "critic_loss", "alpha"):
        if not np.isfinite(pd.to_numeric(train[metric], errors="coerce").to_numpy()).all():
            raise ValueError(f"Seed {seed}: invalid {metric} values")

    return config, meta, periodic, final, train


def main() -> None:
    evaluations = {}
    training = {}
    metadata = {}
    configs = {}
    finals = {}

    for seed in SEEDS:
        config, meta, periodic, final, train = load_run(seed)
        evaluations[seed] = periodic
        training[seed] = train
        metadata[seed] = meta
        configs[seed] = config
        finals[seed] = final.iloc[0]

    # Compare protocol fields. run_id and seed are intentionally different.
    for seed in (1, 2):
        canonical = {k: v for k, v in configs[0].items() if k not in ("seed", "run_id")}
        candidate = {k: v for k, v in configs[seed].items() if k not in ("seed", "run_id")}
        if canonical != candidate:
            raise ValueError(f"Seed {seed}: baseline config differs from seed 0 beyond seed/run_id")

    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False})

    aggregate = pd.DataFrame({"step": EXPECTED_STEPS})
    for seed in SEEDS:
        aggregate[f"seed_{seed}"] = evaluations[seed]["mean_return"].to_numpy(dtype=float)
    matrix = aggregate[[f"seed_{s}" for s in SEEDS]].to_numpy()
    aggregate["mean_return"] = matrix.mean(axis=1)
    aggregate["sd_across_seeds"] = matrix.std(axis=1, ddof=1)
    aggregate.to_csv(OUT / "baseline_periodic_aggregate.csv", index=False)

    summary = pd.DataFrame([
        {
            "seed": seed,
            "final_mean_return": float(finals[seed]["mean_return"]),
            "final_episode_sd": float(finals[seed]["std_return"]),
            "mean_periodic_return": float(evaluations[seed]["mean_return"].mean()),
            "runtime_hours": float(metadata[seed]["runtime_seconds"]) / 3600,
            "total_params": int(metadata[seed]["parameter_counts"]["total"]),
            "git_commit": metadata[seed]["git_commit"],
        }
        for seed in SEEDS
    ])
    summary.to_csv(OUT / "baseline_summary.csv", index=False)
    mean_final = float(summary["final_mean_return"].mean())
    sd_final = float(summary["final_mean_return"].std(ddof=1))

    fig, ax = plt.subplots(figsize=(10, 5.8))
    for seed in SEEDS:
        ax.plot(evaluations[seed]["step"], evaluations[seed]["mean_return"], lw=1.8, label=f"Seed {seed}")
    finish_axes(ax, "Mean return over 5 evaluation episodes")
    ax.set_title("SAC baseline: deterministic evaluation learning curves")
    ax.legend()
    save_figure(fig, "01_learning_curves_by_seed")

    fig, ax = plt.subplots(figsize=(10, 5.8))
    steps = aggregate["step"].to_numpy()
    mu = aggregate["mean_return"].to_numpy()
    sd = aggregate["sd_across_seeds"].to_numpy()
    ax.plot(steps, mu, lw=2, label="Mean (3 training seeds)")
    ax.fill_between(steps, mu - sd, mu + sd, alpha=0.23, label="±1 sample SD between seeds")
    finish_axes(ax, "Mean return over 5 evaluation episodes")
    ax.set_title("SAC baseline: mean learning curve and between-seed variation")
    ax.legend()
    save_figure(fig, "02_learning_curve_mean_sd")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.bar([str(s) for s in SEEDS], summary["final_mean_return"])
    ax.axhline(mean_final, ls="--", lw=1.6, label=f"3-seed mean = {mean_final:,.1f}")
    ax.set_xlabel("Training seed")
    ax.set_ylabel("Final mean return over 10 evaluation episodes")
    ax.set_title("SAC baseline: final policy performance")
    ax.set_ylim(bottom=0)
    ax.legend()
    save_figure(fig, "03_final_return_by_seed")

    # SPS is cumulative steps/elapsed time, not an instantaneous speed measurement.
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for seed in SEEDS:
        ax.plot(training[seed]["step"], training[seed]["sps"], lw=1.5, label=f"Seed {seed}")
    finish_axes(ax, "Cumulative environment steps per second")
    ax.set_title("SAC baseline: throughput (cumulative SPS)")
    ax.legend()
    save_figure(fig, "04_cumulative_sps")

    # Diagnostics are smoothed for readability; no values are changed on disk.
    # The window is 25 logged points (~25k env steps with the frozen log_interval=1k).
    diagnostics = [
        ("critic_loss", "Critic loss", "05_critic_loss"),
        ("alpha", "Entropy coefficient α", "06_entropy_alpha"),
        ("q_mean_abs", "Mean |Q|", "07_mean_abs_q"),
        ("critic_grad_norm", "Critic gradient norm", "08_critic_grad_norm"),
    ]
    for metric, ylabel, filename in diagnostics:
        if not all(metric in training[seed].columns for seed in SEEDS):
            print(f"SKIP {filename}: '{metric}' not found for every seed")
            continue
        fig, ax = plt.subplots(figsize=(10, 5.8))
        for seed in SEEDS:
            df = training[seed]
            values = pd.to_numeric(df[metric], errors="coerce")
            smoothed = values.rolling(window=25, min_periods=1, center=True).mean()
            ax.plot(df["step"], smoothed, lw=1.55, label=f"Seed {seed}")
        finish_axes(ax, ylabel)
        ax.set_title(f"SAC baseline: {ylabel.lower()} (25-point moving average)")
        ax.legend()
        save_figure(fig, filename)

    report = "\n".join([
        "# ELE680 SAC baseline — descriptive results",
        "",
        "- Environment: HalfCheetah-v5; actor and both critics: 2×256 MLP",
        "- Three training seeds (0, 1, 2), each trained for 1,000,000 steps",
        "- Periodic: 5 evaluation episodes every 25,000 steps (40 checkpoints)",
        "- Final: 10 deterministic evaluation episodes per trained seed",
        f"- Source Git revision for all runs: `{EXPECTED_COMMIT}`",
        "",
        "## Per-run metrics",
        "",
        summary[["seed", "final_mean_return", "final_episode_sd", "mean_periodic_return", "runtime_hours"]].round(2).to_markdown(index=False),
        "",
        f"**Final evaluation across training seeds: {mean_final:,.2f} ± {sd_final:,.2f}**",
        "(mean ± sample SD of the *three per-seed final means*; n=3).",
        "The per-seed `final_episode_sd` is within-seed episode variation, not between-seed variation.",
        "",
        "## Interpretation boundaries",
        "",
        "- This baseline alone does not establish architecture scaling effects.",
        "- The shaded learning-curve band represents between-training-seed sample SD, not a confidence interval.",
        "- The final returns use 10 episodes; the learning curves use periodic 5-episode evaluations.",
        "- SPS is a cumulative throughput measure; the seed-2 slowdown cause remains unknown.",
        "- The last periodic evaluation and the final evaluation reuse some deterministic episode seeds; do not treat those as independent samples.",
        "- Diagnostics are moving averages for display; raw CSV files remain unchanged.",
        "",
        "## Reproduce figures",
        "",
        "```powershell",
        ".\\.venv\\Scripts\\python.exe .\\analysis\\plot_baseline.py",
        "```",
        "",
    ])
    (OUT / "baseline_summary.md").write_text(report, encoding="utf-8")
    print(f"PASS: verified comparable baseline source files for seeds {SEEDS}")
    print(f"Final mean ± sample SD across seeds: {mean_final:,.2f} ± {sd_final:,.2f}")
    print(f"Saved PNG/PDF figures and CSV/Markdown summaries to: {OUT}")
    print(f"Files produced: {len(list(OUT.iterdir()))}")


if __name__ == "__main__":
    main()
