#!/usr/bin/env python3
"""ELE680 24-run reproducibility helper. Never launches training or modifies raw runs.

Usage:
  python analysis/final_24run_reproduce.py verify-bundled
  python analysis/final_24run_reproduce.py verify-raw --results-root results
  python analysis/final_24run_reproduce.py plots --output-dir analysis/generated/final_24run
  python analysis/final_24run_reproduce.py plan
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "analysis" / "data" / "final_24run"
SEEDS = (0, 1, 2)
STEPS = tuple(range(25_000, 1_000_001, 25_000))
CONFIGS = {
    "2x256": "baseline.json",
    "2x512": "width_2x512.json",
    "2x1024": "width_2x1024.json",
    "4x256": "depth_4x256.json",
    "8x256": "depth_8x256.json",
    "InputNorm_8x256": "ablation_inputnorm_8x256.json",
    "Residual_8x256": "ablation_residual_8x256.json",
    "LayerNorm_8x256": "ablation_layernorm_8x256.json",
}
ORDER = tuple(CONFIGS)
RUN_FIELDS = ("architecture", "seed", "run_id", "git_commit", "final_mean_return",
              "final_within_seed_episode_sd", "runtime_hours", "documented_gpu_deviation")


def require(ok: bool, msg: str):
    if not ok:
        raise ValueError(msg)


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def approx(a, b, *, tolerance=1e-6):
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=tolerance)


def values_stats(values):
    return statistics.fmean(values), statistics.stdev(values)


def bundled():
    rows = read_csv(DATA / "run_inventory.csv")
    curves = read_csv(DATA / "learning_curves.csv")
    summary = read_csv(DATA / "final_summary.csv")
    require(len(rows) == 24 and len(curves) == 320 and len(summary) == 8,
            "Expected 24 runs, 320 periodic aggregate rows and 8 summaries")
    require({r["architecture"] for r in rows} == set(ORDER), "Unexpected conditions")
    require(len({r["run_id"] for r in rows}) == 24, "Duplicate run IDs")
    require({(r["architecture"], int(r["seed"])) for r in rows}
            == {(a, s) for a in ORDER for s in SEEDS}, "Missing training seeds")
    require({(r["architecture"], int(r["step"])) for r in curves}
            == {(a, s) for a in ORDER for s in STEPS}, "Missing evaluations")
    require({r["architecture"] for r in summary} == set(ORDER), "Summary conditions wrong")
    for row in curves:
        observations = [float(row[f"seed{s}"]) for s in SEEDS]
        avg, sd = values_stats(observations)
        require(approx(avg, row["periodic_mean_across_seeds"]) and
                approx(sd, row["periodic_sample_sd_ddof1"]),
                f"Periodic mean/sample SD mismatch: {row['architecture']} {row['step']}")
    for row in summary:
        key = row["architecture"]
        observed = [float(next(r["final_mean_return"] for r in rows
                               if r["architecture"] == key and int(r["seed"]) == s))
                    for s in SEEDS]
        avg, sd = values_stats(observed)
        require(approx(avg, row["final_mean_across_seeds"]) and
                approx(sd, row["final_sample_sd_ddof1"]), f"Final SD mismatch: {key}")
        require(int(row["n_seeds"]) == 3, "Bad seed count")
        for s in SEEDS:
            require(approx(observed[s], row[f"seed{s}_final"]), "Wrong final seed value")
    flags = {(r["architecture"], int(r["seed"])) for r in rows
             if r["documented_gpu_deviation"].strip().lower() == "true"}
    require(flags == {("LayerNorm_8x256", 1), ("LayerNorm_8x256", 2)},
            "GPU exceptions missing or incorrectly assigned")
    print("PASS: all 24 runs and 320 periodic points have consistent three-seed means/SD")


def verify_raw(path: Path):
    manifest = read_csv(DATA / "run_inventory.csv")
    require(path.is_dir(), f"Missing original results directory: {path}")
    for row in manifest:
        label = row["run_id"]
        folder = path / label
        require(folder.is_dir(), f"Missing original run directory: {folder}")
        for name in ("config.json", "metadata.json", "status.json", "train_metrics.csv",
                     "episode_metrics.csv", "eval_metrics.csv", "checkpoint.pt"):
            require((folder / name).is_file(), f"{label}: missing {name}")
        meta = read_json(folder / "metadata.json")
        config = read_json(folder / "config.json")
        status = read_json(folder / "status.json")
        require(meta["status"] == status["status"] == "completed", f"{label}: not completed")
        require(meta["steps_completed"] == status["steps_completed"] == 1_000_000,
                f"{label}: incomplete environment steps")
        require(meta["git_commit"] == row["git_commit"], f"{label}: SHA mismatch")
        require(meta.get("git_dirty") is False, f"{label}: code was dirty")
        require(meta["config"] == config, f"{label}: resolved config inconsistent")
        require(config["seed"] == int(row["seed"]), f"{label}: wrong training seed")
        require(config["env_id"] == "HalfCheetah-v5", f"{label}: wrong environment")
        require(config["total_steps"] == 1_000_000, f"{label}: wrong budget")
        ev = read_csv(folder / "eval_metrics.csv")
        require(len(ev) == 41, f"{label}: expected 41 evaluations")
        period = [v for v in ev if v["kind"] == "periodic"]
        final = [v for v in ev if v["kind"] == "final"]
        require(len(period) == 40 and len(final) == 1,
                f"{label}: periodic/final count incorrect")
        require([int(v["step"]) for v in period] == list(STEPS),
                f"{label}: evaluation steps incorrect")
        require(all(int(v["num_episodes"]) == 5 for v in period),
                f"{label}: periodic evaluation episode count incorrect")
        require(int(final[0]["step"]) == 1_000_000 and
                int(final[0]["num_episodes"]) == 10,
                f"{label}: final evaluation protocol incorrect")
        for v in ev:
            returns = json.loads(v["returns"])
            require(len(returns) == int(v["num_episodes"]) and
                    all(math.isfinite(float(x)) for x in returns),
                    f"{label}: evaluation returns invalid")
            require(approx(statistics.fmean(returns), v["mean_return"]), 
                    f"{label}: wrong episode mean")
            require(approx(statistics.pstdev(returns), v["std_return"]),
                    f"{label}: wrong within-evaluation episode population SD")
        require(approx(final[0]["mean_return"], row["final_mean_return"]),
                f"{label}: final score differs from published manifest")
        require(len(read_csv(folder / "episode_metrics.csv")) == 1000,
                f"{label}: expected 1000 train episodes")
        require(len(read_csv(folder / "train_metrics.csv")) == 994,
                f"{label}: expected 994 train-metric rows")
        print("PASS:", label)
    print("PASS: all 24 archived runs match manifest and raw-data requirements")
    print("Note: this convenience checker does not replace each frozen-source auditor.")


def plots(out: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("Matplotlib missing; CSV verification still works without it.")
    series = defaultdict(list)
    for r in read_csv(DATA / "learning_curves.csv"):
        series[r["architecture"]].append(r)
    out.mkdir(parents=True, exist_ok=True)
    selections = {
        "width": ("2x256", "2x512", "2x1024"),
        "depth": ("2x256", "4x256", "8x256"),
        "deep_component_ablation": ("8x256", "InputNorm_8x256",
                                    "Residual_8x256", "LayerNorm_8x256"),
    }
    for chart_name, archs in selections.items():
        fig, ax = plt.subplots(figsize=(9, 5))
        for arch in archs:
            vals = sorted(series[arch], key=lambda r: int(r["step"]))
            x = [int(r["step"]) for r in vals]
            y = [float(r["periodic_mean_across_seeds"]) for r in vals]
            sd = [float(r["periodic_sample_sd_ddof1"]) for r in vals]
            line, = ax.plot(x, y, label=arch, linewidth=1.8)
            ax.fill_between(x, [a - b for a, b in zip(y, sd)],
                            [a + b for a, b in zip(y, sd)],
                            color=line.get_color(), alpha=0.12)
        ax.set(xlabel="Environment steps", ylabel="5-episode periodic return",
               title=chart_name.replace("_", " ").title())
        ax.legend(fontsize=8)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(out / f"{chart_name}_mean_sd.png", dpi=180)
        fig.savefig(out / f"{chart_name}_mean_sd.pdf")
        plt.close(fig)
        print("WROTE:", out / f"{chart_name}_mean_sd.png")
    print("Each band is sample SD across independent training seeds (n=3, ddof=1).")


def plan():
    """Print only: never run jobs or change the checkout."""
    manifest = read_csv(DATA / "run_inventory.csv")
    print("# INFORMATIVE DRY RUN ONLY. Requires explicit GPU access and new run IDs.")
    print("# Save this plan outside the checkout before switching historical revisions.")
    print("# Use a separate clean training clone; keep its environment in .venv/.")
    for row in manifest:
        arch, seed, sha = row["architecture"], row["seed"], row["git_commit"]
        cfg = CONFIGS[arch]
        name = f"REPRO_{arch}_seed{seed}".replace("x", "x")
        print(f"git switch --detach {sha}")
        print(f"CUDA_VISIBLE_DEVICES=0 python -m src.train --config configs/{cfg}"
              f" --seed {seed} --device cuda --run-id {name} --output-dir results/reproduced")
    print("# Run serially only with your assigned GPU and prior permission.")
    print("# For historical provenance, use EXACT recorded SHA, not just current main.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("verify-bundled", "verify-raw", "plots", "plan"))
    parser.add_argument("--results-root", type=Path, default=ROOT / "results")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "analysis" / "generated" / "final_24run")
    args = parser.parse_args()
    bundled()
    if args.command == "verify-raw":
        verify_raw(args.results_root)
    elif args.command == "plots":
        plots(args.output_dir)
    elif args.command == "plan":
        plan()


if __name__ == "__main__":
    main()
