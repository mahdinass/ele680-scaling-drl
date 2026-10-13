"""ELE680: reproducible 2x256 / 2x512 / 2x1024 SAC width comparison (HalfCheetah-v5).

Run from repo root: .venv\\Scripts\\python.exe analysis\\plot_width_comparison.py [OUT_DIR] [--no-figures]
Reads only the tracked analysis copies under analysis/data/. Fails explicitly on any
inconsistency (see width_comparison.py). Writes to analysis/figures/width_comparison/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import width_comparison as wc  # noqa: E402

OUT = wc.ANALYSIS_DIR / "figures" / "width_comparison"
CHECKPOINTS = (25_000, 100_000, 250_000, 500_000, 750_000, 1_000_000)
THRESHOLDS = (5_000, 8_000, 10_000)
SMOOTH = 25  # logged points (~25k env steps at log_interval=1000)
DROP_THRESHOLD = 1_000.0  # descriptive: a fall of >1,000 return between consecutive periodic evaluations
DIAG_WINDOW = 25_000  # training steps preceding an evaluation that are inspected for diagnostics
# (name, first step included, first step excluded): 10 + 19 + 11 = 40 periodic evaluations
PHASES = (("early_25k_250k", 0, 275_000), ("middle_275k_725k", 275_000, 750_000), ("late_750k_1M", 750_000, 10**9))
STEADY_AFTER = 250_000  # intervals after this step are used for the less phase-confounded rank correlation
SLOW_FACTOR = 1.5
SLOW_KEY = "runtime_flagged_slow_(>1.5x_median_of_nine)"
PHASE_LABELS = {"early_25k_250k": "25k–250k", "middle_275k_725k": "275k–725k", "late_750k_1M": "750k–1M"}
ARCHS = tuple(wc.SPECS)  # width order: 2x256, 2x512, 2x1024
BASE = ARCHS[0]
PAIRS = tuple((a, b) for i, a in enumerate(ARCHS) for b in ARCHS[i + 1:])  # (reference, wider)
COLORS = {"baseline_2x256": "#0072B2", "width_2x512": "#D55E00", "width_2x1024": "#009E73"}  # Okabe-Ito
MARKERS = {0: "o", 1: "s", 2: "^"}
SEED_LS = {0: "-", 1: "--", 2: ":"}
PNG_META = {"Software": None}
PDF_META = {"CreationDate": None, "Producer": None, "Creator": None}


# ------------------------------------------------------------------ small helpers

def label(runs, name):
    return runs[name][0].spec.label


def pair_key(runs, a, b):
    return f"{label(runs, b)} - {label(runs, a)}"


def save(fig, name, out):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png", dpi=200, bbox_inches="tight", metadata=PNG_META)
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", metadata=PDF_META)
    plt.close(fig)


def style(ax, ylabel, xlabel="Environment steps"):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.22)
    ax.ticklabel_format(axis="x", style="sci", scilimits=(6, 6))


def smoothed(run, column):
    v = pd.to_numeric(run.train[column], errors="coerce")
    return v.rolling(window=SMOOTH, min_periods=1, center=True).mean()


def dual_legend_handles(runs):
    """Colour = architecture, line style = seed: six compact entries instead of nine."""
    arch = [Line2D([0], [0], color=COLORS[n], lw=2.4, label=label(runs, n)) for n in runs]
    seed = [Line2D([0], [0], color="0.25", lw=1.6, ls=SEED_LS[s], label=f"seed {s}") for s in wc.SEEDS]
    return arch + seed


# ------------------------------------------------------------------ tables / statistics

def temporal_diagnostics(runs):
    """Evaluation drops and the optimisation diagnostics logged in the 25k steps before them.

    Descriptive only: n is tiny, intervals overlap in time, and association is not causation."""
    events, rows = {}, []
    for n, rs in runs.items():
        for r in rs:
            key = f"{label(runs, n)} seed {r.seed}"
            spikes = wc.critic_loss_spikes(r.train["critic_loss"], SMOOTH)
            spike_steps = [int(r.train["step"].iloc[i]) for i in spikes["event_positions"]]
            values = r.periodic["mean_return"].to_numpy(dtype=float)
            steps = r.periodic["step"].to_numpy()
            ratios = {c: np.array([wc.window_max_ratio(r.train, c, int(s), DIAG_WINDOW) for s in steps])
                      for c in ("critic_loss", "critic_grad_norm")}
            drops = []
            for d in wc.eval_drops(r.periodic, DROP_THRESHOLD):
                i = int(np.where(steps == d["step"])[0][0])
                drops.append({
                    "step": d["step"], "drop": d["drop"],
                    "return_before": float(values[i - 1]), "return_at_drop": float(values[i]),
                    "return_after": float(values[i + 1]) if i + 1 < len(values) else None,
                    "critic_loss_window_max_over_run_median": float(ratios["critic_loss"][i]),
                    "critic_grad_norm_window_max_over_run_median": float(ratios["critic_grad_norm"][i]),
                    "critic_loss_spike_event_in_window": any(d["step"] - DIAG_WINDOW < s <= d["step"] for s in spike_steps),
                })
            events[key] = {"n_drops": len(drops), "drops": drops, "spike_event_steps": spike_steps,
                           "critic_loss_window_ratio_median_all_evals": float(np.nanmedian(ratios["critic_loss"])),
                           "critic_grad_norm_window_ratio_median_all_evals": float(np.nanmedian(ratios["critic_grad_norm"]))}
            change = np.diff(values)
            for j in range(1, len(values)):
                rows.append({"run": key, "arch": label(runs, n), "step": int(steps[j]), "change": change[j - 1],
                             "loss_ratio": ratios["critic_loss"][j], "grad_ratio": ratios["critic_grad_norm"][j]})
    pooled = pd.DataFrame(rows)

    def spearman(df, col):
        return float(df["change"].rank().corr(df[col].rank()))
    def block(df):
        steady = df[df["step"] > STEADY_AFTER]
        return {"n_intervals": int(len(df)),
                "spearman_return_change_vs_critic_loss_window_ratio": spearman(df, "loss_ratio"),
                "spearman_return_change_vs_critic_grad_norm_window_ratio": spearman(df, "grad_ratio"),
                f"n_intervals_after_{STEADY_AFTER // 1000}k": int(len(steady)),
                f"spearman_after_{STEADY_AFTER // 1000}k_critic_loss": spearman(steady, "loss_ratio"),
                f"spearman_after_{STEADY_AFTER // 1000}k_critic_grad_norm": spearman(steady, "grad_ratio")}
    corr = {"pooled_nine_runs": block(pooled)}
    for lab, df in pooled.groupby("arch", sort=False):
        corr[lab] = block(df)
    return events, corr


def seed_ordering(rs):
    """How often each seed holds the highest / lowest periodic mean return among the three seeds."""
    m = np.column_stack([r.periodic["mean_return"].to_numpy(dtype=float) for r in rs])
    return {"highest_fraction": {str(r.seed): float((m.argmax(axis=1) == k).mean()) for k, r in enumerate(rs)},
            "lowest_fraction": {str(r.seed): float((m.argmin(axis=1) == k).mean()) for k, r in enumerate(rs)}}


def phase_stats(agg, rs):
    """Between-seed SD and max-min range of periodic returns, averaged over training phases."""
    m = np.column_stack([r.periodic["mean_return"].to_numpy(dtype=float) for r in rs])
    steps = agg["step"].to_numpy()
    out = {}
    for name, lo, hi in PHASES:
        sel = (steps >= lo) & (steps < hi)
        out[name] = {"n_evals": int(sel.sum()),
                     "mean_between_seed_sd": float(agg["sd"].to_numpy()[sel].mean()),
                     "mean_range_across_seeds": float((m.max(axis=1) - m.min(axis=1))[sel].mean())}
    return out


def build_tables(runs):
    wc.check_comparable(runs)
    wc.require_seed_set(runs)
    labels = {n: label(runs, n) for n in runs}
    agg = {n: wc.aggregate_periodic([r.periodic for r in rs]) for n, rs in runs.items()}

    out_agg = pd.DataFrame({"step": agg[BASE]["step"]})
    for n in runs:
        out_agg[f"{n}_mean"], out_agg[f"{n}_sd"] = agg[n]["mean"], agg[n]["sd"]
        for r in runs[n]:
            out_agg[f"{n}_seed{r.seed}"] = r.periodic["mean_return"].to_numpy()
    for a, b in PAIRS:  # descriptive gap of architecture means; SE = sqrt(sd_a^2/3 + sd_b^2/3)
        out_agg[f"gap_{b}_minus_{a}"] = agg[b]["mean"] - agg[a]["mean"]
        out_agg[f"gap_se_{b}_minus_{a}"] = np.sqrt(agg[a]["sd"] ** 2 / 3 + agg[b]["sd"] ** 2 / 3)

    rows = []
    for n, rs in runs.items():
        for r in rs:
            spikes = wc.critic_loss_spikes(r.train["critic_loss"], SMOOTH)
            late = r.periodic[r.periodic["step"] >= 750_000]["mean_return"]
            tail = r.periodic[r.periodic["step"] > 500_000]
            isps = wc.interval_sps(r.train)
            half = len(isps) // 2
            rows.append({
                "architecture": r.spec.label, "seed": r.seed, "run_id": r.config["run_id"],
                "git_commit": r.meta["git_commit"], "git_dirty": r.meta["git_dirty"],
                "hidden_width": r.hidden_width, "total_params": r.meta["parameter_counts"]["total"],
                "final_mean_return_10ep": r.final_mean_return,
                "final_within_seed_episode_sd": float(r.final["std_return"]),
                "periodic_mean_return_1M_5ep": float(r.periodic["mean_return"].iloc[-1]),
                "auc_mean_of_40_periodic": float(r.periodic["mean_return"].mean()),
                "steps_to_5000": wc.steps_to_threshold(r.periodic, 5000),
                "steps_to_8000": wc.steps_to_threshold(r.periodic, 8000),
                "steps_to_10000": wc.steps_to_threshold(r.periodic, 10000),
                "late_eval_sd_750k_1M": float(late.std(ddof=1)),
                "max_drawdown_periodic": wc.max_drawdown(r.periodic),
                "low_return_outlier_evals_total": wc.count_low_return_outlier_evals(r.periodic),
                "low_return_outlier_evals_after_500k": wc.count_low_return_outlier_evals(r.periodic, 500_000),
                "critic_loss_spikes_10x_median": spikes["n_spikes"],
                "critic_loss_spike_steps": ";".join(str(int(r.train["step"].iloc[i])) for i in spikes["event_positions"]),
                "critic_loss_max_ratio_to_median": spikes["max_ratio"],
                "critic_loss_max": float(r.train["critic_loss"].max()),
                "critic_loss_median": float(r.train["critic_loss"].median()),
                "runtime_hours": r.runtime_hours,
                "final_cumulative_sps": float(r.train["sps"].iloc[-1]),
                "runtime_seconds": float(r.meta["runtime_seconds"]),
                "plateau_mean_750k_1M": float(late.mean()),
                "slope_return_500k_1M_per_100k_steps": wc.linear_slope(tail["step"], tail["mean_return"]) * 100_000,
                "interval_sps_median": float(isps.median()),
                "interval_sps_median_first_half": float(isps.iloc[:half].median()),
                "interval_sps_median_second_half": float(isps.iloc[half:].median()),
                "n_eval_drops_gt_1000": len(wc.eval_drops(r.periodic, DROP_THRESHOLD)),
                "q_mean_abs_median": float(r.train["q_mean_abs"].median()),
                "critic_grad_norm_median": float(r.train["critic_grad_norm"].median()),
                "alpha_median": float(r.train["alpha"].median()),
            })
    per_run = pd.DataFrame(rows)

    def col(arch_label, c):
        return per_run[per_run["architecture"] == arch_label][c].to_numpy(dtype=float)

    stats = {
        "schema": {
            "version": 2,
            "changes_from_version_1": "extended from two to three architectures; per-architecture entries are now seed_summary "
                                      "dictionaries (n, mean, sd, min, max, range, cv, per_seed) instead of mean/sd/per_seed",
            "removed_keys": {"effect_width_vs_baseline": "replaced by '<metric>.pairwise_differences', keyed 'wider - narrower'; "
                                                         "the '2×512 - 2×256' entry holds the former values unchanged"},
            "renamed_keys": {"runtime_flagged_slow_(>1.5x_median_of_six)": SLOW_KEY},
        },
        "seeds": list(wc.SEEDS), "n_per_architecture": 3,
        "sd_definition": "sample SD, ddof=1, across training seeds (final_within_seed_episode_sd is the population SD, ddof=0, "
                         "across the 10 final-evaluation episodes of one run)",
        "architectures": {labels[n]: {"name": n, "hidden_width": runs[n][0].hidden_width,
                                      "total_params": runs[n][0].spec.total_params,
                                      "param_ratio_vs_2x256": runs[n][0].spec.total_params / runs[BASE][0].spec.total_params,
                                      "git_commit": runs[n][0].spec.git_commit,
                                      "run_ids": [r.config["run_id"] for r in runs[n]]} for n in runs},
        "pairs": [pair_key(runs, a, b) for a, b in PAIRS],
        "pairwise_note": "difference = wider minus narrower architecture; unpaired (the same numeric seed in two architectures "
                         "is not a matched sample); exact permutation p has minimum 0.10 for 3 vs 3 and is exploratory only",
    }
    label_pairs = tuple((labels[a], labels[b]) for a, b in PAIRS)
    metrics = ("final_mean_return_10ep", "final_within_seed_episode_sd", "periodic_mean_return_1M_5ep",
               "auc_mean_of_40_periodic", "late_eval_sd_750k_1M", "max_drawdown_periodic", "plateau_mean_750k_1M",
               "slope_return_500k_1M_per_100k_steps", "low_return_outlier_evals_after_500k", "runtime_hours",
               "final_cumulative_sps", "interval_sps_median", "critic_loss_spikes_10x_median", "critic_loss_max",
               "critic_loss_median", "q_mean_abs_median", "critic_grad_norm_median", "alpha_median")
    for metric in metrics:
        by_label = {labels[n]: col(labels[n], metric) for n in runs}
        stats[metric] = {lab: wc.seed_summary(v) for lab, v in by_label.items()}
        stats[metric]["pairwise_differences"] = wc.pairwise_effects(by_label, label_pairs)
    stats["steps_to_threshold"] = {str(t): {labels[n]: col(labels[n], f"steps_to_{t}").tolist() for n in runs}
                                   for t in THRESHOLDS}
    runtimes = {f"{r.spec.label} seed {r.seed}": r.runtime_hours for rs in runs.values() for r in rs}
    flagged = wc.flag_slow_runs(runtimes, SLOW_FACTOR)
    stats["runtime_hours_by_run"] = runtimes
    stats[SLOW_KEY] = flagged
    stats["runtime_median_excluding_flagged"] = {
        lab: float(np.median([v for k, v in runtimes.items() if k.startswith(lab) and k not in flagged]))
        for lab in labels.values()}
    stats["runtime_median_all_runs"] = float(np.median(list(runtimes.values())))
    stats["throughput_drop_onset"] = {f"{labels[n]} seed {r.seed}": wc.throughput_drop_onset(r.train)
                                      for n, rs in runs.items() for r in rs}
    stats["between_seed_sd_of_periodic_return"] = {
        labels[n]: {"mean_over_all_40_evals": float(agg[n]["sd"].mean()),
                    "at_1M": float(agg[n]["sd"].iloc[-1]),
                    "by_phase": phase_stats(agg[n], runs[n]),
                    "seed_ordering": seed_ordering(runs[n])} for n in runs}
    events, corr = temporal_diagnostics(runs)
    stats["eval_drops"] = {"definition": f"fall of more than {DROP_THRESHOLD:,.0f} in mean periodic return versus the "
                                         f"previous evaluation; diagnostics are the max over the preceding "
                                         f"{DIAG_WINDOW:,} training steps divided by that run's median",
                           "events": events, "spearman": corr,
                           "spearman_note": "Exploratory and descriptive only: runs and consecutive evaluation intervals are pooled although "
                                            "they are temporally dependent (autocorrelated, overlapping training windows), so the effective sample "
                                            "size is far below the number of intervals and no p-value is attached; the early-training phase confounds "
                                            "the all-interval coefficients (see the after-250k columns). A rank correlation is not evidence of causation."}

    ck_rows = []
    for cp in CHECKPOINTS:
        i = int(np.where(agg[BASE]["step"].to_numpy() == cp)[0][0])
        for n, rs in runs.items():
            per_seed = [float(r.periodic["mean_return"].iloc[i]) for r in rs]
            row = {"step": cp, "architecture": labels[n],
                   "eval_return_mean": agg[n]["mean"][i], "eval_return_sd_between_seeds": agg[n]["sd"][i]}
            for c in ("critic_loss", "alpha", "q_mean_abs", "critic_grad_norm"):
                vals = [wc.window_mean(r.train, c, cp) for r in rs]
                row[f"{c}_window_mean"], row[f"{c}_window_sd_between_seeds"] = wc.mean_sd(vals)
            row["eval_return_min_over_seeds"], row["eval_return_max_over_seeds"] = min(per_seed), max(per_seed)
            ck_rows.append(row)
    ck = pd.DataFrame(ck_rows)
    return {"agg": agg, "periodic_aggregate": out_agg, "per_run": per_run, "stats": stats, "checkpoints": ck,
            "flagged": flagged, "runtimes": runtimes}


# ------------------------------------------------------------------ figures

def fig_learning_curves(runs, t, out):
    agg = t["agg"]
    fig = plt.figure(figsize=(14, 8.6))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.45, 1], hspace=0.28, wspace=0.18)
    ax = fig.add_subplot(gs[:, 0])
    for n in runs:
        ax.plot(agg[n]["step"], agg[n]["mean"], color=COLORS[n], lw=2.3, label=f"{label(runs, n)} mean (n=3)")
        ax.fill_between(agg[n]["step"], agg[n]["mean"] - agg[n]["sd"], agg[n]["mean"] + agg[n]["sd"],
                        color=COLORS[n], alpha=0.18, lw=0)
    ax.set_title("Architecture mean ± 1 between-seed sample SD")
    style(ax, "Mean return over 5 evaluation episodes")
    ax.legend(loc="lower right")
    prev = None
    for k, n in enumerate(runs):
        a2 = fig.add_subplot(gs[k, 1], sharex=prev, sharey=prev)
        prev = prev or a2
        for r in runs[n]:
            a2.plot(r.periodic["step"], r.periodic["mean_return"], color=COLORS[n], lw=1.3, ls=SEED_LS[r.seed],
                    label=f"seed {r.seed}")
        a2.set_title(f"{label(runs, n)}: individual seeds", fontsize=10.5)
        style(a2, "Mean return (5 ep.)")
        a2.legend(fontsize=8.5, loc="lower right", ncol=3)
        if k < 2:
            a2.set_xlabel("")
    fig.suptitle("SAC on HalfCheetah-v5: deterministic-evaluation learning curves (periodic, 5 episodes)", y=0.94)
    save(fig, "01_learning_curves_comparison", out)


def fig_final_returns(runs, t, out):
    per_run = t["per_run"]
    fig, ax = plt.subplots(figsize=(8.2, 5.8))
    for k, n in enumerate(runs):
        lab = label(runs, n)
        vals = per_run[per_run["architecture"] == lab]["final_mean_return_10ep"].to_numpy()
        m, s = wc.mean_sd(vals)
        for r, v in zip(runs[n], vals):
            ax.scatter(k + (r.seed - 1) * 0.1, v, color=COLORS[n], marker=MARKERS[r.seed], s=75, zorder=3,
                       edgecolor="white", linewidth=0.6)
        ax.errorbar(k + 0.32, m, yerr=s, color="black", marker="D", capsize=5, lw=1.6, zorder=4)
        ax.annotate(f"{m:,.0f}\n± {s:,.0f}", (k + 0.32, m), textcoords="offset points", xytext=(9, 0),
                    ha="left", va="center", fontsize=9)
    ax.set_xticks(range(len(runs)), [label(runs, n) for n in runs])
    ax.set_xlim(-0.5, len(runs) - 0.1)
    ax.set_ylabel("Final return (10 deterministic episodes)")
    ax.set_title("Final performance of all nine runs\n(black = mean ± between-seed sample SD, n=3)")
    ax.grid(alpha=0.22, axis="y")
    handles = [Line2D([0], [0], marker=MARKERS[s], color="0.35", ls="", markersize=8, label=f"seed {s}") for s in wc.SEEDS]
    handles.append(Line2D([0], [0], marker="D", color="black", lw=1.6, label="mean ± sample SD"))
    ax.legend(handles=handles, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=4, frameon=False)
    save(fig, "02_final_returns", out)


def fig_runtime_throughput(runs, t, out):
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6))
    for n, rs in runs.items():
        for r in rs:
            axes[0].plot(r.train["step"], r.train["sps"], color=COLORS[n], lw=1.4, ls=SEED_LS[r.seed])
            isps = wc.interval_sps(r.train).rolling(SMOOTH, min_periods=1, center=True).mean()
            axes[1].plot(r.train["step"].iloc[1:], isps, color=COLORS[n], lw=1.2, ls=SEED_LS[r.seed], alpha=0.9)
    style(axes[0], "Cumulative environment steps per second")
    axes[0].set_title("Logged cumulative throughput")
    axes[0].legend(handles=dual_legend_handles(runs), ncol=2, fontsize=8.5, loc="lower right")
    style(axes[1], "Steps per second within log interval")
    axes[1].set_title(f"Interval throughput ({SMOOTH}-pt moving avg.)")
    runtimes, flagged = t["runtimes"], set(t["flagged"])
    pos, tick_labels, k = [], [], 0
    for n in runs:
        for r in runs[n]:
            key = f"{label(runs, n)} seed {r.seed}"
            bar = axes[2].bar(k, runtimes[key], color=COLORS[n], width=0.8)
            if key in flagged:
                bar[0].set_hatch("//")
                bar[0].set_edgecolor("black")
            axes[2].text(k, runtimes[key] + 0.15, f"{runtimes[key]:.2f}", ha="center", fontsize=8.5)
            pos.append(k)
            tick_labels.append(f"s{r.seed}")
            k += 1
        k += 0.6
    axes[2].set_xticks(pos, tick_labels)
    for j, n in enumerate(runs):
        axes[2].text(j * 3.6 + 1, -0.1, label(runs, n), transform=axes[2].get_xaxis_transform(), ha="center",
                     va="top", fontsize=10)
    axes[2].set_ylabel("Wall-clock runtime (hours)")
    axes[2].set_title(f"Runtime per run (hatched: >{SLOW_FACTOR}× median)")
    axes[2].grid(alpha=0.22, axis="y")
    axes[2].set_ylim(0, max(runtimes.values()) * 1.12)
    axes[2].legend(handles=[Patch(facecolor="white", edgecolor="black", hatch="//", label="flagged slow run")],
                   loc="upper right", fontsize=9)
    fig.suptitle("Observed (not controlled) throughput on a shared server: GPU load was not recorded, so differences "
                 "must not be attributed to width alone", y=1.0, fontsize=11.5)
    save(fig, "03_throughput_and_runtime", out)


def fig_critic_loss(runs, t, out):
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6), sharey=True)
    for ax, (n, rs) in zip(axes, runs.items()):
        for r in rs:
            raw = pd.to_numeric(r.train["critic_loss"], errors="coerce")
            ax.plot(r.train["step"], raw, color=COLORS[n], lw=0.5, alpha=0.28)
            ax.plot(r.train["step"], smoothed(r, "critic_loss"), color=COLORS[n], lw=1.8, ls=SEED_LS[r.seed],
                    label=f"seed {r.seed}")
            spikes = wc.critic_loss_spikes(raw, SMOOTH)
            if spikes["event_positions"]:
                idx = spikes["event_positions"]
                ax.scatter(r.train["step"].iloc[idx], raw.iloc[idx], marker="v", s=55, color="black", zorder=5,
                           label="spike event (>10× rolling median)" if r.seed == 2 and n == ARCHS[-1] else None)
        ax.set_yscale("log")
        style(ax, "Critic loss (log scale)")
        ax.set_title(label(runs, n))
        ax.legend(fontsize=8.5, loc="lower right")
    fig.suptitle(f"Critic loss per seed: thin = logged values, thick = {SMOOTH}-point moving average, "
                 "triangles = descriptive spike events", y=1.0, fontsize=11.5)
    save(fig, "04_critic_loss", out)


def fig_alpha(runs, t, out):
    fig, ax = plt.subplots(figsize=(10, 5.8))
    for n, rs in runs.items():
        for r in rs:
            ax.plot(r.train["step"], smoothed(r, "alpha"), color=COLORS[n], lw=1.5, ls=SEED_LS[r.seed], alpha=0.9)
    style(ax, "Entropy coefficient α")
    ax.set_title(f"Entropy coefficient α ({SMOOTH}-point moving average; colour = architecture, line style = seed)")
    ax.legend(handles=dual_legend_handles(runs), ncol=2, fontsize=9)
    save(fig, "05_entropy_alpha", out)


def fig_q_and_grad(runs, t, out):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8))
    for ax, (c, yl, logy) in zip(axes, (("q_mean_abs", "Mean |Q|", False), ("critic_grad_norm", "Critic gradient norm", True))):
        for n, rs in runs.items():
            for r in rs:
                ax.plot(r.train["step"], smoothed(r, c), color=COLORS[n], lw=1.4, ls=SEED_LS[r.seed], alpha=0.9)
        if logy:
            ax.set_yscale("log")
        style(ax, yl + (" (log scale)" if logy else ""))
        ax.set_title(f"{yl} ({SMOOTH}-point moving average)")
    axes[0].legend(handles=dual_legend_handles(runs), ncol=2, fontsize=9, loc="lower right")
    save(fig, "06_mean_abs_q_and_critic_grad_norm", out)


def fig_gaps(runs, t, out):
    agg = t["agg"]
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2), sharey=True)
    for ax, (a, b) in zip(axes, PAIRS):
        diff = agg[b]["mean"] - agg[a]["mean"]
        se = np.sqrt(agg[a]["sd"] ** 2 / 3 + agg[b]["sd"] ** 2 / 3)
        ax.plot(agg[a]["step"], diff, color=COLORS[b], lw=2.2)
        ax.fill_between(agg[a]["step"], diff - se, diff + se, color=COLORS[b], alpha=0.22, lw=0)
        ax.axhline(0, color="black", lw=0.9)
        style(ax, f"Mean return difference")
        ax.set_title(f"{label(runs, b)} minus {label(runs, a)}")
    fig.suptitle("Differences of architecture-mean periodic return. Band = ±1 SE of the difference, sqrt(SD_a²/3 + SD_b²/3): "
                 "descriptive, NOT a confidence interval (n = 3 vs 3)", y=1.0, fontsize=11)
    save(fig, "07_return_difference_over_training", out)


def fig_variability_scaling(runs, t, out):
    agg, per_run, runtimes, flagged = t["agg"], t["per_run"], t["runtimes"], set(t["flagged"])
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.8))
    for n in runs:
        axes[0].plot(agg[n]["step"], agg[n]["sd"], color=COLORS[n], lw=2, label=label(runs, n))
    style(axes[0], "Between-seed sample SD of mean return")
    axes[0].set_title("Seed-to-seed variability over training (n = 3 per point)")
    axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=3, frameon=False)
    params = {n: runs[n][0].spec.total_params for n in runs}
    means = []
    for n in runs:
        lab = label(runs, n)
        vals = per_run[per_run["architecture"] == lab]["final_mean_return_10ep"].to_numpy()
        means.append(vals.mean())
        for r, v in zip(runs[n], vals):
            axes[1].scatter(params[n] * (1 + (r.seed - 1) * 0.05), v, color=COLORS[n], marker=MARKERS[r.seed], s=70,
                            edgecolor="white", linewidth=0.6, zorder=3)
        m, s = wc.mean_sd(vals)
        axes[1].errorbar(params[n] * 1.22, m, yerr=s, color="black", marker="D", capsize=4, lw=1.4, zorder=4)
    axes[1].plot([params[n] * 1.22 for n in runs], means, color="black", lw=0.8, alpha=0.5, zorder=2)
    axes[1].set_xscale("log")
    axes[1].set_xticks([params[n] for n in runs], [f"{label(runs, n)}\n{params[n] / 1e6:.2f}M" for n in runs])
    axes[1].minorticks_off()
    axes[1].set_ylabel("Final return (10 episodes)")
    axes[1].set_xlabel("Trainable parameters (log scale)")
    axes[1].set_title("Final return vs. parameter count\n(black = mean ± sample SD)")
    axes[1].grid(alpha=0.22)
    for n in runs:
        for r in runs[n]:
            key = f"{label(runs, n)} seed {r.seed}"
            v = per_run[(per_run["architecture"] == label(runs, n)) & (per_run["seed"] == r.seed)]["final_mean_return_10ep"].iloc[0]
            axes[2].scatter(runtimes[key], v, color=COLORS[n], marker=MARKERS[r.seed], s=75, edgecolor="white",
                            linewidth=0.6, zorder=3)
            if key in flagged:
                axes[2].scatter(runtimes[key], v, s=260, facecolor="none", edgecolor="black", linewidth=1.3, zorder=4)
                axes[2].annotate("flagged slow run", (runtimes[key], v), textcoords="offset points", xytext=(-8, 12),
                                 ha="right", fontsize=9)
    axes[2].set_xlabel("Observed wall-clock runtime (hours; shared GPU, not controlled)")
    axes[2].set_ylabel("Final return (10 episodes)")
    axes[2].set_title("Final return vs. observed runtime")
    axes[2].grid(alpha=0.22)
    handles = [Patch(facecolor=COLORS[n], label=label(runs, n)) for n in runs] + [
        Line2D([0], [0], marker=MARKERS[s], color="0.35", ls="", markersize=8, label=f"seed {s}") for s in wc.SEEDS]
    axes[2].legend(handles=handles, ncol=2, fontsize=8.5, loc="center right")
    save(fig, "08_variability_and_scaling", out)


def make_figures(runs, t, out):
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                         "svg.hashsalt": "ele680", "pdf.fonttype": 42})
    fig_learning_curves(runs, t, out)
    fig_final_returns(runs, t, out)
    fig_runtime_throughput(runs, t, out)
    fig_critic_loss(runs, t, out)
    fig_alpha(runs, t, out)
    fig_q_and_grad(runs, t, out)
    fig_gaps(runs, t, out)
    fig_variability_scaling(runs, t, out)


# ------------------------------------------------------------------ report

def write_summary(runs, t, out):
    st, per_run, ck, agg = t["stats"], t["per_run"], t["checkpoints"], t["agg"]
    L = {n: label(runs, n) for n in runs}
    l1, l2, l3 = (L[n] for n in ARCHS)
    n1, n2, n3 = ARCHS
    fin, auc = st["final_mean_return_10ep"], st["auc_mean_of_40_periodic"]
    pk = st["pairs"]
    f0 = lambda x: f"{x:,.0f}"  # noqa: E731
    f1 = lambda x: f"{x:,.1f}"  # noqa: E731
    f2 = lambda x: f"{x:,.2f}"  # noqa: E731
    pm = lambda s, f=f1: f"{f(s['mean'])} ± {f(s['sd'])}"  # noqa: E731
    drops = st["eval_drops"]["events"]
    spear = st["eval_drops"]["spearman"]["pooled_nine_runs"]
    sd_phase = st["between_seed_sd_of_periodic_return"]
    ordering = sd_phase[l3]["seed_ordering"]
    hi_s = max(ordering["highest_fraction"], key=ordering["highest_fraction"].get)
    lo_s = max(ordering["lowest_fraction"], key=ordering["lowest_fraction"].get)
    hi_seed, lo_seed = int(hi_s), int(lo_s)
    r3 = {r.seed: r for r in runs[n3]}
    rs3 = per_run[per_run["architecture"] == l3].set_index("seed")
    gaps = {(a, b): agg[b]["mean"].to_numpy() - agg[a]["mean"].to_numpy() for a, b in PAIRS}
    steps = agg[n1]["step"].to_numpy()
    runtimes = t["runtimes"]
    flagged = t["flagged"]
    slow_key = flagged[0] if flagged else None
    slow = st["throughput_drop_onset"].get(slow_key) if slow_key else None
    others_steady = all(v["onset_step"] is None for k, v in st["throughput_drop_onset"].items() if k != slow_key)
    med_ex = st["runtime_median_excluding_flagged"]

    arch_tbl = pd.DataFrame([{
        "Architecture": L[n], "Hidden width": runs[n][0].hidden_width, "Trainable parameters": f"{runs[n][0].spec.total_params:,}",
        "× 2×256": f"{st['architectures'][L[n]]['param_ratio_vs_2x256']:.2f}",
        "Scientific commit": runs[n][0].spec.git_commit[:7],
        "Run IDs (seed 0, 1, 2)": "<br>".join(r.config["run_id"] for r in runs[n])} for n in runs])
    val = per_run[["architecture", "seed", "git_commit", "git_dirty", "hidden_width", "total_params",
                   "final_mean_return_10ep", "final_within_seed_episode_sd", "runtime_hours"]].copy()
    val["git_commit"] = val["git_commit"].str[:7]
    final_rows = []
    for n in runs:
        s = fin[L[n]]
        final_rows.append({"Architecture": L[n], "Mean ± sample SD (n=3)": pm(s, f2), "Min – max": f"{f1(s['min'])} – {f1(s['max'])}",
                           "Range": f1(s["range"]), "CV": f"{s['cv']:.3f}",
                           "Per seed (0, 1, 2)": ", ".join(f1(v) for v in s["per_seed"]),
                           "Within-run episode SD (10 ep.)": ", ".join(f0(v) for v in st["final_within_seed_episode_sd"][L[n]]["per_seed"])})
    final_tbl = pd.DataFrame(final_rows)

    def pair_tbl(metric, fmt=f1):
        rows = []
        for k in pk:
            e = st[metric]["pairwise_differences"][k]
            pct = "n/a" if e["percent_difference"] is None else f"{e['percent_difference']:+.2f}%"
            rows.append({"Comparison (wider − narrower)": k, "Mean difference": f"{e['mean_difference']:+,.{2 if fmt is f2 else 1}f}",
                         "Relative to narrower": pct, "Pooled SD": fmt(e["pooled_sd"]),
                         "Hedges' g": "n/a" if e["hedges_g"] is None else f"{e['hedges_g']:+.2f}",
                         "Welch SE of diff.": fmt(e["welch_se_of_difference"]),
                         "Exact permutation p": f"{e['permutation_p_two_sided']:.2f}"})
        return pd.DataFrame(rows)

    cp = ck.copy()
    cp["Mean (min – max over seeds)"] = [f"{f0(a)} ({f0(lo)} – {f0(hi)})" for a, lo, hi in
                                         zip(cp["eval_return_mean"], cp["eval_return_min_over_seeds"], cp["eval_return_max_over_seeds"])]
    cp["Between-seed SD"] = cp["eval_return_sd_between_seeds"].map(f0)
    cp_tbl = cp[["step", "architecture", "Mean (min – max over seeds)", "Between-seed SD"]].rename(
        columns={"step": "Step", "architecture": "Architecture"})
    cp_tbl["Step"] = cp_tbl["Step"].map(lambda v: f"{v // 1000}k")
    auc_tbl = pd.DataFrame([{"Architecture": L[n], "Mean of 40 periodic means ± SD": pm(auc[L[n]]),
                             "Per seed": ", ".join(f0(v) for v in auc[L[n]]["per_seed"])} for n in runs])
    thr = st["steps_to_threshold"]
    thr_tbl = pd.DataFrame([{"Threshold": k, **{L[n]: ", ".join("never" if pd.isna(v) else f"{int(v) // 1000}k" for v in thr[k][L[n]])
                                                   for n in runs}} for k in thr])
    gap_rows = []
    for (a, b), g in gaps.items():
        gap_rows.append({"Comparison": f"{L[b]} − {L[a]}", "Evals with negative mean gap": f"{int((g < 0).sum())} of {len(g)}",
                         "Mean gap, steps ≤ 300k": f0(g[steps <= 300_000].mean()), "Mean gap, steps > 300k": f0(g[steps > 300_000].mean()),
                         "Mean gap, steps ≥ 750k": f0(g[steps >= 750_000].mean())})
    gap_tbl = pd.DataFrame(gap_rows)

    var_rows = []
    for n in runs:
        b = sd_phase[L[n]]
        row = {"Architecture": L[n]}
        for ph, v in b["by_phase"].items():
            row[f"SD {PHASE_LABELS[ph]} (range)"] = f"{f0(v['mean_between_seed_sd'])} ({f0(v['mean_range_across_seeds'])})"
        row["SD at 1M"] = f0(b["at_1M"])
        var_rows.append(row)
    var_tbl = pd.DataFrame(var_rows)
    stab_rows = []
    for n in runs:
        sm = lambda m: ", ".join(f1(v) for v in st[m][L[n]]["per_seed"])  # noqa: E731
        stab_rows.append({"Architecture": L[n], "Plateau: mean of 750k–1M evals": sm("plateau_mean_750k_1M"),
                          "Slope 500k–1M (return per 100k steps)": sm("slope_return_500k_1M_per_100k_steps"),
                          "SD of 750k–1M evals": sm("late_eval_sd_750k_1M"), "Max drawdown": sm("max_drawdown_periodic"),
                          "Evals after 500k with low-return outlier episode¹": ", ".join(str(int(v)) for v in st["low_return_outlier_evals_after_500k"][L[n]]["per_seed"])})
    stab_tbl = pd.DataFrame(stab_rows)
    ev_rows = []
    for key, v in drops.items():
        for d in v["drops"]:
            ev_rows.append({"Run": key, "Evaluation step": f"{d['step'] // 1000}k", "Return before → at → after": f"{f0(d['return_before'])} → {f0(d['return_at_drop'])} → {'n/a' if d['return_after'] is None else f0(d['return_after'])}",
                            "Critic-loss window max / run median": f"{d['critic_loss_window_max_over_run_median']:.2f} (run's typical {v['critic_loss_window_ratio_median_all_evals']:.2f})",
                            "Critic-grad-norm window max / run median": f"{d['critic_grad_norm_window_max_over_run_median']:.2f} (typical {v['critic_grad_norm_window_ratio_median_all_evals']:.2f})",
                            "Spike event in window": "yes" if d["critic_loss_spike_event_in_window"] else "no"})
    ev_tbl = pd.DataFrame(ev_rows)
    spike_tbl = per_run[["architecture", "seed", "critic_loss_spikes_10x_median", "critic_loss_spike_steps",
                         "critic_loss_max", "critic_loss_median", "critic_loss_max_ratio_to_median"]].copy()
    spike_tbl = spike_tbl.rename(columns={"architecture": "Architecture", "seed": "Seed", "critic_loss_spikes_10x_median": "Spike events",
                                          "critic_loss_spike_steps": "Event steps", "critic_loss_max": "Max critic loss",
                                          "critic_loss_median": "Median critic loss", "critic_loss_max_ratio_to_median": "Max / rolling median"})
    diag_rows = []
    for n in runs:
        for m, name in (("critic_loss_median", "Median critic loss"), ("q_mean_abs_median", "Median mean-abs Q"),
                        ("critic_grad_norm_median", "Median critic grad norm"), ("alpha_median", "Median α")):
            diag_rows.append((L[n], name, ", ".join(f"{v:,.3f}" if m == "alpha_median" else f1(v) for v in st[m][L[n]]["per_seed"])))
    diag_tbl = pd.DataFrame(diag_rows, columns=["Architecture", "Statistic", "Per seed (0, 1, 2)"]).pivot(
        index="Architecture", columns="Statistic", values="Per seed (0, 1, 2)").reindex([L[n] for n in runs])[
        ["Median critic loss", "Median mean-abs Q", "Median critic grad norm", "Median α"]].reset_index()
    comp_rows = []
    for n in runs:
        c = st["architectures"][L[n]]
        comp_rows.append({"Architecture": L[n], "Parameters": f"{c['total_params']:,}", "× 2×256": f"{c['param_ratio_vs_2x256']:.2f}",
                          "Runtime h (seeds 0, 1, 2)": ", ".join(f2(v) for v in st["runtime_hours"][L[n]]["per_seed"]),
                          "Median runtime excl. flagged (h)": f2(med_ex[L[n]]),
                          "Final cumulative SPS": ", ".join(f1(v) for v in st["final_cumulative_sps"][L[n]]["per_seed"]),
                          "Median interval SPS": ", ".join(f1(v) for v in st["interval_sps_median"][L[n]]["per_seed"])})
    comp_tbl = pd.DataFrame(comp_rows)

    md = lambda df: df.to_markdown(index=False)  # noqa: E731
    seed_gap_1M = r3[hi_seed].final_mean_return - r3[int(lo_seed)].final_mean_return
    first_range = ck[(ck["step"] == 25_000) & (ck["architecture"] == l3)].iloc[0]
    ranges3 = {int(r["step"]): r["eval_return_max_over_seeds"] - r["eval_return_min_over_seeds"]
               for _, r in ck[ck["architecture"] == l3].iterrows()}
    sp3 = rs3.loc[int(hi_seed)]
    max_loss = float(sp3["critic_loss_max"])
    spike_step = [s for s in sp3["critic_loss_spike_steps"].split(";") if s]
    slow_txt = (f"In `{slow_key}` the interval throughput falls from a median of {slow['median_sps_before']:.1f} to {slow['median_sps_after']:.1f} steps/s "
                f"by step {slow['onset_step']:,} and does not recover (figure 03): an abrupt, persistent change that is not a width effect "
                f"({'the other eight runs show no such change' if others_steady else 'other runs also show throughput changes'}), but its cause was not recorded."
                if slow else "No run shows a persistent throughput drop by this criterion.")
    thr_never = [(L[n], i) for n in runs for i, v in enumerate(thr["10000"][L[n]]) if pd.isna(v)]
    all_drops = [(key, d, v) for key, v in drops.items() for d in v["drops"]]
    n_coincide = sum(d["critic_loss_spike_event_in_window"] for _, d, _ in all_drops)
    elevated = [key for key, d, v in all_drops
                if d["critic_loss_window_max_over_run_median"] > 2 * v["critic_loss_window_ratio_median_all_evals"]
                or d["critic_grad_norm_window_max_over_run_median"] > 2 * v["critic_grad_norm_window_ratio_median_all_evals"]]
    drop_txt = (f"{len(all_drops)} evaluation drops occur in total; {n_coincide} of them have a critic-loss spike event in the preceding window. "
                f"The window ratios (table) are close to each run's typical value for most drops; "
                f"{len(elevated)} drop(s) ({', '.join(sorted(set(elevated))) or 'none'}) have a critic-loss or gradient-norm window maximum more than twice the run's typical window. "
                f"With at most {max(v['n_drops'] for v in drops.values())} drops per run this is anecdotal: **the data do not establish a temporal link between evaluation drops and these diagnostics**, "
                f"and any such link would be an association, not a demonstrated cause.")

    text = f"""# ELE680 — SAC width comparison: 2×256 vs 2×512 vs 2×1024 (HalfCheetah-v5)

*Generated by `analysis/plot_width_comparison.py`; every number below is computed from the tracked run files
(nothing is hard-coded). **Descriptive analysis of n = 3 independent training seeds per architecture; no claim of statistical
significance, superiority or inferiority is made.***

## 1. Experimental setup and scientific provenance

SAC (frozen algorithm and protocol) on HalfCheetah-v5, 1,000,000 environment steps, seeds 0/1/2, periodic evaluation every 25k steps
(5 deterministic episodes) plus one separate final evaluation (10 episodes). Only the hidden width of the two-hidden-layer MLPs
(actor, Q1, Q2) changes. Training was run on the UiS Gorina4 server.

{md(arch_tbl)}

**The three architectures were trained at three different Git commits** (`{runs[n1][0].spec.git_commit}`, `{runs[n2][0].spec.git_commit}`,
`{runs[n3][0].spec.git_commit}`). The commits differ because the configurable-width code (`f25b3a0`) and then the 2×1024 configuration and
tests (`e533a3c`, which added only a config file and tests) were committed after the baseline. The default-width path is covered by
unit tests, but the earlier runs were not re-executed on the later commits. This is a documented confound, not an established equivalence.

## 2. Dataset validation (all nine runs passed)

Each run was checked, without repair or imputation, for: exactly one run directory; all six CSV/JSON files; HalfCheetah-v5; seed and
architecture (hidden width; the baseline omits `hidden_width`, read as the frozen default 256); exact scientific commit and clean tree;
parameter count; `completed` status at exactly 1,000,000 steps; `config.json` identical to the config in `metadata.json`; exactly 40 periodic
evaluations at 25k…1M (5 episodes) and one separate final evaluation at 1M (10 episodes); per-episode return arrays that reproduce the recorded
mean and population SD; metadata/CSV agreement for the final evaluation; exactly 994 training-metric rows at steps 6001, 7001, …, 999001 with finite diagnostics;
exactly 1,000 training episodes (1,000 steps each, steps 1000…1M, finite). Across all nine runs only `hidden_width`, `run_id` and `seed` differ in the configuration.
Final returns were also compared with the independent audit values in the unit tests (`RealSpecsAndNineRuns`).

{md(val.round(2))}

## 3. Final performance (separate 10-episode final evaluation)

{md(final_tbl)}

Between-seed SD (sample, ddof=1) describes spread across training seeds; the within-run episode SD (population, ddof=0, across 10 evaluation episodes of a single trained policy,
{f0(per_run['final_within_seed_episode_sd'].min())}–{f0(per_run['final_within_seed_episode_sd'].max())} across the nine runs) is a different quantity and is much smaller. Neither is a confidence interval.

Pairwise differences (unpaired; the same numeric seed in two architectures is **not** a matched sample):

{md(pair_tbl("final_mean_return_10ep"))}

- The mean final return is lower for each wider network ({f1(fin[l2]['mean'] - fin[l1]['mean'])} for 2×512, {f1(fin[l3]['mean'] - fin[l1]['mean'])} for 2×1024, relative to 2×256),
  but every difference is smaller than the between-seed SD of the 2×1024 group ({f0(fin[l3]['sd'])}), and the seed ranges overlap
  ({l1}: {f0(fin[l1]['min'])}–{f0(fin[l1]['max'])}; {l2}: {f0(fin[l2]['min'])}–{f0(fin[l2]['max'])}; {l3}: {f0(fin[l3]['min'])}–{f0(fin[l3]['max'])}).
- Exact permutation tests are exploratory. With 3 vs 3 seeds there are only 20 relabelings, the smallest attainable two-sided p is 0.10, and no pair can reach conventional
  significance whatever the data. A large p does **not** show equivalence either.
- The 2×1024 group is far more dispersed: coefficient of variation {fin[l3]['cv']:.3f} vs {fin[l1]['cv']:.3f} and {fin[l2]['cv']:.3f}.
- The periodic evaluation at 1M steps (5 episodes) uses the same deterministic episode seeds as the final evaluation and is not independent of it; the headline comparison uses the 10-episode final evaluation only.

## 4. Learning curves and sample efficiency

Mean periodic return (5 episodes) per checkpoint; spread = sample SD across the three seeds, with min–max over seeds:

{md(cp_tbl)}

Sample-efficiency proxy — the mean of the 40 periodic evaluation means (the established proxy; it is **not** a continuous-time integral):

{md(auc_tbl)}

{md(pair_tbl("auc_mean_of_40_periodic"))}

First evaluation step at which a seed's 5-episode mean reaches a threshold (seeds 0, 1, 2; "never" = not reached within 1M steps):

{md(thr_tbl)}

Architecture-mean gap over training (wider − narrower; descriptive, figure 07):

{md(gap_tbl)}

- 2×512 vs 2×256: the mean curve is lower at {int((gaps[(n1, n2)] < 0).sum())} of 40 evaluations, with the gap largest early and narrowing later (as in the earlier two-width analysis).
- 2×1024 lies below 2×256 at {int((gaps[(n1, n3)] < 0).sum())} of 40 and below 2×512 at {int((gaps[(n2, n3)] < 0).sum())} of 40 evaluations. The 2×1024 mean is pulled up by one seed (section 5);
  {'seed 0 of 2×1024 never reaches a mean return of 10,000' if thr_never == [(l3, 0)] else 'see the threshold table for runs that never reach 10,000'}.
- These are patterns in three seeds per group; the between-seed SD of 2×1024 averages {f0(sd_phase[l3]['mean_over_all_40_evals'])} over the 40 evaluations (2×256: {f0(sd_phase[l1]['mean_over_all_40_evals'])}; 2×512: {f0(sd_phase[l2]['mean_over_all_40_evals'])}), which is comparable to the size of the mean gaps in the table.

## 5. Variability and stability diagnostics

### 5.1 Is the 2×1024 seed-{hi_seed} advantage progressive or concentrated?

{md(var_tbl)}

(Entries: mean between-seed sample SD of the periodic mean return, and in brackets the mean max–min range across seeds, over the 10 / 19 / 11 evaluations of each phase.)

- In 2×1024, seed {hi_seed} has the highest 5-episode mean in {ordering['highest_fraction'][hi_s] * 100:.0f}% of the 40 evaluations and seed {lo_seed} the lowest in {ordering['lowest_fraction'][lo_s] * 100:.1f}%.
  The max–min range across seeds is already {f0(first_range['eval_return_max_over_seeds'] - first_range['eval_return_min_over_seeds'])} at 25k steps and {f0(ranges3[250_000])}, {f0(ranges3[500_000])}, {f0(ranges3[1_000_000])} at 250k, 500k, 1M.
  The between-seed SD is high from the first phase onward ({f0(sd_phase[l3]['by_phase']['early_25k_250k']['mean_between_seed_sd'])} early, {f0(sd_phase[l3]['by_phase']['late_750k_1M']['mean_between_seed_sd'])} late),
  versus {f0(sd_phase[l1]['by_phase']['early_25k_250k']['mean_between_seed_sd'])} / {f0(sd_phase[l1]['by_phase']['late_750k_1M']['mean_between_seed_sd'])} (2×256) and
  {f0(sd_phase[l2]['by_phase']['early_25k_250k']['mean_between_seed_sd'])} / {f0(sd_phase[l2]['by_phase']['late_750k_1M']['mean_between_seed_sd'])} (2×512).
  So the observed separation is **not concentrated in late training**: the seed ordering is established within the first few hundred thousand steps and then largely persists. Whether this reflects the initial network/replay randomness of each seed cannot be told from three seeds.
- At 1M steps the final-evaluation gap between the highest and lowest 2×1024 seed is {f0(seed_gap_1M)} return units, about {seed_gap_1M / fin[l3]['sd']:.1f} between-seed SDs of that group.

{md(stab_tbl)}

¹ Low-return outlier episode: an evaluation episode whose return is below {wc.LOW_RETURN_FRACTION:g} × the median episode return of the same 5-episode evaluation. Descriptive heuristic only.

- Different 2×1024 seeds sit at different levels in the last 250k steps (plateau means above). The 500k–1M least-squares slopes are positive for all nine runs
  (smallest {f0(min(min(st['slope_return_500k_1M_per_100k_steps'][L[n]]['per_seed']) for n in runs))} per 100k steps), so none of the runs has clearly stopped improving; a "plateau" here is a level reached at 1M steps, not a demonstrated asymptote.
- Largest drawdown below the running best: {', '.join(f"{L[n]} {f0(min(st['max_drawdown_periodic'][L[n]]['per_seed']))}–{f0(max(st['max_drawdown_periodic'][L[n]]['per_seed']))}" for n in runs)}.
  The late within-run SD (750k–1M) of 2×1024 ({', '.join(f0(v) for v in st['late_eval_sd_750k_1M'][l3]['per_seed'])}) is similar to 2×256, i.e. late-training noise of individual 2×1024 runs is not unusual;
  the group's large spread is between runs, not within a run's late curve.

### 5.2 Evaluation drops and optimisation diagnostics

A *drop* is a fall of more than {f0(DROP_THRESHOLD)} in mean periodic return versus the previous evaluation. For each drop, the maximum of the critic loss and critic gradient norm in the preceding {DIAG_WINDOW:,} training steps is divided by that run's median.

{md(ev_tbl)}

{md(spike_tbl.round(2))}

{md(diag_tbl)}

- Spike events (a contiguous run above 10× the centred 25-point rolling median) occur in all three architectures. The largest logged critic loss in the study is {f2(max_loss)} (2×1024 seed {hi_seed}); it is finite, belongs to a single
  logged value ({f1(sp3['critic_loss_max_ratio_to_median'])}× its local median, spike event at step {', '.join(spike_step) or 'n/a'}), and the policy shows no matching evaluation drop around it
  (periodic means {f0(float(r3[int(hi_seed)].periodic.set_index('step').loc[725_000, 'mean_return']))} at 725k and {f0(float(r3[int(hi_seed)].periodic.set_index('step').loc[750_000, 'mean_return']))} at 750k). That is not evidence of divergence.
- {drop_txt}
- Pooled over the nine runs ({spear['n_intervals']} evaluation intervals), the Spearman rank correlation between the change in return and the critic-loss (critic-gradient-norm) window ratio is {spear['spearman_return_change_vs_critic_loss_window_ratio']:+.2f} ({spear['spearman_return_change_vs_critic_grad_norm_window_ratio']:+.2f}),
  but it falls to {spear['spearman_after_250k_critic_loss']:+.2f} ({spear['spearman_after_250k_critic_grad_norm']:+.2f}) when only intervals after 250k steps are used. The early negative value reflects training phase
  (returns rise fastest when losses are still small), not a mechanism. This analysis is **exploratory**: pooling runs and consecutive intervals ignores temporal dependence (the intervals are autocorrelated and their windows overlap), so the effective sample size is much smaller than {spear['n_intervals']}, no p-value is reported, and the coefficients are descriptive only — a rank correlation would not be evidence of causation even if it were larger.
- Across architectures, median critic loss, critic gradient norm and mean |Q| are lower for 2×1024 than for 2×256 in this sample, and within 2×1024 the best-performing seed has the *largest* loss, |Q| and gradient norm.
  Q-value magnitude scales with return, so larger |Q| in the highest-return seed is expected and does not indicate instability.
- The entropy coefficient α is auto-tuned; its run medians (table) are similar for most runs, but 2×1024 seed 0 follows a visibly lower α trajectory for much of training and seed 1 ends higher than any other run (figure 05). Three seeds cannot show whether this relates to the seed-level return differences.

## 6. Compute and network scaling

{md(comp_tbl)}

- Parameters grow {st['architectures'][l2]['param_ratio_vs_2x256']:.2f}× (2×512) and {st['architectures'][l3]['param_ratio_vs_2x256']:.2f}× (2×1024) relative to 2×256, whereas the median runtimes excluding the flagged run are {f2(med_ex[l1])}, {f2(med_ex[l2])} and {f2(med_ex[l3])} h:
  runtime is nearly flat in width over this range on this hardware. This is an observation, not a controlled measurement.
- Runs flagged as more than {SLOW_FACTOR}× the median of all nine runtimes ({f2(st['runtime_median_all_runs'])} h): {', '.join(f'`{k}`' for k in flagged) or 'none'}. {slow_txt}
  The all-seed 2×256 runtime mean ({f2(st['runtime_hours'][l1]['mean'])} h) is distorted by this run.
- Runtimes of 2×1024 ({', '.join(f2(v) for v in st['runtime_hours'][l3]['per_seed'])} h) and 2×512 ({', '.join(f2(v) for v in st['runtime_hours'][l2]['per_seed'])} h) differ by {st['runtime_hours']['pairwise_differences'][pk[2]]['percent_difference']:+.1f}% in the means ({'every 2×1024 run is slower than every 2×512 run' if min(st['runtime_hours'][l3]['per_seed']) > max(st['runtime_hours'][l2]['per_seed']) else 'the groups overlap'});
  this is consistent with a small width cost, but runs were executed at different times on a shared GPU server whose load was not recorded, so it cannot be attributed to width.
- Peak sampled GPU memory of 417 MiB per 2×1024 run was reported by the independent audit; the GPU-monitoring CSVs remain on Gorina4 and are **not** part of the tracked dataset, so this number is not verified here, it is a sampled value (not a hardware maximum), and no GPU-utilisation series is shown.
- Return versus parameters and versus runtime are shown in figure 08. Final return does not increase with parameter count in this sample, and with essentially constant runtime there is no meaningful "return per unit compute" trade-off to extract from three noisy runs.

## 7. Discussion and alternative explanations

**Observed:** the mean final return is {f0(fin[l1]['mean'])} (2×256), {f0(fin[l2]['mean'])} (2×512) and {f0(fin[l3]['mean'])} (2×1024); the 2×1024 group has {fin[l3]['sd'] / fin[l1]['sd']:.1f}× and {fin[l3]['sd'] / fin[l2]['sd']:.1f}× the between-seed SD of 2×256 and 2×512; the best and worst 2×1024 seeds are {f0(seed_gap_1M)} apart at 1M steps, an ordering that appears early and persists.

**Why the strong 2×1024 seed-{hi_seed} result does not show that wider networks are better.** That run reached {f0(r3[int(hi_seed)].final_mean_return)}, the highest final return of all nine runs, yet the other two 2×1024 runs are the lowest and second-lowest of the nine ({', '.join(f0(per_run.sort_values('final_mean_return_10ep')['final_mean_return_10ep'].iloc[i]) for i in (0, 1))}). One seed out of three is a sample of size one for the claim "this architecture can reach high return": it shows that high return is *attainable*, not that it is *typical*. It exceeds the best seed of the narrower networks ({f0(max(fin[l1]['max'], fin[l2]['max']))}) by {f0(r3[int(hi_seed)].final_mean_return - max(fin[l1]['max'], fin[l2]['max']))}.

**Why the lower 2×1024 mean does not show that wider networks are worse.** With a between-seed SD of {f0(fin[l3]['sd'])} and n = 3, the standard error of the 2×1024 mean is about {f0(fin[l3]['sd'] / 3 ** 0.5)}, comparable to the observed mean differences to the other architectures ({f0(abs(fin[l3]['mean'] - fin[l1]['mean']))} and {f0(abs(fin[l3]['mean'] - fin[l2]['mean']))}); the smallest attainable permutation p-value is 0.10; and
the two lowest 2×1024 seeds are still improving at 1M steps (positive late slopes), so the 1M-step comparison may favour architectures that learn faster.

**Alternative explanations (none tested here):**
1. Ordinary training-seed variability (initial weights, replay-buffer sampling, environment randomness) — the most economical explanation for a high-return outlier among three seeds.
2. Evaluation noise: the final evaluation uses only 10 deterministic episodes and the periodic ones only 5; within-run episode SDs are small relative to the between-seed spread, so evaluation noise alone is unlikely to explain the 2×1024 seed differences, but it adds to the visible dips of individual evaluations.
3. Optimisation effects of a fixed learning rate and target-update rate that were not retuned for width; larger networks often need different learning rates or initialisation scales. The 2×1024 critics and policies here use the same hyperparameters as 2×256.
4. Differences in the auto-tuned entropy coefficient (section 5.2) leading to different exploration regimes per seed.
5. Different Git commits per architecture and a shared, uncontrolled compute environment (runtime only; the learning dynamics are not expected to depend on server load).

## 8. Scientific limitations

- n = 3 training seeds per architecture; exact permutation tests cannot reach p < 0.10; standardised effect sizes from three seeds are very imprecise; bands in the figures are descriptive and not confidence intervals.
- One environment (HalfCheetah-v5), one hyperparameter setting, three widths with fixed depth 2, 1M steps (several runs still improving).
- Architectures were trained at different commits; baseline seed 2 had an unexplained runtime anomaly; shared-GPU load, peak GPU utilisation and other possible confounds of the server are not in the tracked data.
- The same numeric seed in different architectures is not a matched sample (no paired analysis was performed).
- The rank correlations between return changes and critic diagnostics pool nine runs and consecutive, temporally dependent intervals; they are exploratory, carry no inferential weight, and are confounded by training phase.
- Drop/spike/outlier criteria are descriptive heuristics with arbitrary thresholds ({f0(DROP_THRESHOLD)} return, 10× rolling median, {wc.LOW_RETURN_FRACTION:g}× evaluation median); they do not identify causes.
- Training diagnostics are logged every 1,000 steps and smoothed in figures; single-point spikes can be hidden by smoothing (figure 04 shows the raw values).

## 9. Conclusions (supported by the measured data)

1. All nine runs validate. In this experiment, increasing the hidden width from 256 to 512 to 1024 did **not** increase the mean final return ({f0(fin[l1]['mean'])} → {f0(fin[l2]['mean'])} → {f0(fin[l3]['mean'])}) or the sample-efficiency proxy
   ({f0(auc[l1]['mean'])} → {f0(auc[l2]['mean'])} → {f0(auc[l3]['mean'])}); the differences are within or near seed-to-seed variation and are not statistically testable with n = 3.
2. The 2×1024 group differs mainly in its **between-seed spread** ({f0(fin[l3]['sd'])} vs {f0(fin[l1]['sd'])} and {f0(fin[l2]['sd'])}), which is visible from the earliest evaluations and persists; it is not a late-training effect.
3. No evidence was found that large critic losses (maximum {f2(max_loss)}) or evaluation drops reflect divergence or are demonstrably linked to each other; all logged diagnostics are finite and no run shows a persistent collapse of return.
4. Runtime was essentially independent of width over a {st['architectures'][l3]['param_ratio_vs_2x256']:.1f}× parameter range on this shared hardware (observational).
5. The data are compatible with the project hypothesis that naive width scaling does not guarantee improvement, but they cannot establish it, nor the opposite; more seeds, width-specific hyperparameter tuning and a common code revision are needed for firm conclusions.

## 10. Files and reproduction

Figures (PNG + PDF): `01_learning_curves_comparison`, `02_final_returns`, `03_throughput_and_runtime`, `04_critic_loss`, `05_entropy_alpha`,
`06_mean_abs_q_and_critic_grad_norm`, `07_return_difference_over_training`, `08_variability_and_scaling`.
Tables: `width_comparison_per_run.csv`, `width_comparison_checkpoints.csv`, `width_comparison_periodic_aggregate.csv`, `width_comparison_stats.json`.

Reproduce (repo root, local plotting environment; no training, no GPU):

```
.\\.venv\\Scripts\\python.exe -m unittest discover -s analysis/tests -v
.\\.venv\\Scripts\\python.exe .\\analysis\\plot_width_comparison.py
```

Text outputs and PNG files are regenerated byte-identically; PDF files embed no timestamp but may differ with matplotlib versions.
"""
    (out / "width_comparison_summary.md").write_text(text, encoding="utf-8")


def main(out_dir=None, figures: bool = True) -> None:
    out = Path(out_dir) if out_dir else OUT
    runs = wc.load_all()
    t = build_tables(runs)  # raises ValidationError on any inconsistency
    out.mkdir(parents=True, exist_ok=True)
    t["periodic_aggregate"].to_csv(out / "width_comparison_periodic_aggregate.csv", index=False)
    t["per_run"].to_csv(out / "width_comparison_per_run.csv", index=False)
    t["checkpoints"].to_csv(out / "width_comparison_checkpoints.csv", index=False)
    (out / "width_comparison_stats.json").write_text(wc.dumps_strict(t["stats"]), encoding="utf-8")
    if figures:
        make_figures(runs, t, out)
    write_summary(runs, t, out)
    print("PASS: all nine runs validated and comparable.")
    for n in runs:
        m, s = wc.mean_sd([r.final_mean_return for r in runs[n]])
        print(f"{n}: final return {m:,.2f} +/- {s:,.2f} (sample SD, n=3)")
    print(f"Outputs in {out}: {len(list(out.iterdir()))} files")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--no-figures"]
    main(args[0] if args else None, figures="--no-figures" not in sys.argv[1:])
