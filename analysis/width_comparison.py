"""Loading, validation and statistics for the 2x256 vs 2x512 SAC width comparison.

Pure analysis code (pandas/numpy only). Original run files are only ever read.
Anything inconsistent raises ValidationError; nothing is silently repaired.
"""
from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

import numpy as np
import pandas as pd

ANALYSIS_DIR = Path(__file__).resolve().parent
DATA_ROOT = ANALYSIS_DIR / "data"
SEEDS = (0, 1, 2)
TOTAL_STEPS = 1_000_000
EVAL_STEPS = np.arange(25_000, TOTAL_STEPS + 1, 25_000, dtype=np.int64)
EVAL_KINDS = ("periodic", "final")
DEFAULT_LOG_INTERVAL = 1000
# An evaluation episode is a "low-return outlier" if its return is below this fraction of the
# median episode return of the same evaluation (descriptive heuristic, not a failure diagnosis).
LOW_RETURN_FRACTION = 0.5
DEFAULT_WIDTH = 256  # frozen baseline; baseline configs omit `hidden_width`
REQUIRED_FILES = ("config.json", "metadata.json", "status.json",
                  "eval_metrics.csv", "train_metrics.csv", "episode_metrics.csv")
TRAIN_FINITE_COLUMNS = ("sps", "critic_loss", "alpha", "q_mean_abs", "critic_grad_norm")
# Config keys allowed to differ between runs/architectures in a controlled width comparison.
CONFIG_IGNORED_ACROSS_RUNS = ("seed", "run_id")
CONFIG_IGNORED_ACROSS_ARCHS = CONFIG_IGNORED_ACROSS_RUNS + ("hidden_width",)


class ValidationError(Exception):
    """Scientific data are missing or inconsistent."""


@dataclass(frozen=True)
class ArchSpec:
    name: str
    label: str
    data_dir: Path
    run_prefix: str
    git_commit: str
    hidden_width: int
    total_params: int


SPECS = {
    "baseline_2x256": ArchSpec(
        "baseline_2x256", "2×256", DATA_ROOT / "baseline", "BASELINE_1M",
        "1a9f2405b5fdb9cffa379b72f569a0a3a01515e4", 256, 217_870),
    "width_2x512": ArchSpec(
        "width_2x512", "2×512", DATA_ROOT / "width_2x512", "WIDTH_2x512_1M",
        "f25b3a0cb93da0a605b3dbac6254a866f83f664e", 512, 828_942),
}


@dataclass
class Run:
    spec: ArchSpec
    seed: int
    run_dir: Path
    config: dict
    meta: dict
    periodic: pd.DataFrame
    final: pd.Series
    train: pd.DataFrame
    hidden_width: int

    @property
    def final_mean_return(self) -> float:
        return float(self.final["mean_return"])

    @property
    def runtime_hours(self) -> float:
        return float(self.meta["runtime_seconds"]) / 3600


def _fail(run_label: str, msg: str) -> NoReturn:
    raise ValidationError(f"{run_label}: {msg}")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _check_return_arrays(label: str, evaluation: pd.DataFrame) -> None:
    """Per-row `returns` arrays must match num_episodes, be finite, and reproduce the reported
    mean and *population* SD (ddof=0), the convention of the frozen run logger."""
    for _, row in evaluation.iterrows():
        where = f"{row['kind']} evaluation at step {int(row['step'])}"
        try:
            returns = np.asarray(json.loads(row["returns"]), dtype=float)
        except (TypeError, ValueError) as exc:
            _fail(label, f"{where}: returns array is not valid JSON ({exc})")
        if returns.ndim != 1 or len(returns) != int(row["num_episodes"]):
            _fail(label, f"{where}: returns array has {returns.size} episodes, expected {int(row['num_episodes'])}")
        if not np.isfinite(returns).all():
            _fail(label, f"{where}: non-finite values in returns array (not finite)")
        if not math.isclose(float(returns.mean()), float(row["mean_return"]), rel_tol=1e-9, abs_tol=1e-6):
            _fail(label, f"{where}: reported mean_return disagrees with the returns array")
        if not math.isclose(float(returns.std()), float(row["std_return"]), rel_tol=1e-9, abs_tol=1e-6):
            _fail(label, f"{where}: reported std_return is not the population std (ddof=0) of the returns array")


def _check_training_steps(label: str, train: pd.DataFrame, log_interval: int) -> None:
    steps = train["step"].to_numpy()
    if len(steps) < 2:
        _fail(label, "training log is empty or has a single row")
    diffs = np.diff(steps)
    if (diffs <= 0).any():
        _fail(label, "training log steps are duplicated or not strictly increasing")
    if not (diffs == log_interval).all():
        _fail(label, f"training log step spacing is irregular (expected constant {log_interval})")
    if steps[-1] < TOTAL_STEPS - log_interval:
        _fail(label, f"training log truncated: last step {int(steps[-1])} < {TOTAL_STEPS - log_interval}")


def load_run(spec: ArchSpec, seed: int) -> Run:
    label = f"{spec.name} seed {seed}"
    matches = sorted(spec.data_dir.glob(f"{spec.run_prefix}_seed{seed}_*"))
    matches = [m for m in matches if m.is_dir()]
    if len(matches) != 1:
        _fail(label, f"expected exactly one run directory {spec.run_prefix}_seed{seed}_* in "
                     f"{spec.data_dir}, found {len(matches)}")
    folder = matches[0]
    missing = [f for f in REQUIRED_FILES if not (folder / f).is_file()]
    if missing:
        _fail(label, f"missing files {missing} in {folder}")

    config = _read_json(folder / "config.json")
    meta = _read_json(folder / "metadata.json")
    status = _read_json(folder / "status.json")

    for src_name, src in (("status.json", status), ("metadata.json", meta)):
        if src.get("status") != "completed" or src.get("steps_completed") != TOTAL_STEPS:
            _fail(label, f"{src_name} status is not 'completed' at {TOTAL_STEPS} steps")
    if meta.get("env_id") != "HalfCheetah-v5" or config.get("env_id") != "HalfCheetah-v5":
        _fail(label, "unexpected environment (expected HalfCheetah-v5)")
    if meta.get("seed") != seed or config.get("seed") != seed:
        _fail(label, "seed in metadata/config does not match the expected seed")
    if config.get("total_steps") != TOTAL_STEPS:
        _fail(label, "config total_steps is not 1,000,000")
    if meta.get("config") != config:
        _fail(label, "metadata.json config differs from config.json")
    if meta.get("git_commit") != spec.git_commit:
        _fail(label, f"git commit {meta.get('git_commit')} != expected {spec.git_commit}")
    if meta.get("git_dirty") is not False:
        _fail(label, "git working tree was dirty (or dirty state missing)")
    if meta.get("parameter_counts", {}).get("total") != spec.total_params:
        _fail(label, f"parameter count {meta.get('parameter_counts', {}).get('total')} != "
                     f"expected {spec.total_params}")

    if "hidden_width" in config:
        width = config["hidden_width"]
    elif spec.hidden_width == DEFAULT_WIDTH:
        width = DEFAULT_WIDTH  # baseline predates the field: frozen default
    else:
        _fail(label, f"config lacks hidden_width (expected {spec.hidden_width})")
    if width != spec.hidden_width:
        _fail(label, f"config hidden_width {width} != expected {spec.hidden_width}")

    evaluation = pd.read_csv(folder / "eval_metrics.csv")
    train = pd.read_csv(folder / "train_metrics.csv")
    for cols, df, name in (({"step", "kind", "num_episodes", "mean_return", "std_return", "returns"}, evaluation, "eval"),
                           (set(TRAIN_FINITE_COLUMNS) | {"step"}, train, "train")):
        if not cols <= set(df.columns):
            _fail(label, f"{name} metrics lack columns {sorted(cols - set(df.columns))}")

    unexpected = sorted(set(evaluation["kind"]) - set(EVAL_KINDS))
    if unexpected:
        _fail(label, f"unexpected evaluation kind(s) {unexpected}; allowed {list(EVAL_KINDS)}")
    periodic = evaluation[evaluation["kind"] == "periodic"].sort_values("step").reset_index(drop=True)
    final = evaluation[evaluation["kind"] == "final"]
    if len(periodic) != len(EVAL_STEPS) or not np.array_equal(periodic["step"].to_numpy(), EVAL_STEPS):
        _fail(label, "periodic evaluations are not exactly steps 25k..1M in 25k increments")
    if len(final) != 1 or int(final.iloc[0]["step"]) != TOTAL_STEPS:
        _fail(label, "expected exactly one final evaluation at step 1,000,000")
    if not (periodic["num_episodes"] == 5).all():
        _fail(label, "periodic evaluation episode count is not 5")
    if int(final.iloc[0]["num_episodes"]) != 10:
        _fail(label, "final evaluation episode count is not 10")
    if not np.isfinite(evaluation[["mean_return", "std_return"]].to_numpy(dtype=float)).all():
        _fail(label, "non-finite evaluation metrics")
    _check_return_arrays(label, evaluation)
    if not math.isclose(float(meta["final_eval"]["mean_return"]), float(final.iloc[0]["mean_return"]), abs_tol=1e-5):
        _fail(label, "metadata final evaluation disagrees with eval_metrics.csv")
    meta_returns = meta["final_eval"].get("returns")
    if meta_returns is not None and not np.allclose(
            meta_returns, json.loads(final.iloc[0]["returns"]), rtol=0, atol=1e-6):
        _fail(label, "metadata final evaluation returns differ from eval_metrics.csv")

    _check_training_steps(label, train, int(config.get("log_interval", DEFAULT_LOG_INTERVAL)))
    values = train[list(TRAIN_FINITE_COLUMNS)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        _fail(label, "non-finite training diagnostics")

    return Run(spec, seed, folder, config, meta, periodic, final.iloc[0], train, width)


def load_all(specs: dict[str, ArchSpec] | None = None, seeds=SEEDS) -> dict[str, list[Run]]:
    specs = specs or SPECS
    return {name: [load_run(spec, s) for s in seeds] for name, spec in specs.items()}


def _config_diff(a: dict, b: dict, ignore) -> list[str]:
    keys = (set(a) | set(b)) - set(ignore)
    return sorted(k for k in keys if a.get(k) != b.get(k))


def check_comparable(runs_by_arch: dict[str, list[Run]]) -> None:
    """All runs share the protocol; only hidden_width (and run_id/seed) may differ."""
    seed_sets = {name: sorted(r.seed for r in rs) for name, rs in runs_by_arch.items()}
    if len({tuple(v) for v in seed_sets.values()}) != 1:
        raise ValidationError(f"seed sets differ between architectures: {seed_sets}")
    reference = None
    for name, rs in runs_by_arch.items():
        for r in rs:
            if reference is None:
                reference = r
                continue
            same_arch = r.spec.name == reference.spec.name
            ignore = CONFIG_IGNORED_ACROSS_RUNS if same_arch else CONFIG_IGNORED_ACROSS_ARCHS
            diff = _config_diff(reference.config, r.config, ignore)
            if diff:
                raise ValidationError(
                    f"{name} seed {r.seed}: config differs from reference beyond the width variable: {diff}")


# ---------------------------------------------------------------- statistics

def mean_sd(values) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1))


def aggregate_periodic(frames: list[pd.DataFrame]) -> pd.DataFrame:
    """Cross-seed mean and sample SD (ddof=1) of mean_return at each evaluation step."""
    steps = frames[0]["step"].to_numpy()
    for f in frames[1:]:
        if not np.array_equal(f["step"].to_numpy(), steps):
            raise ValidationError("periodic evaluation steps are not aligned across seeds")
    matrix = np.column_stack([f["mean_return"].to_numpy(dtype=float) for f in frames])
    return pd.DataFrame({"step": steps, "mean": matrix.mean(axis=1), "sd": matrix.std(axis=1, ddof=1)})


def effect_size(a, b) -> dict:
    """Descriptive comparison b vs a (a = reference). Not a significance claim."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    na, nb = len(a), len(b)
    diff = float(b.mean() - a.mean())
    pooled = math.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    d = diff / pooled if pooled > 0 else float("nan")
    hedges = d * (1 - 3 / (4 * (na + nb) - 9))
    se = math.sqrt(a.var(ddof=1) / na + b.var(ddof=1) / nb)
    # Exact two-sided permutation test over all group relabelings (exploratory only).
    pool = np.concatenate([a, b])
    obs = abs(diff)
    count = total = 0
    for idx in itertools.combinations(range(na + nb), na):
        mask = np.zeros(na + nb, dtype=bool)
        mask[list(idx)] = True
        total += 1
        if abs(pool[~mask].mean() - pool[mask].mean()) >= obs - 1e-9:
            count += 1
    return {
        "mean_difference": diff,
        "percent_difference": diff / float(a.mean()) * 100 if a.mean() != 0 else float("nan"),
        "pooled_sd": pooled,
        "cohens_d": d,
        "hedges_g": hedges,
        "welch_se_of_difference": se,
        "permutation_p_two_sided": count / total,
        "permutation_min_attainable_p": 2 / total,
    }


def steps_to_threshold(periodic: pd.DataFrame, threshold: float) -> float:
    """First evaluation step whose mean return is >= threshold (NaN if never)."""
    hit = periodic[periodic["mean_return"] >= threshold]
    return float(hit["step"].iloc[0]) if len(hit) else float("nan")


def max_drawdown(periodic: pd.DataFrame) -> float:
    values = periodic["mean_return"].to_numpy(dtype=float)
    return float((np.maximum.accumulate(values) - values).max())


def window_mean(train: pd.DataFrame, column: str, checkpoint: int, window: int = 25_000) -> float:
    """Mean of `column` over logged rows with checkpoint-window < step <= checkpoint."""
    sel = train[(train["step"] > checkpoint - window) & (train["step"] <= checkpoint)]
    return float(pd.to_numeric(sel[column], errors="coerce").mean()) if len(sel) else float("nan")


def critic_loss_spikes(loss: pd.Series, window: int = 25, factor: float = 10.0) -> dict:
    """Descriptive spike count: contiguous runs of points exceeding factor x rolling median."""
    values = pd.to_numeric(loss, errors="coerce").reset_index(drop=True)
    median = values.rolling(window=window, min_periods=1, center=True).median()
    ratio = values / median
    over = (ratio > factor).to_numpy()
    starts = over & ~np.concatenate([[False], over[:-1]])
    positions = [int(i) for i in np.flatnonzero(starts)]
    return {"n_spikes": len(positions), "max_ratio": float(ratio.max()), "event_positions": positions}


def count_low_return_outlier_evals(periodic: pd.DataFrame, after_step: int = 0) -> int:
    """Periodic evaluations (step > after_step) containing a low-return outlier episode: one whose
    return is below LOW_RETURN_FRACTION x the median episode return of that evaluation. This is a
    descriptive heuristic; it does not establish why the episode scored low."""
    count = 0
    for _, row in periodic[periodic["step"] > after_step].iterrows():
        returns = np.asarray(json.loads(row["returns"]), dtype=float)
        if (returns < LOW_RETURN_FRACTION * np.median(returns)).any():
            count += 1
    return count


def flag_slow_runs(runtimes: dict[str, float], factor: float = 1.5) -> list[str]:
    median = float(np.median(list(runtimes.values())))
    return sorted(k for k, v in runtimes.items() if v > factor * median)


def _json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, np.ndarray)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return value if math.isfinite(value) else None  # NaN/Inf are not standard JSON
    return obj


def dumps_strict(obj, indent: int = 2) -> str:
    """Standards-compliant JSON: undefined (NaN/Inf) values become null; allow_nan is disabled."""
    return json.dumps(_json_safe(obj), indent=indent, allow_nan=False)
