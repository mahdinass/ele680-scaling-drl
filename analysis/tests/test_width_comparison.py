"""Tests for analysis/width_comparison.py (local plotting env; no torch needed).

Run from repo root: .venv\\Scripts\\python.exe -m unittest discover -s analysis/tests -v
"""
from __future__ import annotations

import copy
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ANALYSIS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSIS))

import width_comparison as wc  # noqa: E402

STEPS = np.arange(25_000, 1_000_001, 25_000)
COMMIT_A = "a" * 40
COMMIT_B = "b" * 40
BASE_CONFIG = {
    "env_id": "HalfCheetah-v5", "seed": 0, "device": "cuda", "total_steps": 1_000_000,
    "buffer_size": 1_000_000, "batch_size": 256, "learning_starts": 5000,
    "policy_frequency": 2, "target_network_frequency": 1, "gamma": 0.99, "tau": 0.005,
    "policy_lr": 0.0003, "q_lr": 0.001, "eval_interval": 25000, "eval_episodes": 5,
    "final_eval_episodes": 10, "output_dir": "results", "run_id": "X", "log_interval": 1000,
    "checkpoint_interval": 100000, "tensorboard": True,
}


def episode_returns(mean: float, n: int) -> list[float]:
    """n symmetric offsets around `mean`: exact mean, population SD independent of `mean`."""
    return [float(mean + o) for o in np.linspace(-1.0, 1.0, n)]


def make_spec(root: Path, name="small", commit=COMMIT_A, width=256, params=1000, prefix="SMALL"):
    return wc.ArchSpec(name=name, label=name, data_dir=root / name, run_prefix=prefix,
                       git_commit=commit, hidden_width=width, total_params=params)


def write_run(spec: wc.ArchSpec, seed: int, final_mean=1000.0, level=5000.0, **over):
    """Write a valid synthetic run directory; `over` can sabotage individual pieces."""
    d = spec.data_dir / f"{spec.run_prefix}_seed{seed}_2026"
    d.mkdir(parents=True, exist_ok=True)
    config = copy.deepcopy(BASE_CONFIG)
    config["seed"] = seed
    if spec.hidden_width != 256:
        config["hidden_width"] = spec.hidden_width
    config.update(over.get("config", {}))
    meta = {
        "status": "completed", "steps_completed": 1_000_000, "env_id": "HalfCheetah-v5",
        "seed": seed, "git_commit": spec.git_commit, "git_dirty": False,
        "runtime_seconds": 18000.0, "config": config,
        "parameter_counts": {"total": spec.total_params},
        "final_eval": {"step": 1_000_000, "num_episodes": 10, "mean_return": final_mean},
    }
    meta.update(over.get("meta", {}))
    status = {"status": "completed", "steps_completed": 1_000_000}
    status.update(over.get("status", {}))
    rng = np.random.default_rng(seed)
    pmeans = level * (1 - np.exp(-STEPS / 3e5)) + rng.normal(0, 1, len(STEPS))
    periodic = pd.DataFrame({
        "step": STEPS, "kind": "periodic", "num_episodes": 5, "mean_return": pmeans,
        "std_return": float(np.std(episode_returns(0.0, 5))),
        "returns": [json.dumps(episode_returns(m, 5)) for m in pmeans]})
    final = pd.DataFrame([{"step": 1_000_000, "kind": "final", "num_episodes": 10,
                           "mean_return": final_mean, "std_return": float(np.std(episode_returns(0.0, 10))),
                           "returns": json.dumps(episode_returns(final_mean, 10))}])
    evals = periodic if over.get("drop_final") else pd.concat([periodic, final])
    if "evals" in over:
        evals = over["evals"](evals)
    tsteps = np.arange(6001, 1_000_001, 1000)
    train = pd.DataFrame({"step": tsteps, "sps": 50.0, "critic_loss": 1.0, "alpha": 0.2,
                          "q_mean_abs": 100.0, "critic_grad_norm": 10.0, "actor_loss": -1.0})
    if "train" in over:
        train = over["train"](train)
    files = {"config.json": config, "metadata.json": meta, "status.json": status}
    for fname, obj in files.items():
        if fname != over.get("omit"):
            (d / fname).write_text(json.dumps(obj), encoding="utf-8")
    for fname, df in (("eval_metrics.csv", evals), ("train_metrics.csv", train),
                      ("episode_metrics.csv", pd.DataFrame({"step": [1], "episode_return": [0.0]}))):
        if fname != over.get("omit"):
            df.to_csv(d / fname, index=False)
    return d


class LoadRunValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.spec = make_spec(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def load(self, seed=0):
        return wc.load_run(self.spec, seed)

    def test_valid_run_loads(self):
        write_run(self.spec, 0, final_mean=1234.5)
        run = self.load()
        self.assertEqual(run.seed, 0)
        self.assertEqual(len(run.periodic), 40)
        self.assertAlmostEqual(run.final_mean_return, 1234.5)

    def test_missing_file_fails(self):
        write_run(self.spec, 0, omit="eval_metrics.csv")
        with self.assertRaisesRegex(wc.ValidationError, "missing"):
            self.load()

    def test_missing_run_directory_fails(self):
        self.spec.data_dir.mkdir(parents=True)
        with self.assertRaisesRegex(wc.ValidationError, "run directory"):
            self.load()

    def test_wrong_commit_fails(self):
        write_run(self.spec, 0, meta={"git_commit": COMMIT_B})
        with self.assertRaisesRegex(wc.ValidationError, "commit"):
            self.load()

    def test_dirty_tree_fails(self):
        write_run(self.spec, 0, meta={"git_dirty": True})
        with self.assertRaisesRegex(wc.ValidationError, "dirty"):
            self.load()

    def test_incomplete_status_fails(self):
        write_run(self.spec, 0, status={"status": "failed"})
        with self.assertRaisesRegex(wc.ValidationError, "status"):
            self.load()

    def test_wrong_parameter_count_fails(self):
        write_run(self.spec, 0, meta={"parameter_counts": {"total": 7}})
        with self.assertRaisesRegex(wc.ValidationError, "parameter"):
            self.load()

    def test_seed_mismatch_fails(self):
        write_run(self.spec, 0, meta={"seed": 1})
        with self.assertRaisesRegex(wc.ValidationError, "seed"):
            self.load()

    def test_wrong_environment_fails(self):
        write_run(self.spec, 0, meta={"env_id": "Hopper-v5"})
        with self.assertRaisesRegex(wc.ValidationError, "environment"):
            self.load()

    def test_missing_final_evaluation_fails(self):
        write_run(self.spec, 0, drop_final=True)
        with self.assertRaisesRegex(wc.ValidationError, "final"):
            self.load()

    def test_missing_periodic_step_fails(self):
        write_run(self.spec, 0, evals=lambda e: e[e.step != 500_000])
        with self.assertRaisesRegex(wc.ValidationError, "periodic"):
            self.load()

    def test_wrong_episode_count_fails(self):
        def sabotage(e):
            e = e.copy()
            e.loc[e.index[3], "num_episodes"] = 4
            return e
        write_run(self.spec, 0, evals=sabotage)
        with self.assertRaisesRegex(wc.ValidationError, "episode"):
            self.load()

    def test_nonfinite_evaluation_fails(self):
        def sabotage(e):
            e = e.copy()
            e.loc[e.index[2], "mean_return"] = float("nan")
            return e
        write_run(self.spec, 0, evals=sabotage)
        with self.assertRaisesRegex(wc.ValidationError, "finite"):
            self.load()

    def test_nonfinite_training_metric_fails(self):
        def sabotage(t):
            t = t.copy()
            t.loc[10, "critic_loss"] = float("inf")
            return t
        write_run(self.spec, 0, train=sabotage)
        with self.assertRaisesRegex(wc.ValidationError, "finite"):
            self.load()

    def test_metadata_final_disagreeing_with_csv_fails(self):
        write_run(self.spec, 0, meta={"final_eval": {"step": 1_000_000, "num_episodes": 10,
                                                      "mean_return": 5.0}}, final_mean=1000.0)
        with self.assertRaisesRegex(wc.ValidationError, "final"):
            self.load()

    def test_unexpected_evaluation_kind_fails(self):
        def extra(e):
            row = e[e.kind == "final"].copy()
            row["kind"] = "debug"
            return pd.concat([e, row])
        write_run(self.spec, 0, evals=extra)
        with self.assertRaisesRegex(wc.ValidationError, "kind"):
            self.load()

    def test_duplicate_final_row_fails(self):
        write_run(self.spec, 0, evals=lambda e: pd.concat([e, e[e.kind == "final"]]))
        with self.assertRaisesRegex(wc.ValidationError, "final"):
            self.load()

    def test_duplicate_periodic_step_fails(self):
        write_run(self.spec, 0, evals=lambda e: pd.concat([e, e[e.step == 25_000]]))
        with self.assertRaisesRegex(wc.ValidationError, "periodic"):
            self.load()

    def test_missing_returns_column_fails(self):
        write_run(self.spec, 0, evals=lambda e: e.drop(columns=["returns"]))
        with self.assertRaisesRegex(wc.ValidationError, "returns"):
            self.load()

    @staticmethod
    def _tamper(fn, row=3):
        def sabotage(e):
            e = e.reset_index(drop=True)
            e.at[row, "returns"] = json.dumps(fn(json.loads(e.at[row, "returns"])))
            return e
        return sabotage

    def test_returns_array_wrong_length_fails(self):
        write_run(self.spec, 0, evals=self._tamper(lambda r: r[:-1]))
        with self.assertRaisesRegex(wc.ValidationError, "returns.*episodes"):
            self.load()

    def test_returns_array_nonfinite_fails(self):
        write_run(self.spec, 0, evals=self._tamper(lambda r: [float("inf")] + r[1:]))
        with self.assertRaisesRegex(wc.ValidationError, "finite"):
            self.load()

    def test_returns_array_unparseable_fails(self):
        def sabotage(e):
            e = e.reset_index(drop=True)
            e.at[2, "returns"] = "not json"
            return e
        write_run(self.spec, 0, evals=sabotage)
        with self.assertRaisesRegex(wc.ValidationError, "returns"):
            self.load()

    def test_reported_mean_disagreeing_with_returns_fails(self):
        write_run(self.spec, 0, evals=self._tamper(lambda r: [x + 50 for x in r]))
        with self.assertRaisesRegex(wc.ValidationError, "mean"):
            self.load()

    def test_reported_std_must_be_population_std(self):
        def sample_sd(e):
            e = e.reset_index(drop=True)
            r = json.loads(e.at[3, "returns"])
            e.at[3, "std_return"] = float(np.std(r, ddof=1))  # wrong convention
            return e
        write_run(self.spec, 0, evals=sample_sd)
        with self.assertRaisesRegex(wc.ValidationError, "std"):
            self.load()

    def test_final_returns_must_match_metadata_returns(self):
        write_run(self.spec, 0, final_mean=1000.0)
        d = self.spec.data_dir / f"{self.spec.run_prefix}_seed0_2026"
        meta = json.loads((d / "metadata.json").read_text())
        meta["final_eval"]["returns"] = episode_returns(1000.0, 10)[:9] + [0.0]
        (d / "metadata.json").write_text(json.dumps(meta))
        with self.assertRaisesRegex(wc.ValidationError, "final"):
            self.load()

    def test_duplicate_training_steps_fail(self):
        def dup(t):
            return pd.concat([t.iloc[:10], t.iloc[9:10], t.iloc[10:]]).reset_index(drop=True)
        write_run(self.spec, 0, train=dup)
        with self.assertRaisesRegex(wc.ValidationError, "training"):
            self.load()

    def test_decreasing_training_steps_fail(self):
        def swap(t):
            t = t.copy()
            t.loc[[10, 11], "step"] = t.loc[[11, 10], "step"].to_numpy()
            return t
        write_run(self.spec, 0, train=swap)
        with self.assertRaisesRegex(wc.ValidationError, "training"):
            self.load()

    def test_irregular_training_step_spacing_fails(self):
        write_run(self.spec, 0, train=lambda t: t.drop(index=[50, 51]).reset_index(drop=True))
        with self.assertRaisesRegex(wc.ValidationError, "training"):
            self.load()

    def test_truncated_training_log_fails(self):
        write_run(self.spec, 0, train=lambda t: t.iloc[:500])
        with self.assertRaisesRegex(wc.ValidationError, "truncated"):
            self.load()

    def test_baseline_without_hidden_width_means_256(self):
        write_run(self.spec, 0)
        self.assertEqual(self.load().hidden_width, 256)

    def test_config_width_must_match_spec(self):
        write_run(self.spec, 0, config={"hidden_width": 512})
        with self.assertRaisesRegex(wc.ValidationError, "hidden_width"):
            self.load()

    def test_width_spec_requires_recorded_width(self):
        spec = make_spec(self.root, name="wide", commit=COMMIT_B, width=512, params=4000, prefix="WIDE")
        d = write_run(spec, 0)
        cfg = json.loads((d / "config.json").read_text())
        del cfg["hidden_width"]
        (d / "config.json").write_text(json.dumps(cfg))
        meta = json.loads((d / "metadata.json").read_text())
        meta["config"] = cfg  # keep metadata consistent so only the width check can fire
        (d / "metadata.json").write_text(json.dumps(meta))
        with self.assertRaisesRegex(wc.ValidationError, "hidden_width"):
            wc.load_run(spec, 0)


class Comparability(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.small = make_spec(self.root)
        self.wide = make_spec(self.root, name="wide", commit=COMMIT_B, width=512, params=4000, prefix="WIDE")

    def tearDown(self):
        self.tmp.cleanup()

    def runs(self, wide_over=None):
        for s in (0, 1):
            write_run(self.small, s)
            write_run(self.wide, s, **(wide_over or {}))
        return {self.small.name: [wc.load_run(self.small, s) for s in (0, 1)],
                self.wide.name: [wc.load_run(self.wide, s) for s in (0, 1)]}

    def test_only_width_differing_is_comparable(self):
        wc.check_comparable(self.runs())

    def test_other_hyperparameter_difference_fails(self):
        runs = self.runs(wide_over={"config": {"gamma": 0.98}})
        with self.assertRaisesRegex(wc.ValidationError, "gamma"):
            wc.check_comparable(runs)

    def test_seed_sets_must_match(self):
        runs = self.runs()
        runs[self.wide.name] = runs[self.wide.name][:1]
        with self.assertRaisesRegex(wc.ValidationError, "seed"):
            wc.check_comparable(runs)


class Statistics(unittest.TestCase):
    def test_effect_size_known_values(self):
        a = [10.0, 12.0, 14.0]  # mean 12, sd 2
        b = [20.0, 22.0, 24.0]  # mean 22, sd 2
        e = wc.effect_size(a, b)
        self.assertAlmostEqual(e["mean_difference"], 10.0)
        self.assertAlmostEqual(e["pooled_sd"], 2.0)
        self.assertAlmostEqual(e["cohens_d"], 5.0)
        self.assertAlmostEqual(e["percent_difference"], 10.0 / 12.0 * 100)
        self.assertLess(e["hedges_g"], e["cohens_d"])  # small-sample correction shrinks d
        # Fully separated groups of 3 v 3: exact two-sided permutation p = 2/20.
        self.assertAlmostEqual(e["permutation_p_two_sided"], 0.1)

    def test_effect_size_identical_groups_p_is_one(self):
        e = wc.effect_size([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        self.assertAlmostEqual(e["permutation_p_two_sided"], 1.0)
        self.assertAlmostEqual(e["mean_difference"], 0.0)

    def test_effect_size_zero_reference_mean_gives_nan_percent(self):
        e = wc.effect_size([0.0, 0.0, 0.0], [0.0, 1.0, 2.0])
        self.assertTrue(math.isnan(e["percent_difference"]))
        self.assertAlmostEqual(e["mean_difference"], 1.0)

    def test_effect_size_zero_variance_gives_nan_standardised_effects(self):
        e = wc.effect_size([1.0, 1.0, 1.0], [1.0, 1.0, 1.0])
        self.assertTrue(math.isnan(e["cohens_d"]))
        self.assertTrue(math.isnan(e["hedges_g"]))

    def test_mean_sd_uses_sample_sd(self):
        m, s = wc.mean_sd([1.0, 2.0, 3.0])
        self.assertAlmostEqual(m, 2.0)
        self.assertAlmostEqual(s, 1.0)  # ddof=1 (ddof=0 would give 0.8165)

    def test_aggregate_periodic_mean_and_sample_sd(self):
        steps = [25_000, 50_000]
        frames = [pd.DataFrame({"step": steps, "mean_return": v}) for v in ([1.0, 10.0], [3.0, 10.0], [5.0, 13.0])]
        agg = wc.aggregate_periodic(frames)
        self.assertEqual(agg["step"].tolist(), steps)
        self.assertAlmostEqual(agg["mean"][0], 3.0)
        self.assertAlmostEqual(agg["sd"][0], 2.0)
        self.assertAlmostEqual(agg["sd"][1], math.sqrt(3.0))

    def test_aggregate_periodic_rejects_misaligned_steps(self):
        a = pd.DataFrame({"step": [1, 2], "mean_return": [1.0, 2.0]})
        b = pd.DataFrame({"step": [1, 3], "mean_return": [1.0, 2.0]})
        with self.assertRaises(wc.ValidationError):
            wc.aggregate_periodic([a, b])

    def test_steps_to_threshold(self):
        df = pd.DataFrame({"step": [25_000, 50_000, 75_000], "mean_return": [100.0, 600.0, 900.0]})
        self.assertEqual(wc.steps_to_threshold(df, 500.0), 50_000)
        self.assertTrue(math.isnan(wc.steps_to_threshold(df, 5000.0)))

    def test_window_mean_uses_preceding_window(self):
        train = pd.DataFrame({"step": [1000, 2000, 3000, 4000], "x": [1.0, 2.0, 3.0, 4.0]})
        # (cp-window, cp] -> steps 3000 and 4000 for cp=4000, window=2000
        self.assertAlmostEqual(wc.window_mean(train, "x", 4000, 2000), 3.5)

    def test_critic_spikes_counts_events_over_rolling_median(self):
        loss = pd.Series(np.ones(100))
        loss[40] = 50.0
        loss[70] = 30.0
        res = wc.critic_loss_spikes(loss, window=25, factor=10.0)
        self.assertEqual(res["n_spikes"], 2)
        self.assertAlmostEqual(res["max_ratio"], 50.0)

    def test_critic_spikes_reports_event_start_positions(self):
        loss = pd.Series(np.ones(100))
        loss[40] = 50.0
        loss[41] = 50.0  # contiguous -> one event starting at 40
        loss[70] = 30.0
        res = wc.critic_loss_spikes(loss, window=25, factor=10.0)
        self.assertEqual(res["n_spikes"], 2)
        self.assertEqual(res["event_positions"], [40, 70])

    def test_low_return_outlier_evals_counts_evaluations_with_a_low_episode(self):
        ok = [100.0, 101.0, 99.0, 100.0, 100.0]
        bad = [100.0, 101.0, 99.0, 100.0, -900.0]
        periodic = pd.DataFrame({
            "step": [100_000, 600_000, 700_000],
            "returns": [json.dumps(ok), json.dumps(bad), json.dumps(ok)],
            "mean_return": [100.0, 20.0, 100.0],
        })
        # outlier = episode return below LOW_RETURN_FRACTION (0.5) x the median of that evaluation
        self.assertEqual(wc.LOW_RETURN_FRACTION, 0.5)
        self.assertEqual(wc.count_low_return_outlier_evals(periodic, after_step=500_000), 1)
        self.assertEqual(wc.count_low_return_outlier_evals(periodic, after_step=0), 1)

    def test_critic_spikes_none_in_flat_series(self):
        self.assertEqual(wc.critic_loss_spikes(pd.Series(np.ones(60)))["n_spikes"], 0)

    def test_max_drawdown(self):
        df = pd.DataFrame({"step": [1, 2, 3, 4], "mean_return": [1.0, 5.0, 2.0, 6.0]})
        self.assertAlmostEqual(wc.max_drawdown(df), 3.0)

    def test_flag_slow_runs_uses_median_multiple(self):
        flagged = wc.flag_slow_runs({"a": 5.0, "b": 5.1, "c": 12.3, "d": 4.9}, factor=1.5)
        self.assertEqual(flagged, ["c"])


class StrictJson(unittest.TestCase):
    @staticmethod
    def _reject_constants(name):
        raise ValueError(f"non-standard JSON constant {name}")

    def test_nan_and_inf_serialise_as_null_recursively(self):
        text = wc.dumps_strict({"a": float("nan"), "b": [1.0, float("inf"), {"c": -float("inf")}],
                                "d": np.float64("nan"), "e": 2.5})
        parsed = json.loads(text, parse_constant=self._reject_constants)
        self.assertEqual(parsed, {"a": None, "b": [1.0, None, {"c": None}], "d": None, "e": 2.5})

    def test_effect_size_with_undefined_percent_is_strict_json(self):
        e = wc.effect_size([0.0, 0.0, 0.0], [0.0, 1.0, 2.0])
        parsed = json.loads(wc.dumps_strict(e), parse_constant=self._reject_constants)
        self.assertIsNone(parsed["percent_difference"])

    def test_numpy_scalars_and_arrays_serialise(self):
        parsed = json.loads(wc.dumps_strict({"i": np.int64(3), "a": np.array([1.0, np.nan])}))
        self.assertEqual(parsed, {"i": 3, "a": [1.0, None]})

    def test_strict_dump_never_emits_nan_token(self):
        self.assertNotIn("NaN", wc.dumps_strict({"x": float("nan")}))


class RealData(unittest.TestCase):
    """Validates the six real scientific runs (skipped if data are absent)."""

    @classmethod
    def setUpClass(cls):
        if not all(spec.data_dir.is_dir() for spec in wc.SPECS.values()):
            raise unittest.SkipTest("scientific data not present")
        cls.runs = wc.load_all()

    def test_six_runs_validate_and_are_comparable(self):
        wc.check_comparable(self.runs)
        self.assertEqual({k: [r.seed for r in v] for k, v in self.runs.items()},
                         {"baseline_2x256": [0, 1, 2], "width_2x512": [0, 1, 2]})

    def test_known_final_returns(self):
        base = [r.final_mean_return for r in self.runs["baseline_2x256"]]
        wide = [r.final_mean_return for r in self.runs["width_2x512"]]
        m_b, s_b = wc.mean_sd(base)
        m_w, s_w = wc.mean_sd(wide)
        self.assertAlmostEqual(m_b, 11906.25, places=1)
        self.assertAlmostEqual(s_b, 497.45, places=1)
        self.assertAlmostEqual(m_w, 11645.37, places=1)
        self.assertAlmostEqual(s_w, 482.26, places=1)


if __name__ == "__main__":
    unittest.main()
