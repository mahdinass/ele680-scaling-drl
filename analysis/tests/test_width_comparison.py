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
    esteps = np.arange(1000, 1_000_001, 1000)
    episodes = pd.DataFrame({"step": esteps, "elapsed_sec": np.linspace(10.0, 18000.0, len(esteps)),
                             "episode_return": np.linspace(-300.0, level, len(esteps)),
                             "episode_length": 1000})
    if "episodes" in over:
        episodes = over["episodes"](episodes)
    files = {"config.json": config, "metadata.json": meta, "status.json": status}
    for fname, obj in files.items():
        if fname != over.get("omit"):
            (d / fname).write_text(json.dumps(obj), encoding="utf-8")
    for fname, df in (("eval_metrics.csv", evals), ("train_metrics.csv", train),
                      ("episode_metrics.csv", episodes)):
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
        # Original two-width case, preserved: restricted to the 2x256 and 2x512 runs.
        six = {k: v for k, v in self.runs.items() if k in ("baseline_2x256", "width_2x512")}
        wc.check_comparable(six)
        self.assertEqual({k: [r.seed for r in v] for k, v in six.items()},
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


class EpisodeAndTrainingLogValidation(unittest.TestCase):
    """Training-episode log (1,000 episodes) and exact training-metric step grid (994 rows)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.spec = make_spec(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_valid_run_has_994_training_rows_and_1000_episodes(self):
        write_run(self.spec, 0)
        run = wc.load_run(self.spec, 0)
        self.assertEqual(len(run.train), 994)
        self.assertEqual(run.train["step"].iloc[0], 6001)
        self.assertEqual(run.train["step"].iloc[-1], 999_001)
        self.assertEqual(len(run.episodes), 1000)

    def test_expected_training_steps_follow_config(self):
        steps = wc.expected_train_steps(BASE_CONFIG)
        self.assertEqual((len(steps), int(steps[0]), int(steps[-1])), (994, 6001, 999_001))

    def test_too_few_episodes_fail(self):
        write_run(self.spec, 0, episodes=lambda e: e.iloc[:-1])
        with self.assertRaisesRegex(wc.ValidationError, "training episode"):
            wc.load_run(self.spec, 0)

    def test_duplicate_episode_row_fails(self):
        write_run(self.spec, 0, episodes=lambda e: pd.concat([e.iloc[:10], e.iloc[9:]]).reset_index(drop=True))
        with self.assertRaisesRegex(wc.ValidationError, "training episode"):
            wc.load_run(self.spec, 0)

    def test_misaligned_episode_steps_fail(self):
        def shift(e):
            e = e.copy()
            e["step"] = e["step"] + 1
            return e
        write_run(self.spec, 0, episodes=shift)
        with self.assertRaisesRegex(wc.ValidationError, "training episode"):
            wc.load_run(self.spec, 0)

    def test_nonfinite_episode_return_fails(self):
        def sabotage(e):
            e = e.copy()
            e.loc[7, "episode_return"] = float("nan")
            return e
        write_run(self.spec, 0, episodes=sabotage)
        with self.assertRaisesRegex(wc.ValidationError, "finite"):
            wc.load_run(self.spec, 0)

    def test_wrong_episode_length_fails(self):
        def sabotage(e):
            e = e.copy()
            e.loc[7, "episode_length"] = 999
            return e
        write_run(self.spec, 0, episodes=sabotage)
        with self.assertRaisesRegex(wc.ValidationError, "episode_length"):
            wc.load_run(self.spec, 0)

    def test_missing_episode_column_fails(self):
        write_run(self.spec, 0, episodes=lambda e: e.drop(columns=["episode_return"]))
        with self.assertRaisesRegex(wc.ValidationError, "episode"):
            wc.load_run(self.spec, 0)

    def test_training_grid_must_start_at_expected_step(self):
        # Regular spacing and an untruncated end, but the first logged row is missing.
        write_run(self.spec, 0, train=lambda t: t.iloc[1:].reset_index(drop=True))
        with self.assertRaisesRegex(wc.ValidationError, "training log.*exactly"):
            wc.load_run(self.spec, 0)


class ThreeArchitectureValidation(unittest.TestCase):
    """Synthetic three-architecture, three-seed study: discovery, provenance, equivalence."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.specs = {
            "a": make_spec(root, name="a", commit=COMMIT_A, width=256, params=1000, prefix="A"),
            "b": make_spec(root, name="b", commit=COMMIT_B, width=512, params=4000, prefix="B"),
            "c": make_spec(root, name="c", commit="c" * 40, width=1024, params=16000, prefix="C"),
        }

    def tearDown(self):
        self.tmp.cleanup()

    def write_all(self, **over_by_arch):
        for key, spec in self.specs.items():
            for s in wc.SEEDS:
                write_run(spec, s, **over_by_arch.get(key, {}))

    def test_all_nine_runs_load_and_are_comparable(self):
        self.write_all()
        runs = wc.load_all(self.specs)
        self.assertEqual(sum(len(v) for v in runs.values()), 9)
        wc.check_comparable(runs)
        wc.require_seed_set(runs)

    def test_each_architecture_must_have_exactly_the_three_seeds(self):
        self.write_all()
        runs = wc.load_all(self.specs)
        runs["c"] = runs["c"][:2]
        with self.assertRaisesRegex(wc.ValidationError, "seed"):
            wc.require_seed_set(runs)

    def test_extra_seed_is_rejected(self):
        self.write_all()
        write_run(self.specs["a"], 3)
        runs = wc.load_all(self.specs)
        runs["a"].append(wc.load_run(self.specs["a"], 3))
        with self.assertRaisesRegex(wc.ValidationError, "seed"):
            wc.require_seed_set(runs)

    def test_missing_run_directory_in_third_architecture_fails(self):
        self.write_all()
        import shutil
        shutil.rmtree(next(self.specs["c"].data_dir.glob("C_seed1_*")))
        with self.assertRaisesRegex(wc.ValidationError, "c seed 1.*run directory"):
            wc.load_all(self.specs)

    def test_duplicate_run_directory_fails(self):
        self.write_all()
        write_run(self.specs["c"], 2)
        (self.specs["c"].data_dir / "C_seed2_2026").rename(self.specs["c"].data_dir / "C_seed2_2026b")
        write_run(self.specs["c"], 2)
        with self.assertRaisesRegex(wc.ValidationError, "found 2"):
            wc.load_run(self.specs["c"], 2)

    def test_wrong_scientific_commit_in_third_architecture_fails(self):
        self.write_all(c={"meta": {"git_commit": COMMIT_A}})
        with self.assertRaisesRegex(wc.ValidationError, "c seed 0.*commit"):
            wc.load_all(self.specs)

    def test_wrong_parameter_count_in_third_architecture_fails(self):
        self.write_all(c={"meta": {"parameter_counts": {"total": 3_230_735}}})
        with self.assertRaisesRegex(wc.ValidationError, "parameter"):
            wc.load_all(self.specs)

    def test_swapped_seed_in_third_architecture_fails(self):
        write_run(self.specs["a"], 0)
        self.write_all()
        d = next(self.specs["c"].data_dir.glob("C_seed2_*"))
        meta = json.loads((d / "metadata.json").read_text())
        cfg = json.loads((d / "config.json").read_text())
        meta["seed"] = cfg["seed"] = 1  # directory says seed 2
        meta["config"] = cfg
        (d / "metadata.json").write_text(json.dumps(meta))
        (d / "config.json").write_text(json.dumps(cfg))
        with self.assertRaisesRegex(wc.ValidationError, "seed"):
            wc.load_all(self.specs)

    def test_hyperparameter_drift_in_third_architecture_is_rejected(self):
        self.write_all(c={"config": {"tau": 0.01}})
        runs = wc.load_all(self.specs)
        with self.assertRaisesRegex(wc.ValidationError, "tau"):
            wc.check_comparable(runs)

    def test_nonfinite_diagnostic_in_third_architecture_names_the_run(self):
        def sabotage(t):
            t = t.copy()
            t.loc[3, "critic_grad_norm"] = float("nan")
            return t
        self.write_all(c={"train": sabotage})
        with self.assertRaisesRegex(wc.ValidationError, "c seed 0.*finite"):
            wc.load_all(self.specs)


class RealSpecsAndNineRuns(unittest.TestCase):
    """Provenance constants and independent reference values for the nine real scientific runs."""

    REFERENCE_FINALS = {
        "baseline_2x256": [12250.47937710342, 11335.909955033207, 12132.360888023763],
        "width_2x512": [11552.7038222595, 12167.23611545362, 11216.167640507401],
        "width_2x1024": [9845.709139447974, 10522.409506619237, 12883.070731103693],
    }
    REFERENCE_RUNTIME_S_1024 = [18937.389, 18752.774, 18647.929]

    @classmethod
    def setUpClass(cls):
        if not all(spec.data_dir.is_dir() for spec in wc.SPECS.values()):
            raise unittest.SkipTest("scientific data not present")
        cls.runs = wc.load_all()

    def test_specs_pin_architecture_commit_and_parameters(self):
        self.assertEqual(list(wc.SPECS), ["baseline_2x256", "width_2x512", "width_2x1024"])
        s = wc.SPECS["width_2x1024"]
        self.assertEqual((s.hidden_width, s.total_params, s.git_commit, s.run_prefix),
                         (1024, 3_230_734, "e533a3c76ef069ffe37affc9a6a15ea2b1ae2e14", "WIDTH_2x1024_1M"))
        self.assertEqual(wc.SPECS["baseline_2x256"].git_commit, "1a9f2405b5fdb9cffa379b72f569a0a3a01515e4")
        self.assertEqual(wc.SPECS["width_2x512"].git_commit, "f25b3a0cb93da0a605b3dbac6254a866f83f664e")

    def test_nine_runs_with_three_seeds_each(self):
        self.assertEqual(sum(len(v) for v in self.runs.values()), 9)
        wc.check_comparable(self.runs)
        wc.require_seed_set(self.runs)

    def test_commits_differ_between_architectures_and_are_preserved(self):
        commits = {n: {r.meta["git_commit"] for r in rs} for n, rs in self.runs.items()}
        self.assertTrue(all(len(c) == 1 for c in commits.values()))
        self.assertEqual(len({next(iter(c)) for c in commits.values()}), 3)

    def test_final_returns_match_independent_audit(self):
        for name, expected in self.REFERENCE_FINALS.items():
            got = [r.final_mean_return for r in self.runs[name]]
            for g, e in zip(got, expected):
                self.assertAlmostEqual(g, e, places=6, msg=name)

    def test_1024_runtimes_and_training_shapes(self):
        got = [float(r.meta["runtime_seconds"]) for r in self.runs["width_2x1024"]]
        self.assertEqual(got, self.REFERENCE_RUNTIME_S_1024)
        for rs in self.runs.values():
            for r in rs:
                self.assertEqual(len(r.train), 994)
                self.assertEqual(len(r.episodes), 1000)
                self.assertEqual(len(r.periodic), 40)

    def test_1024_max_logged_critic_loss_is_finite_and_matches_audit(self):
        seed2 = self.runs["width_2x1024"][2]
        self.assertAlmostEqual(float(seed2.train["critic_loss"].max()), 784.9713, places=3)


class SeedSummaryAndPairwise(unittest.TestCase):
    def test_seed_summary_values(self):
        s = wc.seed_summary([10.0, 12.0, 14.0])
        self.assertAlmostEqual(s["mean"], 12.0)
        self.assertAlmostEqual(s["sd"], 2.0)  # ddof=1
        self.assertEqual((s["min"], s["max"], s["range"], s["n"]), (10.0, 14.0, 4.0, 3))
        self.assertAlmostEqual(s["cv"], 2.0 / 12.0)
        self.assertEqual(s["per_seed"], [10.0, 12.0, 14.0])

    def test_seed_summary_cv_undefined_for_nonpositive_mean(self):
        self.assertTrue(math.isnan(wc.seed_summary([-1.0, 0.0, 1.0])["cv"]))

    def test_pairwise_effects_direction_is_later_minus_reference(self):
        values = {"s": [10.0, 12.0, 14.0], "m": [20.0, 22.0, 24.0], "l": [15.0, 17.0, 19.0]}
        pairs = (("s", "m"), ("s", "l"), ("m", "l"))
        out = wc.pairwise_effects(values, pairs)
        self.assertEqual(list(out), ["m - s", "l - s", "l - m"])
        self.assertAlmostEqual(out["m - s"]["mean_difference"], 10.0)
        self.assertAlmostEqual(out["l - s"]["mean_difference"], 5.0)
        self.assertAlmostEqual(out["l - m"]["mean_difference"], -5.0)
        self.assertAlmostEqual(out["l - m"]["percent_difference"], -5.0 / 22.0 * 100)

    def test_pairwise_matches_effect_size(self):
        a, b = [1.0, 2.0, 4.0], [3.0, 5.0, 9.0]
        self.assertEqual(wc.pairwise_effects({"a": a, "b": b}, (("a", "b"),))["b - a"], wc.effect_size(a, b))

    def test_permutation_resolution_is_limited_for_all_pairs(self):
        out = wc.pairwise_effects({"a": [1.0, 2.0, 3.0], "b": [10.0, 11.0, 12.0]}, (("a", "b"),))
        self.assertAlmostEqual(out["b - a"]["permutation_min_attainable_p"], 0.1)

    def test_no_seed_pairing_is_assumed(self):
        # Relabelling the seed order inside one group must not change the unpaired statistics.
        a, b = [1.0, 5.0, 3.0], [8.0, 2.0, 9.0]
        self.assertEqual(wc.effect_size(a, b), wc.effect_size(a, b[::-1]))

    def test_population_episode_sd_differs_from_sample_sd_across_seeds(self):
        returns = [100.0, 110.0, 90.0]
        self.assertAlmostEqual(float(np.std(returns)), math.sqrt(200 / 3))
        self.assertAlmostEqual(wc.mean_sd(returns)[1], 10.0)
        self.assertNotAlmostEqual(float(np.std(returns)), wc.mean_sd(returns)[1])

    def test_final_and_periodic_evaluations_are_kept_separate(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        spec = make_spec(Path(tmp.name))
        write_run(spec, 0, final_mean=4321.0, level=9000.0)
        run = wc.load_run(spec, 0)
        self.assertEqual(int(run.final["num_episodes"]), 10)
        self.assertEqual(set(run.periodic["num_episodes"]), {5})
        self.assertEqual(len(run.periodic), 40)
        self.assertAlmostEqual(run.final_mean_return, 4321.0)
        self.assertNotAlmostEqual(float(run.periodic["mean_return"].iloc[-1]), 4321.0, places=1)


class TemporalDiagnosticsHelpers(unittest.TestCase):
    def test_eval_drops_finds_falls_larger_than_threshold(self):
        p = pd.DataFrame({"step": [25_000, 50_000, 75_000, 100_000, 125_000],
                          "mean_return": [100.0, 200.0, 150.0, 1.0, 300.0]})
        drops = wc.eval_drops(p, threshold=100.0)
        self.assertEqual([(d["step"], d["drop"]) for d in drops], [(100_000, 149.0)])
        self.assertEqual(wc.eval_drops(p, threshold=40.0)[0]["step"], 75_000)

    def test_eval_drops_empty_for_monotone_curve(self):
        p = pd.DataFrame({"step": [1, 2, 3], "mean_return": [1.0, 2.0, 3.0]})
        self.assertEqual(wc.eval_drops(p, threshold=0.5), [])

    def test_window_max_ratio_is_window_max_over_run_median(self):
        train = pd.DataFrame({"step": [1000, 2000, 3000, 4000, 5000], "x": [1.0, 1.0, 9.0, 1.0, 1.0]})
        # window (1000, 3000] contains the 9.0; run median is 1.0
        self.assertAlmostEqual(wc.window_max_ratio(train, "x", 3000, 2000), 9.0)
        self.assertAlmostEqual(wc.window_max_ratio(train, "x", 5000, 1000), 1.0)

    def test_window_max_ratio_nan_for_empty_window(self):
        train = pd.DataFrame({"step": [1000, 2000], "x": [1.0, 2.0]})
        self.assertTrue(math.isnan(wc.window_max_ratio(train, "x", 10_000, 1000)))

    def test_interval_sps_from_elapsed_time(self):
        train = pd.DataFrame({"step": [1000, 2000, 3000, 4000],
                              "elapsed_sec": [10.0, 20.0, 40.0, 50.0]})
        np.testing.assert_allclose(wc.interval_sps(train).to_numpy(), [100.0, 50.0, 100.0])

    @staticmethod
    def _train_with_rates(rates):
        elapsed = np.cumsum([1000.0 / r for r in rates])
        return pd.DataFrame({"step": 1000 * (np.arange(len(rates)) + 1), "elapsed_sec": elapsed})

    def test_throughput_drop_onset_detects_persistent_slowdown(self):
        train = self._train_with_rates([55.0] * 120 + [22.0] * 200)
        out = wc.throughput_drop_onset(train)
        self.assertIsNotNone(out["onset_step"])
        self.assertTrue(120_000 <= out["onset_step"] <= 150_000)
        self.assertAlmostEqual(out["median_sps_before"], 55.0, places=1)
        self.assertAlmostEqual(out["median_sps_after"], 22.0, places=1)

    def test_throughput_drop_onset_none_for_steady_run(self):
        out = wc.throughput_drop_onset(self._train_with_rates([55.0, 54.0, 56.0] * 100))
        self.assertIsNone(out["onset_step"])

    def test_linear_slope_per_step(self):
        self.assertAlmostEqual(wc.linear_slope([0, 10, 20], [1.0, 3.0, 5.0]), 0.2)


PER_RUN_COLUMNS_LEGACY = [
    "architecture", "seed", "run_id", "git_commit", "git_dirty", "hidden_width", "total_params",
    "final_mean_return_10ep", "final_within_seed_episode_sd", "periodic_mean_return_1M_5ep",
    "auc_mean_of_40_periodic", "steps_to_5000", "steps_to_8000", "steps_to_10000", "late_eval_sd_750k_1M",
    "max_drawdown_periodic", "low_return_outlier_evals_total", "low_return_outlier_evals_after_500k",
    "critic_loss_spikes_10x_median", "critic_loss_spike_steps", "critic_loss_max_ratio_to_median",
    "critic_loss_max", "critic_loss_median", "runtime_hours", "final_cumulative_sps"]
PAIR_KEYS = ["2×512 - 2×256", "2×1024 - 2×256", "2×1024 - 2×512"]


class PipelineOnRealData(unittest.TestCase):
    """Table/JSON/Markdown outputs of analysis/plot_width_comparison.py on the nine real runs.

    Figures are NOT rendered here (main(..., figures=False)); rendering and reproducibility are
    tested once, in short-lived subprocesses, by FiguresAndReproducibility."""

    @classmethod
    def setUpClass(cls):
        if not all(spec.data_dir.is_dir() for spec in wc.SPECS.values()):
            raise unittest.SkipTest("scientific data not present")
        import matplotlib
        matplotlib.use("Agg")
        import plot_width_comparison as pw
        cls.pw = pw
        cls.runs = wc.load_all()
        cls.tables = pw.build_tables(cls.runs)
        cls.out = tempfile.TemporaryDirectory()
        cls.out_dir = Path(cls.out.name)
        pw.main(cls.out_dir, figures=False)

    @classmethod
    def tearDownClass(cls):
        cls.out.cleanup()

    def raw_final(self, spec):
        """Final-evaluation mean computed straight from the CSV files, bypassing the library."""
        vals = []
        for seed in wc.SEEDS:
            folder = next(spec.data_dir.glob(f"{spec.run_prefix}_seed{seed}_*"))
            df = pd.read_csv(folder / "eval_metrics.csv")
            vals.append(float(df[df["kind"] == "final"]["mean_return"].iloc[0]))
        return vals

    def test_per_run_table_has_nine_rows_three_per_architecture(self):
        per_run = self.tables["per_run"]
        self.assertEqual(len(per_run), 9)
        self.assertEqual(per_run.groupby("architecture").size().to_dict(), {"2×1024": 3, "2×256": 3, "2×512": 3})
        self.assertEqual(sorted(per_run["seed"].tolist()), [0, 0, 0, 1, 1, 1, 2, 2, 2])

    def test_per_run_schema_keeps_legacy_columns_in_order(self):
        cols = list(self.tables["per_run"].columns)
        self.assertEqual(cols[:len(PER_RUN_COLUMNS_LEGACY)], PER_RUN_COLUMNS_LEGACY)
        self.assertEqual(len(cols), len(set(cols)))
        header = (self.out_dir / "width_comparison_per_run.csv").read_text(encoding="utf-8").splitlines()[0]
        self.assertEqual(header.split(","), cols)

    def test_final_statistics_match_independent_computation(self):
        for name, spec in wc.SPECS.items():
            vals = np.array(self.raw_final(spec))
            s = self.tables["stats"]["final_mean_return_10ep"][spec.label]
            self.assertAlmostEqual(s["mean"], vals.mean(), places=9)
            self.assertAlmostEqual(s["sd"], vals.std(ddof=1), places=9)
            self.assertAlmostEqual(s["min"], vals.min(), places=9)
            self.assertAlmostEqual(s["max"], vals.max(), places=9)
            self.assertAlmostEqual(s["range"], vals.max() - vals.min(), places=9)
            self.assertEqual(s["n"], 3)

    def test_original_two_width_statistics_are_unchanged(self):
        fin = self.tables["stats"]["final_mean_return_10ep"]
        self.assertAlmostEqual(fin["2×256"]["mean"], 11906.25, places=1)
        self.assertAlmostEqual(fin["2×256"]["sd"], 497.45, places=1)
        self.assertAlmostEqual(fin["2×512"]["mean"], 11645.37, places=1)
        self.assertAlmostEqual(fin["2×512"]["sd"], 482.26, places=1)
        eff = fin["pairwise_differences"]["2×512 - 2×256"]
        self.assertAlmostEqual(eff["mean_difference"], -260.9, places=1)
        self.assertAlmostEqual(eff["percent_difference"], -2.19, places=2)
        auc = self.tables["stats"]["auc_mean_of_40_periodic"]
        self.assertAlmostEqual(auc["2×256"]["mean"], 9911.5, places=1)
        self.assertAlmostEqual(auc["2×512"]["mean"], 9560.3, places=1)
        self.assertEqual(self.tables["stats"]["runtime_flagged_slow_(>1.5x_median_of_nine)"], ["2×256 seed 2"])

    def test_three_pairwise_differences_use_later_minus_earlier_width(self):
        pw = self.tables["stats"]["final_mean_return_10ep"]["pairwise_differences"]
        self.assertEqual(list(pw), PAIR_KEYS)
        m = {l: self.tables["stats"]["final_mean_return_10ep"][l]["mean"] for l in ("2×256", "2×512", "2×1024")}
        self.assertAlmostEqual(pw["2×512 - 2×256"]["mean_difference"], m["2×512"] - m["2×256"], places=9)
        self.assertAlmostEqual(pw["2×1024 - 2×256"]["mean_difference"], m["2×1024"] - m["2×256"], places=9)
        self.assertAlmostEqual(pw["2×1024 - 2×512"]["mean_difference"], m["2×1024"] - m["2×512"], places=9)
        self.assertLess(pw["2×1024 - 2×256"]["mean_difference"], 0)  # sanity: direction, not a claim

    def test_periodic_aggregate_schema_and_values(self):
        agg = pd.read_csv(self.out_dir / "width_comparison_periodic_aggregate.csv")
        self.assertEqual(len(agg), 40)
        expected = ["step"]
        for name in wc.SPECS:
            expected += [f"{name}_mean", f"{name}_sd"] + [f"{name}_seed{s}" for s in wc.SEEDS]
        self.assertEqual(list(agg.columns)[:len(expected)], expected)
        seeds = np.column_stack([agg[f"width_2x1024_seed{s}"] for s in wc.SEEDS])
        np.testing.assert_allclose(agg["width_2x1024_mean"], seeds.mean(axis=1))
        np.testing.assert_allclose(agg["width_2x1024_sd"], seeds.std(axis=1, ddof=1))  # sample SD across seeds

    def test_checkpoint_table_covers_six_checkpoints_for_three_architectures(self):
        ck = pd.read_csv(self.out_dir / "width_comparison_checkpoints.csv")
        self.assertEqual(sorted(ck["step"].unique()), [25_000, 100_000, 250_000, 500_000, 750_000, 1_000_000])
        self.assertEqual(len(ck), 18)
        self.assertEqual(set(ck["architecture"]), {"2×256", "2×512", "2×1024"})

    def test_stats_json_is_strict_and_has_stable_top_level_keys(self):
        text = (self.out_dir / "width_comparison_stats.json").read_text(encoding="utf-8")

        def reject(name):
            raise ValueError(name)
        stats = json.loads(text, parse_constant=reject)
        for key in ("seeds", "n_per_architecture", "sd_definition", "architectures", "pairs",
                    "final_mean_return_10ep", "auc_mean_of_40_periodic", "steps_to_threshold",
                    "runtime_hours_by_run", "between_seed_sd_of_periodic_return", "eval_drops"):
            self.assertIn(key, stats)
        self.assertEqual(stats["n_per_architecture"], 3)
        self.assertEqual(stats["pairs"], PAIR_KEYS)
        self.assertEqual(len(stats["runtime_hours_by_run"]), 9)
        self.assertIsNone(stats["steps_to_threshold"]["10000"]["2×1024"][0])  # seed 0 never reaches 10,000

    def test_every_architecture_reports_provenance_and_parameters(self):
        arch = self.tables["stats"]["architectures"]
        self.assertEqual(arch["2×1024"]["total_params"], 3_230_734)
        self.assertEqual(arch["2×1024"]["git_commit"], "e533a3c76ef069ffe37affc9a6a15ea2b1ae2e14")
        self.assertEqual(arch["2×512"]["git_commit"], "f25b3a0cb93da0a605b3dbac6254a866f83f664e")
        self.assertEqual(arch["2×256"]["git_commit"], "1a9f2405b5fdb9cffa379b72f569a0a3a01515e4")
        self.assertAlmostEqual(arch["2×1024"]["param_ratio_vs_2x256"], 3_230_734 / 217_870)

    def test_within_seed_episode_sd_is_kept_separate_from_between_seed_sd(self):
        per_run = self.tables["per_run"]
        fin = self.tables["stats"]["final_mean_return_10ep"]["2×1024"]
        within = per_run[per_run["architecture"] == "2×1024"]["final_within_seed_episode_sd"]
        self.assertTrue((within < fin["sd"]).all())  # episode SD (<=~0.2k) vs between-seed SD (~1.6k)
        self.assertNotAlmostEqual(float(within.mean()), fin["sd"], places=0)

    def test_tables_only_mode_writes_text_outputs_and_no_figures(self):
        names = {p.name for p in self.out_dir.iterdir()}
        self.assertEqual(names, {"width_comparison_per_run.csv", "width_comparison_periodic_aggregate.csv",
                                 "width_comparison_checkpoints.csv", "width_comparison_stats.json",
                                 "width_comparison_summary.md"})

    def test_schema_change_is_documented_in_the_json(self):
        schema = self.tables["stats"]["schema"]
        self.assertEqual(schema["version"], 2)
        self.assertIn("effect_width_vs_baseline", schema["removed_keys"])
        self.assertIn("pairwise_differences", schema["removed_keys"]["effect_width_vs_baseline"])

    def test_rank_correlation_is_labelled_exploratory_with_dependence_caveat(self):
        sp = self.tables["stats"]["eval_drops"]
        note = sp["spearman_note"].lower()
        for word in ("exploratory", "autocorrelat", "pooled", "not evidence of causation"):
            self.assertIn(word, note)
        text = (self.out_dir / "width_comparison_summary.md").read_text(encoding="utf-8").lower()
        self.assertIn("exploratory", text)
        self.assertIn("temporal dependence", text)

    def test_flagged_slow_run_has_throughput_drop_and_others_do_not(self):
        onset = self.tables["stats"]["throughput_drop_onset"]
        self.assertIsNotNone(onset["2×256 seed 2"]["onset_step"])
        self.assertTrue(all(v["onset_step"] is None for k, v in onset.items() if k != "2×256 seed 2"))

    def test_summary_mentions_all_architectures_and_is_numerically_consistent(self):
        text = (self.out_dir / "width_comparison_summary.md").read_text(encoding="utf-8")
        for token in ("2×256", "2×512", "2×1024", "e533a3c76ef069ffe37affc9a6a15ea2b1ae2e14",
                      "11,083.", "Reproduc"):
            self.assertIn(token, text)
        import re
        self.assertIsNone(re.search(r"\bnan\b", text, flags=re.IGNORECASE))  # no undefined values leak into prose


FIGURE_STEMS = ["01_learning_curves_comparison", "02_final_returns", "03_throughput_and_runtime",
                "04_critic_loss", "05_entropy_alpha", "06_mean_abs_q_and_critic_grad_norm",
                "07_return_difference_over_training", "08_variability_and_scaling"]
TEXT_OUTPUTS = ("width_comparison_per_run.csv", "width_comparison_periodic_aggregate.csv",
                "width_comparison_checkpoints.csv", "width_comparison_stats.json", "width_comparison_summary.md")


def run_pipeline_subprocess(out_dir: Path) -> None:
    """Full generation in a short-lived child process: matplotlib memory is returned to the OS on exit.

    Callers run these one after another, never concurrently."""
    import subprocess
    script = ANALYSIS / "plot_width_comparison.py"
    done = subprocess.run([sys.executable, str(script), str(out_dir)], capture_output=True, text=True, timeout=900)
    if done.returncode != 0:
        raise AssertionError(f"plot_width_comparison.py failed ({done.returncode}):\n{done.stdout}\n{done.stderr}")


class FiguresAndReproducibility(unittest.TestCase):
    """Figure rendering and the single two-generation reproducibility comparison (2 sequential subprocesses)."""

    @classmethod
    def setUpClass(cls):
        if not all(spec.data_dir.is_dir() for spec in wc.SPECS.values()):
            raise unittest.SkipTest("scientific data not present")
        cls.tmp = tempfile.TemporaryDirectory()
        cls.first = Path(cls.tmp.name) / "first"
        run_pipeline_subprocess(cls.first)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_all_eight_figures_exist_as_png_and_pdf(self):
        for stem in FIGURE_STEMS:
            png, pdf = self.first / f"{stem}.png", self.first / f"{stem}.pdf"
            self.assertEqual(png.read_bytes()[:4], bytes([0x89, 0x50, 0x4E, 0x47]), stem)  # PNG signature
            self.assertTrue(pdf.read_bytes().startswith(b"%PDF"), stem)
            self.assertGreater(png.stat().st_size, 20_000, stem)
        self.assertEqual({p.name for p in self.first.iterdir()},
                         {f"{s}.{e}" for s in FIGURE_STEMS for e in ("png", "pdf")} | set(TEXT_OUTPUTS))

    def test_text_outputs_do_not_depend_on_rendering_figures(self):
        import plot_width_comparison as pw
        with tempfile.TemporaryDirectory() as tables_only:
            pw.main(Path(tables_only), figures=False)
            for name in TEXT_OUTPUTS:
                self.assertEqual((self.first / name).read_bytes(), (Path(tables_only) / name).read_bytes(), name)

    def test_regeneration_is_byte_identical_and_inputs_are_untouched(self):
        import hashlib

        def digest():
            return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for spec in wc.SPECS.values() for p in sorted(spec.data_dir.rglob("*")) if p.is_file()}
        before = digest()
        second = Path(self.tmp.name) / "second"
        run_pipeline_subprocess(second)
        for path in sorted(self.first.iterdir()):
            if path.suffix in {".csv", ".json", ".md", ".png"}:  # PDFs may embed library-version metadata
                self.assertEqual(path.read_bytes(), (second / path.name).read_bytes(), path.name)
        self.assertEqual(before, digest())
        self.assertEqual(len(before), 54)  # 9 runs x 6 files, unchanged by the pipeline


if __name__ == "__main__":
    unittest.main()
