"""On-disk run record: JSON files, flushed CSVs (source of truth) and TensorBoard.

Run directory layout (output_dir/run_id/):
  config.json          full TrainConfig (with the resolved run_id)
  metadata.json        provenance + status + timing + parameter counts + final eval
  status.json          small quick-look subset of metadata.json
  train_metrics.csv    periodic losses / diagnostics
  episode_metrics.csv  one row per finished training episode
  eval_metrics.csv     one row per evaluation (periodic and final)
  checkpoint.pt        latest checkpoint (written by train.py)
  tensorboard/         TensorBoard event files (convenience only)
"""

import csv
import json
import os
import time
import uuid
import warnings
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.evaluate import EvalResult
from src.run_metadata import collect_environment_info

TRAIN_FIELDS = [
    "step", "elapsed_sec", "sps", "critic_loss", "actor_loss", "alpha_loss",
    "alpha", "q_mean_abs", "actor_grad_norm", "critic_grad_norm",
]  # fmt: skip
EPISODE_FIELDS = ["step", "elapsed_sec", "episode_return", "episode_length"]
EVAL_FIELDS = ["step", "kind", "num_episodes", "mean_return", "std_return", "returns"]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def make_run_id(env_id: str, seed: int) -> str:
    """Unique id: <env>_seed<seed>_<UTC time>_<4 random hex> (os.urandom, so it never
    touches the seeded RNG streams)."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{env_id}_seed{seed}_{stamp}_{uuid.uuid4().hex[:4]}"


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


class _Csv:
    """CSV file that writes the header once and flushes after every row."""

    def __init__(self, path: Path, fields: list[str]) -> None:
        self.file = open(path, "w", newline="", encoding="utf-8")
        self.writer = csv.DictWriter(self.file, fieldnames=fields)
        self.writer.writeheader()
        self.file.flush()

    def write(self, row: dict) -> None:
        self.writer.writerow(row)
        self.file.flush()

    def close(self) -> None:
        self.file.close()


class RunLogger:
    def __init__(self, config, verbose: bool = False) -> None:
        self.verbose = verbose
        self.run_id = config.run_id or make_run_id(config.env_id, config.seed)
        self.run_dir = Path(config.output_dir) / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=False)  # never overwrite a previous run
        self.checkpoint_path = self.run_dir / "checkpoint.pt"
        self.t0 = time.time()

        config_dict = asdict(config)
        config_dict["run_id"] = self.run_id
        _write_json(self.run_dir / "config.json", config_dict)

        self.metadata = {
            "run_id": self.run_id,
            "status": "running",
            "start_time": _now(),
            "end_time": None,
            "runtime_seconds": None,
            "steps_completed": 0,
            "failure_type": None,
            "failure_message": None,
            "env_id": config.env_id,
            "seed": config.seed,
            "device": config.device,
            "run_dir": str(self.run_dir),
            "checkpoint_path": str(self.checkpoint_path),
            "config": config_dict,
            "parameter_counts": None,  # filled by train() once the agent exists
            "final_eval": None,
            **collect_environment_info(config.device),
        }

        self._train_csv = _Csv(self.run_dir / "train_metrics.csv", TRAIN_FIELDS)
        self._episode_csv = _Csv(self.run_dir / "episode_metrics.csv", EPISODE_FIELDS)
        self._eval_csv = _Csv(self.run_dir / "eval_metrics.csv", EVAL_FIELDS)

        self.tb = None
        if config.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.tb = SummaryWriter(log_dir=str(self.run_dir / "tensorboard"))
            except Exception as exc:  # TensorBoard is a convenience; never kill a run for it
                warnings.warn(f"TensorBoard disabled: {exc!r}")
        self.metadata["tensorboard_enabled"] = self.tb is not None
        self._write_status_files()

    # ---- metadata / status -------------------------------------------------
    def _write_status_files(self) -> None:
        _write_json(self.run_dir / "metadata.json", self.metadata)
        keys = [
            "run_id", "status", "start_time", "end_time", "runtime_seconds", "steps_completed",
            "failure_type", "failure_message", "checkpoint_path",
        ]  # fmt: skip
        _write_json(self.run_dir / "status.json", {k: self.metadata[k] for k in keys})

    def note_step(self, step: int) -> None:
        """Record the latest completed env step in memory only (no file write).
        finish() persists it, so a failed run reports the exact progress."""
        self.metadata["steps_completed"] = step

    def update_metadata(self, **fields) -> None:
        self.metadata.update(fields)
        self._write_status_files()

    def finish(self, status: str, exc: BaseException | None = None) -> None:
        self.metadata.update(
            status=status,
            end_time=_now(),
            runtime_seconds=round(time.time() - self.t0, 3),
            failure_type=type(exc).__name__ if exc is not None else None,
            failure_message=str(exc) if exc is not None else None,
        )
        self._write_status_files()

    # ---- metrics -----------------------------------------------------------
    def _elapsed(self) -> float:
        return time.time() - self.t0

    def _tb(self, tag: str, value, step: int) -> None:
        if self.tb is not None and value is not None:
            self.tb.add_scalar(tag, value, step)

    def log_train(self, step: int, metrics: dict) -> None:
        """`metrics` may omit keys (written blank); known keys are TRAIN_FIELDS[3:]."""
        elapsed = self._elapsed()
        row = {"step": step, "elapsed_sec": round(elapsed, 3), "sps": round(step / elapsed, 2) if elapsed > 0 else None}
        row.update(metrics)
        self._train_csv.write(row)
        if self.verbose:
            print(f"step={step} sps={row['sps']} critic_loss={metrics.get('critic_loss')} alpha={metrics.get('alpha')}", flush=True)
        for key, tag in [
            ("critic_loss", "losses/critic_loss"), ("actor_loss", "losses/actor_loss"),
            ("alpha_loss", "losses/alpha_loss"), ("alpha", "losses/alpha"),
            ("q_mean_abs", "diagnostics/q_mean_abs"), ("actor_grad_norm", "diagnostics/actor_grad_norm"),
            ("critic_grad_norm", "diagnostics/critic_grad_norm"), ("sps", "charts/SPS"),
        ]:  # fmt: skip
            self._tb(tag, row.get(key), step)

    def log_episode(self, step: int, episode_return: float, episode_length: int) -> None:
        self._episode_csv.write(
            {
                "step": step,
                "elapsed_sec": round(self._elapsed(), 3),
                "episode_return": episode_return,
                "episode_length": episode_length,
            }
        )
        self._tb("charts/episodic_return", episode_return, step)

    def log_eval(self, step: int, result: EvalResult, kind: str = "periodic") -> None:
        """kind is "periodic" or "final". std_return is the population std (ddof=0)."""
        std = float(np.std(result.returns))
        self._eval_csv.write(
            {
                "step": step,
                "kind": kind,
                "num_episodes": len(result.returns),
                "mean_return": result.mean_return,
                "std_return": std,
                "returns": json.dumps(result.returns),
            }
        )
        if self.verbose:
            print(f"step={step} eval[{kind}] mean_return={result.mean_return:.2f} std={std:.2f}", flush=True)
        self._tb("eval/mean_return" if kind == "periodic" else "eval/final_mean_return", result.mean_return, step)
        if kind == "final":
            self.update_metadata(
                final_eval={
                    "step": step,
                    "num_episodes": len(result.returns),
                    "mean_return": result.mean_return,
                    "std_return": std,
                    "returns": result.returns,
                }
            )

    def close(self) -> None:
        for f in (self._train_csv, self._episode_csv, self._eval_csv):
            f.close()
        if self.tb is not None:
            self.tb.close()
