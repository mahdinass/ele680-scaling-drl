# ELE680 Experiment Infrastructure Policy

## 1. Purpose and status

This document defines the canonical infrastructure and operational rules for ELE680
experiments ("Scaling Up Networks in Deep Reinforcement Learning"). It is written so that
a team member, or a future version of us, can follow it without needing chat history.

- The infrastructure was **verified and frozen on 2026-10-08**, before scientific
  experiments began.
- Experiment comparability depends on following this policy.
- Infrastructure changes after the freeze must be explicitly reviewed and documented
  (see section 19).
- If a change can affect experiment results or comparability, affected experiments may
  need to be rerun.

**Freeze point**

- Repository commit: `0dabac2`
- 58/58 tests passed on Gorina4
- TensorBoard 2.18.0 verified
- Real HalfCheetah CLI and CUDA smoke paths verified
- 20k-step non-scientific runtime benchmark completed successfully
- Run artifacts and provenance verified
- tmux persistence across SSH disconnect verified

`0dabac2` is the **infrastructure freeze point**. Later commits may add experiment
architectures and configs, so it is not necessarily the commit every future experiment
must use. Every run records its exact commit (section 12).

## 2. Repository / development workflow

Canonical flow:

```text
LOCAL WINDOWS -> local Git -> GitHub -> clean-shell Git pull on Gorina4
              -> activate UiS uenv/Conda -> tests -> experiments
```

| Location | Path / URL |
|---|---|
| Local repository (Windows) | `C:\Users\mbnas\.vscode\ele680-scaling-drl` |
| Gorina4 repository | `/bhome/u273406/ele680-scaling-drl` |
| GitHub repository | https://github.com/mahdinass/ele680-scaling-drl |

Code and config changes move through Git. Do not manually maintain divergent source-code
copies between Windows and Gorina4.

## 3. Gorina4 clean-shell Git rule

> **Git operations on Gorina4 must be done from a fresh shell BEFORE running
> `uenv miniconda3-py312` or activating Conda.**

Reason: the UiS Miniconda environment can alter shared-library resolution and has already
caused:

```text
OpenSSL version mismatch. Built against 30000020, you have 30300010
```

Canonical order:

```bash
ssh ssh1.ux.uis.no
ssh gorina4

cd /bhome/u273406/ele680-scaling-drl
git pull
git status

uenv miniconda3-py312
conda activate /bhome/u273406/conda-envs/ele680
```

If Git fails after `uenv`/Conda has been loaded, open a fresh SSH shell rather than trying
to repair the active shell.

Do not run Python/GPU training workloads on `ssh1`.

## 4. Python environment

Canonical environment: `/bhome/u273406/conda-envs/ele680`

Activation:

```bash
uenv miniconda3-py312
conda activate /bhome/u273406/conda-envs/ele680
```

Verified major versions:

| Package | Version |
|---|---|
| Python | 3.12.x |
| PyTorch | 2.5.1+cu121 |
| PyTorch CUDA runtime | 12.1 |
| NumPy | 1.26.4 |
| Gymnasium | 1.1.1 |
| MuJoCo | 3.15.0 |
| TensorBoard | 2.18.0 |

Dependencies are installed from `requirements/gorina4.txt`
(which includes `requirements/base.txt`).

Do not replace the environment with an ad-hoc pip install or a different environment
during comparable experiments.

## 5. GPU policy

- Team: **Group 1**
- Assigned physical GPU: **GPU 0**
- Hardware verified: **Tesla P100-PCIE-12GB**

All project GPU runs must use `CUDA_VISIBLE_DEVICES=0`:

```bash
CUDA_VISIBLE_DEVICES=0 python ...
```

Do not silently use another Gorina4 GPU. If GPU allocation changes officially, document
the change before running further comparable experiments.

## 6. Long-running session policy

Scientific runs must be launched inside **tmux** on Gorina4. tmux persistence across SSH
disconnect was explicitly verified with a detached 120-second test.

```bash
tmux new -s ele680_<run-name>          # start a session
# Ctrl+B, then D                       # detach
tmux attach -t ele680_<run-name>       # reattach
```

Do not rely on an ordinary SSH foreground shell for multi-hour experiments.

## 7. Reproducibility / seed policy

Project scientific seeds: **0, 1, 2**.

The implementation seeds:

- Python `random`
- NumPy
- PyTorch CPU
- PyTorch CUDA
- environment initial reset
- action space
- replay-buffer RNG

Further rules:

- Training and evaluation use separate environment instances.
- Evaluation uses a separate deterministic seed stream: **training seed + 10,000**.
  Evaluation episode *i* resets with seed `(training seed + 10,000) + i`.
- Subsequent training episode resets are unseeded, so the environment continues its
  seeded RNG stream.
- GPU execution is seeded but **not promised to be bit-for-bit deterministic**, because
  `torch.use_deterministic_algorithms(True)` is intentionally not forced.

## 8. Environment interaction semantics

- Primary environment: **HalfCheetah-v5** (state observations, continuous actions).
- Use non-vectorized Gymnasium environments for the current baseline.
- Termination and truncation must remain separate.
- Both `terminated` and `truncated` end an episode and trigger a reset.
- **Only true termination cuts SAC bootstrapping.** Time-limit truncation must continue
  bootstrapping from the true next observation.
- The replay buffer stores observation, next observation, action, reward, termination and
  truncation **separately**. Do not replace this with one combined `done` flag.

## 9. SAC baseline implementation policy

Algorithm: Soft Actor-Critic (SAC). Reference behavior: CleanRL-style continuous-action
SAC, adapted transparently.

Baseline networks (kept separate from the SAC algorithm logic):

- Actor: 2×256
- Q1: 2×256
- Q2: 2×256

Locked baseline defaults (`configs/baseline.json`):

```text
env_id                   = HalfCheetah-v5
total_steps              = 1,000,000
buffer_size              = 1,000,000
batch_size               = 256
learning_starts          = 5,000
policy_frequency         = 2
target_network_frequency = 1
gamma                    = 0.99
tau                      = 0.005
policy_lr                = 3e-4
q_lr                     = 1e-3
automatic entropy tuning = enabled (target entropy = -action_dim)
```

Verified scheduling:

- `global_step` is 0-indexed.
- Random actions when `global_step < learning_starts`; policy actions afterward.
- Optimization begins only when `global_step > learning_starts`.
- One critic update per update step.
- When `global_step % policy_frequency == 0`, `policy_frequency` actor+alpha updates are
  performed on the same batch. This matches CleanRL's compensated delayed-policy
  schedule (long-run ratio: one actor update per environment step).
- Target networks are updated according to `target_network_frequency`.

Do not silently change these values during the baseline comparison.

## 10. Evaluation protocol

Evaluation must be deterministic (squashed-mean action), performed on a separate
environment, run under `no_grad`, and free of optimizer updates and replay-buffer
mutation.

Locked project protocol:

- **Periodic evaluation:** every 25,000 environment steps, 5 deterministic episodes.
- **Final evaluation:** 10 deterministic episodes.

Record each episode return, the mean, and the population standard deviation (ddof = 0).

Evaluation is not training.

## 11. Logging policy

Machine-readable CSV/JSON files are the **source of truth**. TensorBoard is a
convenience visualization layer only.

Each run has its own directory `results/<run_id>/` (never overwritten) with:

```text
config.json
metadata.json
status.json
train_metrics.csv
episode_metrics.csv
eval_metrics.csv
checkpoint.pt
tensorboard/
```

- `train_metrics.csv`: step, elapsed time, SPS, critic loss, actor loss, alpha loss,
  alpha, mean absolute Q, actor gradient norm, critic gradient norm. Actor columns are
  blank on rows where no actor update happened.
- `episode_metrics.csv`: step, elapsed time, episode return, episode length.
- `eval_metrics.csv`: step, evaluation kind (`periodic` / `final`), number of episodes,
  mean return, std return, per-episode returns.

Every logged `step` means **completed environment steps**.

## 12. Required provenance metadata

Every scientific run must record at least (in `metadata.json`):

- run ID; start/end timestamps; run status; failure type/message
- exact Git commit; Git dirty state
- Python, PyTorch, NumPy, Gymnasium, MuJoCo and TensorBoard versions
- CUDA runtime; CUDA availability; `CUDA_VISIBLE_DEVICES`; GPU name; hostname
- environment ID; observation/action shapes; seed; device; complete config
- Actor, Q1, Q2 and total parameter counts
- checkpoint path; wall-clock runtime; final evaluation

Git dirty state: **untracked files count as a dirty working tree.** Generated results are
ignored through `.gitignore` (`results/`) and therefore do not make the repository dirty.
A scientific run should normally begin from a clean Git working tree.

## 13. Failure handling

If a run fails:

- `status` becomes `failed`;
- the exact number of completed environment steps is recorded;
- the exception type and message are recorded;
- metrics already written are preserved;
- the exception is re-raised.

Errors must not be silently swallowed.

## 14. Checkpoint policy

A checkpoint contains: actor, Q1/Q2, target Q1/Q2, actor optimizer, critic optimizer,
alpha optimizer, `log_alpha`, the global (completed) step, the config and the seed.

Current checkpoints intentionally do **not** contain the replay buffer or RNG states.
Checkpoints therefore support inspection, evaluation, and recovery of model/optimizer
state, but are **not** guaranteed to give an exact continuation of an interrupted
trajectory.

Baseline config: periodic checkpoint every 100,000 steps, and a final checkpoint is always
saved. Checkpoint writes are atomic (temporary file + rename); one `checkpoint.pt` per run
holds the latest state.

## 15. stdout/stderr and process exit status

Every scientific long run must capture stdout and stderr to an external log **and**
preserve the real Python exit status even when `tee` is used.

Canonical pattern:

```bash
set -o pipefail
python ... 2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}
echo "$STATUS" > "<run>.exitcode"
```

The external log and exit-code file live under `/bhome/u273406/` and are separate from the
structured run artifacts.

Do not treat the presence of a log file as proof that a run succeeded. `status.json` and
the preserved Python exit status must also be checked.

## 16. Scientific runs vs infrastructure benchmarks

Infrastructure, smoke and benchmark runs must be clearly labelled and must **not** be used
as scientific experimental evidence.

Verified benchmark: **`BENCHMARK_20K_seed0`**, for runtime/infrastructure validation only.
It completed 20,000 steps with exit status 0 in 306.694 s.

Measured steady-state baseline throughput: approximately **56-59 environment steps/s**
after warm-up.

Projected baseline runtime:

- 500k steps: about 2.4-2.5 h
- 1M steps: about 4.7-5.0 h, plus small evaluation/checkpoint overhead

The scientific baseline budget is therefore locked to **1,000,000 steps**.

The benchmark's observed return must not be used in final scientific comparisons.

## 17. Official baseline launch template

The currently approved seed-0 launch template. It is documented here and has **not** been
executed as part of this policy.

> **DO NOT copy this blindly for later seeds: change both the run ID and the seed.**

In a fresh shell, before loading Conda, verify the Git commit and status
(`git log -1 --oneline`, `git status`). Then:

```bash
tmux new -s ele680_baseline_seed0
```

Inside tmux:

```bash
uenv miniconda3-py312
conda activate /bhome/u273406/conda-envs/ele680
cd /bhome/u273406/ele680-scaling-drl

export CUDA_VISIBLE_DEVICES=0

RUN_ID="BASELINE_1M_seed0_20261008"
LOG="/bhome/u273406/${RUN_ID}.log"

set -o pipefail

python -m src.train \
  --config configs/baseline.json \
  --seed 0 \
  --device cuda \
  --run-id "$RUN_ID" \
  2>&1 | tee "$LOG"

STATUS=${PIPESTATUS[0]}

echo "=== TRAIN EXIT STATUS ===" | tee -a "$LOG"
echo "$STATUS" | tee -a "$LOG"
echo "$STATUS" > "/bhome/u273406/${RUN_ID}.exitcode"
```

## 18. Monitoring

Approved monitoring commands:

```bash
tmux attach -t <session>
tail -f /bhome/u273406/<run-id>.log
cat results/<run-id>/status.json
watch -n 5 nvidia-smi -i 0
```

Monitoring must not modify the running experiment.

## 19. Change-control policy

After the first scientific baseline run begins, infrastructure must not drift silently.

Changes to any of the following require explicit review and documentation:

- dependencies
- environment version
- SAC update semantics
- termination/truncation handling
- seed behavior
- evaluation procedure
- logging semantics
- checkpoint semantics
- GPU assignment
- run budget
- baseline hyperparameters
- architecture construction
- normalization behavior
- launch procedure

If a change can alter results, comparable runs performed before and after the change must
not simply be pooled without justification.

Record meaningful infrastructure/protocol changes in:

1. Git
2. this policy document, where applicable
3. the Notion Decisions / Implementation / Compute-Reproducibility documentation

## 20. Verification record

| Item | Result |
|---|---|
| Baseline networks | verified (7-test stage) |
| SAC / replay primitives | 21/21 tests |
| Train / eval skeleton | 35/35 tests |
| Final infrastructure | 58/58 tests (Gorina4, commit `0dabac2`) |
| TensorBoard 2.18.0 | import verified |
| Real HalfCheetah smoke | verified |
| CUDA smoke | verified |
| Logging, checkpoints, failure handling | verified |
| `BENCHMARK_20K_seed0` | completed successfully |
| Benchmark artifacts / provenance | inspected |
| tmux disconnect persistence | verified |

The infrastructure is ready for the first scientific baseline experiment.
**The scientific baseline itself had NOT yet been launched at the time this policy was
frozen.**

## 21. InputNorm-only extension (implementation candidate, 16 October 2026)

The reviewed design adds optional observation normalization to 8x256 SAC. See
`INPUTNORM_ABLATION.md` for its exact estimator, inference/checkpoint contract and
manual validation/launch gates. This is a deliberate controlled intervention:
the default disabled path preserves legacy learning and network state dictionaries.
Enabled checkpoints additionally save versioned running moments; evaluation uses
independent frozen snapshots. Resolved config gains `input_norm` and enabled-run
metadata gains a normalization summary. Existing CSV schemas, schedules, dependencies,
GPU assignment and historical artifacts remain unchanged. Exact training resumption
is still unsupported. No scientific launch is authorized by this documentation;
local CPU tests do not establish server CUDA readiness.

## 22. Residual-only extension (code-review candidate, 18 October 2026)

The independent residual-only configuration disables InputNorm and preserves the
eight hidden Linear layers, initializers and all 1402126 actor/online-critic
parameters. Seven post-addition ReLU identity shortcuts are the only scientific
intervention. `residual_connections` defaults to false; checkpoint config records
it and save/load reject semantic disagreement across agent and all five networks
before restoration. Missing historical flags mean false. Network weight keys alone
cannot identify residual semantics. Existing InputNorm implementation/source and
completed scientific artifacts remain frozen.

See `RESIDUAL_ABLATION.md` for the manual, exact-commit CPU/CUDA tests, separate
6200/26000 preflights, monitored launch and artifact audits. Residual receipts live
in `results/residual_gates/<SHA>/`; InputNorm receipts cannot satisfy these gates.
No server package, GPU allocation, SAC/replay/evaluation/logging schedule or
scientific budget changes are introduced. No server access or scientific training
is authorized by this code-review candidate; independent code review precedes any
user-operated Gorina4 validation. Exact training resumption remains unsupported.
