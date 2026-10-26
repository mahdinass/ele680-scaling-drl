# Residual connections ONLY: SAC 8x256

Status: **CODE-REVIEW CANDIDATE ONLY**, 18 October 2026. No Gorina4 access,
server validation, preflight training or scientific residual run has been performed
by this implementation task. The user must independently review the code before
authorizing server validation, and review server evidence before scientific runs.

## Question and frozen references

Does adding identity skips alone change final return, sample efficiency or
between-seed variability of the naive 8x256 model under the fixed SAC protocol?
Improvement is a hypothesis. Keep naive seed 2 and all future residual seeds.

- Scientific naive control: `6c674babc344e62566692e1a4fcf3611dc53a6cc`,
  `configs/depth_8x256.json`.
- Starting source: `689dbee26af0ab44957bd17dafc7435d8f4a6109`,
  `codex/inputnorm-8x256`, PR #1 (preserved, unmerged).
- New branch: `codex/residual-8x256`, stacked PR targets
  `codex/inputnorm-8x256`. Dependency on PR #1 is intentional; normalization code
  is inherited but disabled for this experiment. No automatic merge or rebase.
- The reviewed full residual PR-head SHA in the implementation handoff is the
  validation candidate. Do not substitute a moving ref. Changing the SHA requires
  new gates; never validate a future merge SHA using old receipts.

## Exact architecture and parameter accounting

Actor input is the 17-dimensional state. Critic input is the concatenation of
17 state coordinates and six actions. For each actor, online critic and target:

```python
x = relu(fc1(input))
for layer in [fc2, *hidden_extra]:  # seven existing 256 -> 256 Linear layers
    x = relu(layer(x) + x) if residual_connections else relu(layer(x))
# existing actor mean/log_std heads or scalar Q head
```

There are exactly eight hidden Linear transformations, eight hidden ReLUs and
seven parameter-free identity additions. The addition occurs before the existing
ReLU; no shortcut surrounds fc1 or an output head. No new embedding, Linear,
projection skip, activation, normalization, dropout, trainable scale or changed
initializer is added. PyTorch Linear initialization and construction order are
unchanged in both modes. This is a single-layer residual formulation, not a
two-layer residual block or a full SimBa architecture.

| Network | Trainable parameters |
|---|---:|
| Actor | 468236 |
| Q1 | 466945 |
| Q2 | 466945 |
| Actor + online Q1/Q2 | **1402126** |

Targets are independent copies and excluded from this total. Existing action
scale/bias buffers are unchanged. Parameter count and hidden depth are matched
to the naive control; the function and gradient paths intentionally change.

`residual_connections: bool = False` is appended after all historical arguments
in Actor/QNetwork/SACAgent/TrainConfig. Non-bools (including 0/1 and strings) are
rejected. `configs/ablation_residual_8x256.json` adds only
`residual_connections: true` and explicit `input_norm: false` to the control JSON.
LayerNorm is absent. General flags remain independently configurable for future
work, but the residual-only gate rejects any normalized/mixed experiment.

## Frozen learning protocol and compatibility

HalfCheetah-v5, seeds 0/1/2, 1M environment steps, replay capacity 1M, batch size 256,
5000 random actions, gamma 0.99, tau 0.005, policy Adam LR 3e-4 and critic/alpha
Adam LR 1e-3 remain fixed. Critic updates start at `global_step > 5000`; actor and
alpha use the compensated frequency 2 schedule; target frequency 1 is unchanged.
Only termination masks bootstrapping. Raw observations/actions/rewards and
separate truncations are stored in replay; residual forwards do not transform them.
Learned policy actions may change because architecture is the intended variable.

Evaluation remains deterministic on a separate environment with seed+10000:
five episodes every 25k and a separate ten-episode final evaluation. It consumes
no policy-sampling RNG and updates no optimizer/replay. Checkpoints remain atomic,
every 100k plus final; CSV/logging cadence and schemas are unchanged. Report
mean +/- between-training-seed sample SD (ddof=1, n=3); within-evaluation episode
SD uses ddof=0 and is a different quantity. Preflight returns are never science.

Checkpoint config always saves the residual flag. Load checks its strict boolean
value against the receiving agent AND actor, Q1/Q2 and both targets before loading
any network, optimizer, alpha or normalization state. Missing historical flags
mean false; false/true mismatches fail even when all weight shapes/keys match.
Save similarly rejects config/agent/network disagreement. Bare state_dict loading
cannot protect semantic compatibility: retain full checkpoint config for inference.
The inherited InputNorm checkpoint contract is preserved. Exact continuation is
still unsupported because replay, RNG and environment state are not saved.

Disabled-mode tests load the actual networks, SAC and training-loop source from
the frozen naive Git commit, rather than reimplementing a reference. They compare
initialization, forward/sampled actions, weight keys/values, RNG, losses, optimizers,
target updates, raw replay, evaluations and short trajectories at depths 2 and 8.

## Attribution and interpretation

The exact wiring follows the user-approved ELE680 controlled-ablation protocol.
Conceptual identity-shortcut attribution: He, Zhang, Ren and Sun, *Deep Residual
Learning for Image Recognition*, CVPR 2016, https://arxiv.org/abs/1512.03385.
No external residual repository code is copied. Existing CleanRL continuous SAC
attribution is retained. The inherited normalization code has separate SimBa
attribution and is disabled here. This experiment does not test full ResNet/SimBa.
Small n, one environment and shared server load limit interpretation; do not claim
significance, a mechanism or general RL benefit from local implementation tests.

## Manual Gorina4 procedure, ONLY after independent code-review authorization

The assistant does not connect or execute any of these remote commands. The user
operates the terminal, one operation at a time. Preserve existing results, PRs and
the frozen Conda environment; do not install packages or run on ssh1. Run one
training job at a time, using assigned physical GPU 0 Tesla P100-PCIE-12GB.

### 1. Fresh Git shell before uenv/Conda

On Gorina4, inspect the current checkout and choose a safe time with no jobs:

```bash
cd /bhome/u273406/ele680-scaling-drl
test -z "${CONDA_PREFIX:-}" || { echo 'Open a fresh shell'; exit 1; }
test -z "$(git status --porcelain)" || { echo 'Dirty checkout; inspect, do not discard'; exit 1; }
EXPECTED_SHA=PASTE_INDEPENDENTLY_REVIEWED_FULL_RESIDUAL_SHA
git fetch origin codex/residual-8x256
git switch --detach "$EXPECTED_SHA"
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA" && test -z "$(git status --porcelain)"
git log -1 --format=fuller
```

Stop on any failure. Inspect the exact source/diff before the next operation.

### 2. Existing environment and GPU inspection

```bash
uenv miniconda3-py312
conda activate /bhome/u273406/conda-envs/ele680
PY=/bhome/u273406/conda-envs/ele680/bin/python
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PY" -c 'import sys,torch; print(sys.executable,torch.__version__,torch.cuda.is_available())'
nvidia-smi -i 0
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
```

Require frozen Python 3.12, Torch 2.5.1+cu121/CUDA 12.1, NumPy 1.26.4,
Gymnasium 1.1.1, MuJoCo 3.15.0 and TensorBoard 2.18.0. Confirm physical GPU 0 is
idle. The helper checks UUID/processes and utilization twice around launch;
a cooperative residual lock cannot reserve the device against other users.

### 3. Exact-commit CPU and CUDA suites (separate operations)

```bash
CUDA_VISIBLE_DEVICES='' "$PY" scripts/residual_gorina4.py tests --sha "$EXPECTED_SHA" --mode cpu
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/residual_gorina4.py tests --sha "$EXPECTED_SHA" --mode cuda
```

CUDA receipts require zero skips. Failed reruns invalidate old successes.
Receipts use `results/residual_gates/<SHA>/` plus a residual experiment identity;
InputNorm receipts cannot satisfy them. Version, source, config and idle GPU
checks fail closed. Local CPU evidence is not a server CUDA gate receipt.

### 4. Two distinct NON-SCIENTIFIC preflights, sequentially

Create tmux from a regular shell; inside it activate the existing environment as
above, set the exact reviewed EXPECTED_SHA and OMP/MKL settings. Run 6,200 first:

```bash
tmux new -s ele680_residual_pre6200 -c /bhome/u273406/ele680-scaling-drl
# Inside tmux, after environment setup:
bash scripts/run_residual_gorina4.sh "$EXPECTED_SHA" preflight6200 0
```

Capture its printed run ID and inspect the log, exit code, telemetry and audit.
Independently audit that exact ID:

```bash
"$PY" scripts/residual_gorina4.py audit --sha "$EXPECTED_SHA" --run-id EXACT_6200_RUN_ID --steps 6200 --seed 0
```

After review, create a separate tmux session for 26,000 and repeat setup:

```bash
tmux new -s ele680_residual_pre26000 -c /bhome/u273406/ele680-scaling-drl
# Inside tmux, after environment setup:
bash scripts/run_residual_gorina4.sh "$EXPECTED_SHA" preflight26000 0
"$PY" scripts/residual_gorina4.py audit --sha "$EXPECTED_SHA" --run-id EXACT_26000_RUN_ID --steps 26000 --seed 0
```

The 26,000 runner re-audits 6,200. Audit requires no normalization statistics,
residual-enabled checkpoint/config semantics, actual 8x256 topology/counts,
finite reloaded actor/all critics, complete unchanged CSV schemas/schedules,
zero Python exit, completion log, frozen provenance and numeric P100 telemetry.
6,200 has one final 10-episode evaluation; 26,000 also has periodic 5 at 25,000.
No return threshold is used. Review timing/SPS/memory and all artifacts.
Return evidence to the reviewer and STOP before any scientific authorization.

### 5. Scientific workflow, separately authorized only after both preflights

Prepared for future use, not authorized by this document. In separate tmux
sessions on the same exact reviewed SHA, run scientific seed 0, independently audit and
review it, then scientific seed 1, audit/review, then scientific seed 2. The runner re-audits
both preflights and the previous science seed from raw artifacts before launch.

```bash
bash scripts/run_residual_gorina4.sh "$EXPECTED_SHA" scientific 0
"$PY" scripts/residual_gorina4.py audit --sha "$EXPECTED_SHA" --run-id EXACT_SCIENCE_RUN_ID --steps 1000000 --seed 0
```

Change seed and run ID appropriately only after the preceding run is reviewed.
Each 1M audit expects 994 finite train rows, 1,000 training episodes, 40 periodic
evaluations and one final 10-episode evaluation, a finite reloadable checkpoint,
no normalization artifacts and matching residual semantics. Raw CSV/JSON are the
source of truth. Logs, true Python exit and sampled GPU telemetry are external
under `/bhome/u273406/`; the runner also requires successful tee and a monitor
still alive at training completion. Any failure stops the workflow. Archive new
artifacts and aligned 25k curves before reporting three-seed statistics.
