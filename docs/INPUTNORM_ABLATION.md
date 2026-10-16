# InputNorm-only SAC, 8 x 256

Status: implementation candidate; no scientific InputNorm result exists. Local CPU
verification is recorded in `INPUTNORM_LOCAL_VALIDATION.md`. CUDA/server validation
is outstanding. Maximum readiness before server evidence: READY FOR GORINA4 PREFLIGHT.

## Scientific question and fixed control

Does online observation standardization change return or between-seed variability
of the naive 8 x 256 SAC network under the fixed HalfCheetah-v5 protocol? A benefit
is a hypothesis, not an expected result. The weak naive seed 2 does not establish
an observation-scale mechanism. Keep all seeds and report uncertainty.

Control: `6c674babc344e62566692e1a4fcf3611dc53a6cc`, `configs/depth_8x256.json`.
`configs/ablation_inputnorm_8x256.json` is the same JSON object with only
`input_norm: true` added. Existing configs/data/checkpoints are untouched.

The actor and both critics have eight 256-wide hidden Linear+ReLU layers: fc1,
fc2, and six hidden_extra layers. No extra embedding, residual, LayerNorm,
activation, trainable scale/bias, or changed initializer is added. Parameter counts
remain actor 468236, Q1 466945, Q2 466945; total 1402126 excluding targets.

SAC losses, Adam settings, alpha tuning, compensated delayed actor updates,
target Polyak updates, replay sampling, raw rewards, termination/truncation masks,
environment/time limit and seed streams are preserved. Three seeds 0/1/2, 1M steps,
5k random actions, periodic 5-episode deterministic evaluation every 25k, separate
final 10-episode evaluation, same GPU0 policy, logging cadence and CSV schemas.

## Estimator and exact timing

All 17 state coordinates are standardized independently (8 qpos, 9 qvel).
No action/reward/output normalization. Convert the current raw observation to
float32 (matching replay precision), then accumulate Welford moments in float64:

```
n += 1
delta = x - mean
mean += delta / n
M2 += delta * (x - mean)
variance = M2 / n                      # population, not sample variance
z = (x - mean) / sqrt(max(variance, 0) + 1e-8)
```

The float32 device cache stores mean and inverse standard deviation. Cached
reciprocal multiplication implements the displayed division. Empty state is the
identity; first incorporated observation maps to zero. Epsilon gives a minimum
denominator of 1e-4. No clipping, pseudocount, EMA, or reward normalization.
Nonfinite observations/results and incompatible restored moments fail explicitly.

Call update exactly ONCE at the start of every training iteration, before its
action, including random warm-up. Initial and subsequent reset decision states
count. Interior next observations count at their next decision. Terminal/time-limit
endpoints and the final next observation are not counted separately; they remain
available for replay/bootstrapping. Thus first policy action (global_step=5000)
sees count 5001; after T completed iterations count is T, including T=1000000.
Do not count both obs and next_obs or update from replay samples. Cumulative
decision-state statistics avoid dependence on replay reuse/update frequencies;
they retain warm-up influence and do not remove temporal correlation.

## Interfaces and isolation

One `SACAgent.observation_normalizer` supplies the transform for collection, actor
loss, online critics, next-state actor/target critics, fresh alpha sampling and
diagnostics. It stays fixed across an iteration's optimizer updates. Targets do
not have separate or Polyak-averaged statistics. Replay stores raw observations
and true next observations; transforms allocate outputs without mutating batches.

Low-level Actor/QNetwork still accept network-ready tensors. For raw inference:

```python
agent = SACAgent(obs_space, action_space, hidden_depth=8, input_norm=True)
load_checkpoint(path, agent)
with torch.no_grad():
    z = agent.normalize_observation(raw_float32_tensor)
    action = agent.actor.get_deterministic_action(z)
```

Never pass z through normalize_observation a second time. Bare actor weights do
not describe an enabled policy's raw-input behavior without its normalizer.
Evaluation receives `normalizer.snapshot()`: independent cached tensors with a
pure transform and NO update method. Neither forwards nor no_grad implicitly
update statistics. Tests compare count, mean, M2, cached mean and cached inverse
standard deviation before/after BOTH periodic and final evaluation, including the
actual training call sites and 5-/10-episode settings.

## Checkpoints, metadata and compatibility

Enabled checkpoints add a top-level `observation_normalization` entry (version,
convention, dimension, epsilon, clipping=None, integer count, float64 mean/M2
tensors). Caches are deterministically rebuilt on the requested device. Counts
must equal completed steps. Config/agent mismatch, missing enabled statistics,
corrupt moments or architecture mismatch are rejected before model restoration.
Only tensors/plain containers are saved; weights_only=True remains supported.
Atomic replacement and checkpoint cadence remain unchanged.

An additive metadata summary records estimator specification and final moments;
existing CSV fields/meaning are unchanged. Metadata count starts at zero and is
updated at successful completion; intermediate checkpoint is authoritative for
intermediate moments. No replay/RNG/environment state is saved: exact training
resume remains unsupported. No resume launcher was introduced.

Disabled is the default for missing historical flags. No estimator is constructed,
the transform returns the original tensor, network state dictionaries/initialization
and sampling RNG consumption are unchanged. Historical checkpoints load strictly;
disabled weights remain loadable into the frozen networks. Resolved config
snapshots gain the explicit default flag. Enabled files must use the new loader;
old loaders cannot enforce the normalization input contract.

## Verification and review checklist

- [ ] Review all changes against control; old configs, networks, replay and data untouched.
- [ ] Full real-dependency local suite and analysis suite; classify CUDA as BLOCKED locally.
- [ ] Welford numerical reference, shapes, zero variance, finite guards and restore validation.
- [ ] Explicit first-policy count 5001; million-update estimator count 1000000.
- [ ] Raw replay, actions unchanged; hooks verify all actor/Q/target/alpha/diagnostic paths.
- [ ] Enabled/disabled topology and parameter counts; disabled bitwise CPU trajectory,
      optimizer states and RNG against actual frozen SAC AND training-loop source.
- [ ] Frozen evaluator snapshots; both evaluation kinds leave every training statistic unchanged.
- [ ] Enabled inference roundtrip, legacy reload and rejected semantic mismatches.
- [ ] Server CPU/CUDA tests, exact versions/GPU, separate 6200/26000 preflights and audits.
- [ ] Freeze exact reviewed full Git SHA, clean source; new SHA invalidates old gate receipts.

Local evidence is in the validation record. These unchecked items are the reusable
review checklist, not claims that local checks are still unexecuted. Remaining
scientific limits: n=3, one task, fixed hyperparameters, changing input representation,
shared GPU load, historical control versus new implementation commit. Matching
seed numbers does not establish independent paired statistical observations.

## Attribution

Lee et al., *SimBa: Simplicity Bias for Scaling Up Parameters in Deep Reinforcement
Learning*, ICLR 2025, Section 4 equations 3-4:
https://arxiv.org/abs/2410.09754. Welford accumulation is independently implemented;
no external repository code is copied. This treatment is observation normalization
on the existing plain MLP, not full SimBa. Existing SAC attribution remains in
`src/networks.py` and `src/sac_update.py` (CleanRL continuous SAC).

## Manual Gorina4 runbook — do not skip gates

The agent does not execute these commands. Never install/update server packages.
Set EXPECTED_SHA to the exact full PR-head SHA returned in the implementation
handoff, not a moving branch name or `git rev-parse HEAD`. That SHA is a validation
candidate; freeze it for science only after approval. If merging creates a different
SHA, validate that new clean SHA again. Do not run scientific training before the
user supplies server preflight evidence and it has been reviewed.

### 1. Fresh shell, before uenv/Conda

```bash
cd /bhome/u273406/ele680-scaling-drl
test -z "${CONDA_PREFIX:-}" || { echo 'Open a fresh shell'; exit 1; }
test -z "$(git status --porcelain)" || { echo 'Dirty checkout; stop'; exit 1; }
EXPECTED_SHA=PASTE_REVIEWED_FULL_PR_HEAD_SHA
git fetch origin codex/inputnorm-8x256
git switch --detach "$EXPECTED_SHA"
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA" && test -z "$(git status --porcelain)"
git log -1 --format=fuller
```

Do not reset/discard server changes. Inspect the output and exact SHA before the
next block. Detached checkout permits PR validation without merging main.

### 2. Load the EXISTING environment; inspect physical GPU0

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

Confirm Tesla P100-PCIE-12GB, correct physical UUID and sustained idleness. Gate
commands reject an occupied GPU at launch; a team lock cannot reserve it against
other users, so retain telemetry and do not assume future load is controlled.

### 3. CPU and CUDA suites, with exact-commit gate receipts

Run blocks separately and stop on any nonzero exit. A failed rerun must not be
treated as passed based on an older receipt. CUDA skips are rejected.

```bash
CUDA_VISIBLE_DEVICES='' "$PY" scripts/inputnorm_gorina4.py tests --sha "$EXPECTED_SHA" --mode cpu
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/inputnorm_gorina4.py tests --sha "$EXPECTED_SHA" --mode cuda
```

The helper enforces canonical Python, frozen core versions, clean source and config.
CPU CUDA skips are expected; the CUDA suite must run all tests without skips.
Test execution is not scientific evidence. Gate records are under ignored
`results/inputnorm_gates/<SHA>/`.

### 4. NON-SCIENTIFIC 6200 and 26000 CUDA/HalfCheetah preflights

```bash
tmux new -s ele680_inputnorm_preflight
```

Inside tmux, activate the existing environment once if not inherited; set the
same literal EXPECTED_SHA again if needed. Run one command, inspect its completed
audit, then run the next:

```bash
bash scripts/run_inputnorm_gorina4.sh "$EXPECTED_SHA" preflight6200 0
bash scripts/run_inputnorm_gorina4.sh "$EXPECTED_SHA" preflight26000 0
```

Names are `INPUTNORM_8x256_<kind>_seed<N>_<YYYYMMDDTHHMMSSZ>` in UTC.
The runner refuses wrong/dirty source, config drift, missing test gates, wrong
environment/GPU, occupied device, missing prerequisite audits, duplicate artifacts
or missing tmux. It captures external stdout/stderr `.log`, physical-GPU telemetry
`.gpu.csv`, and the actual Python PIPESTATUS[0] in `.exitcode`, all under
`/bhome/u273406/`. A failed tee also fails the command. Each run is automatically
audited, and prerequisites are re-audited from raw files before subsequent launches.

Manual independent re-audit (substitute the printed exact run ID):

```bash
"$PY" scripts/inputnorm_gorina4.py audit --sha "$EXPECTED_SHA" \
  --run-id INPUTNORM_8x256_preflight6200_seed0_EXACT_UTC --steps 6200 --seed 0
"$PY" scripts/inputnorm_gorina4.py audit --sha "$EXPECTED_SHA" \
  --run-id INPUTNORM_8x256_preflight26000_seed0_EXACT_UTC --steps 26000 --seed 0
```

Audit checks zero process exit, telemetry, status, exact config/commit/clean state,
parameter counts, checkpoint finiteness/count and restored inference, metadata
moments, evaluation schedule and episode-return aggregates, training-row cadence
and episode records. Expected count 6200/26000; 6200 has final evaluation only,
26000 has periodic evaluation at 25000 plus separate final evaluation. No return
threshold is used. Inspect timing/SPS, raw logs and GPU telemetry as well; short
preflight throughput is not a reliable 1M runtime prediction.

Return the SHA, CPU/CUDA test output, both run IDs, gate/audit output and telemetry
summary to the reviewer. STOP here until reviewed.

### 5. Scientific seed0 — ONLY after successful server evidence and authorization

```bash
tmux new -s ele680_inputnorm_seed0
# In that session: existing environment, same reviewed EXPECTED_SHA
bash scripts/run_inputnorm_gorina4.sh "$EXPECTED_SHA" scientific 0
```

This preserves 1M steps and all scientific settings. Monitor without changing the
run; capture external exit/log/telemetry plus structured artifacts. Independently
re-audit using `audit --steps 1000000 --seed 0` and its exact run ID. Expected count
1000000, 994 finite train rows, 1000 training episodes, 40 periodic evaluations,
one final ten-episode evaluation and finite final checkpoint.

Only after seed0 is independently reviewed, repeat in distinct tmux sessions with
`scientific 1`, then after seed1 review `scientific 2`. The runner re-audits the
previous seed. Do not pool preflights into the three-seed study or modify naive
results. Archive/audit the new scientific artifacts before comparing final means,
sample SD and full per-seed learning curves.
