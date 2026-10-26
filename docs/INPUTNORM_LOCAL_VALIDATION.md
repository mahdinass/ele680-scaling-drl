# InputNorm local validation — 16 October 2026

Readiness: **READY FOR GORINA4 PREFLIGHT**. No server commands or scientific
training were executed. No scientific intervention result is claimed.

## Environment

Dedicated ignored `.venv-inputnorm`, separate from the existing plotting `.venv`:

| Component | Actual local version |
|---|---|
| Windows Python | 3.12.14 |
| PyTorch | 2.5.1+cpu |
| NumPy | 1.26.4 |
| Gymnasium | 1.1.1 |
| MuJoCo | 3.15.0 |
| TensorBoard | 2.18.0 |

CPU build deliberately uses the same Torch release; it is not the frozen server's
CUDA wheel. Server requirements/environment and plotting packages were not changed.
Creation/install commands (uv cache kept inside ignored `.venv`):

```powershell
uv --cache-dir .venv/.uv-cache venv --python .venv/Scripts/python.exe .venv-inputnorm
uv --cache-dir .venv/.uv-cache pip install --python .venv-inputnorm/Scripts/python.exe -r requirements/base.txt torch==2.5.1 mujoco==3.15.0 --extra-index-url https://download.pytorch.org/whl/cpu
```

Installation exited 0 after sandbox network access was authorized. An initial
default-cache attempt hit filesystem permission errors and the sandbox initially
blocked package DNS access; both were resolved without changing dependencies.

## Executed verification

Set workspace-local temporary storage to avoid Windows sandbox temp ACL failures:

```powershell
New-Item -ItemType Directory -Force .venv-inputnorm/tmp | Out-Null
$env:TEMP=(Resolve-Path .venv-inputnorm/tmp).Path
$env:TMP=$env:TEMP
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
.\.venv-inputnorm\Scripts\python.exe -B -m unittest discover -s tests -t . -v
.\.venv-inputnorm\Scripts\python.exe -B -m unittest tests.test_inputnorm_audit -v
.\.venv\Scripts\python.exe -B -m unittest discover -s analysis/tests -v
```

| Command | Actual result | Exit |
|---|---|---:|
| Complete implementation suite | 153 discovered; **148 PASSED**, **5 BLOCKED** (unittest skipped); 103.238 s | 0 |
| Additional synthetic audit integration test, added after suite discovery | **1 PASSED**, including corruption subtests; 10.916 s | 0 |
| Existing complete analysis suite in plotting environment | **115 PASSED**, 75.299 s | 0 |
| `git diff --check` | PASSED, after removing an added blank EOF line | 0 |
| Python AST parse of all src files, new tests and audit helper | PASSED | 0 |
| Git Bash `bash -n scripts/run_inputnorm_gorina4.sh` | PASSED (syntax only) | 0 |

Thus all 154 current implementation tests were covered: 149 passed, 5 blocked.
The extra audit integration test was run separately; do not describe the earlier
153-test discovery as including that later-added test. Logs remain local under
ignored `.venv-inputnorm/` (`tests-final.log`, `audit-tests.log`, `analysis-tests.log`).

The first implementation/analysis attempts FAILED with sandbox temporary-directory
PermissionErrors (not accepted as passing evidence). Reruns using the workspace
temp directory produced the successful results above. No tests were deleted or
weakened to resolve those permissions.

## Explicitly BLOCKED, never passed locally

CUDA is unavailable in this CPU environment:

- `tests.test_evaluate.EvaluateTests.test_cuda`
- `tests.test_networks.NetworkTests.test_cuda_device_placement`
- `tests.test_sac_update.SACAgentTests.test_update_on_cuda`
- `tests.test_train_smoke.TrainScheduleTests.test_cuda_smoke`
- `tests.test_inputnorm.AgentTests.test_cuda_normalization_updates_evaluation_and_checkpoint`

Actual Gorina4 CUDA/MuJoCo preflights, hardware availability and full scientific
results are BLOCKED pending the user's manual execution. Shell syntax and mocked
gate tests do not establish successful server orchestration.

## Material checks that passed

- Real HalfCheetah-v5 CPU smoke with 8x256 InputNorm, optimization and evaluation;
  existing real-environment and TensorBoard tests also executed with real dependencies.
- First policy action at global_step 5000 observes count **5001**.
- Exactly **1000000** sequential updates of the production Welford estimator produce
  count 1000000. This is an estimator unit stress test, NOT a million-step RL run.
  Training integration verifies one update per decision, and enabled checkpoints
  require count == completed steps. Actual full-run count remains a server audit gate.
- Explicit 5-episode periodic and 10-episode final evaluation preserve count, mean,
  M2, cached mean and cached inverse standard deviation; update is patched to fail
  if invoked. Actual training evaluation call sites also pass independent snapshots.
- Same transform reaches current actor/critics, next actor/targets, fresh alpha
  sampling and diagnostics; raw replay and actions remain unchanged.
- Disabled CPU initialization/RNG, actions/log-probabilities, losses, optimizer
  states, target updates and miniature training trajectories match actual source
  at control `6c674babc344e62566692e1a4fcf3611dc53a6cc`, for depth 2 and 8.
- Architecture/parameter counts, enabled inference/checkpoint roundtrip, historical
  flag-absent checkpoint and strict legacy network weights, corrupt state rejection.
- Synthetic preflight audit accepts a complete fixture and rejects altered process
  exit, missing telemetry, config/provenance/statistics drift, missing evaluations,
  nonfinite training metrics and incorrect normalizer count. Fixtures are not results.
- Wrong SHA/dirty source, config drift, wrong/busy GPU, wrong Python, missing/skipped
  CUDA gates and stale successful receipts after a failed test rerun are rejected.

## Complete-diff review

Original scientific configs, `src/networks.py`, `src/replay_buffer.py`, `src/envs.py`,
all requirements, historical data and figures have no changes. No report changes.
The only scientific configuration difference is `input_norm: true`. All 8 hidden
layers and 1402126 trainable actor/critic parameters remain. Source modifications
connect a pure observation transform; no SAC loss/optimizer/sampling schedule or
reward/bootstrap change was found. Added scripts are manual-only validation tools.

Do not infer efficacy, CUDA equivalence, an observation-scale failure mechanism,
or exact training-resume support from these local checks. Freeze the exact full
reviewed PR-head SHA supplied in the final handoff; if it changes, redo server gates.
