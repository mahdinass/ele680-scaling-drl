# Residual-only local validation — 18 October 2026

Readiness: **CODE REVIEW ONLY**. Independent user inspection precedes Gorina4
validation. No remote connection, server change, CUDA preflight or scientific
residual training was performed. No residual efficacy claim is made.

## Actual environment

Reused the existing ignored `.venv-inputnorm` CPU environment; installed or
updated no dependencies. The plotting `.venv` remained separate.

| Component | Actual local version |
|---|---|
| Windows Python | 3.12.14 |
| PyTorch | 2.5.1+cpu |
| NumPy | 1.26.4 |
| Gymnasium | 1.1.1 |
| MuJoCo | 3.15.0 |
| TensorBoard | 2.18.0 |
| CUDA available | False |

## Executed commands and results

Workspace-local temporary storage avoids sandbox temp ACL failures:

```powershell
New-Item -ItemType Directory -Force .venv-inputnorm/tmp | Out-Null
$env:TEMP=(Resolve-Path .venv-inputnorm/tmp).Path
$env:TMP=$env:TEMP
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
.\.venv-inputnorm\Scripts\python.exe -B -m unittest tests.test_residual -v
.\.venv-inputnorm\Scripts\python.exe -B -m unittest tests.test_residual_gates tests.test_residual_audit -v
.\.venv-inputnorm\Scripts\python.exe -B -m unittest discover -s tests -t . -v
.\.venv\Scripts\python.exe -B -m unittest discover -s analysis/tests -v
```

| Verification | Actual result | Exit |
|---|---|---:|
| Focused architecture/checkpoint/integration | 17 discovered, **16 passed**, **1 CUDA skipped**, 58.296 s | 0 |
| Initial residual gate/artifact tests | **12 passed**, 29.562 s | 0 |
| Full implementation discovery | 183 discovered, **177 passed**, **6 CUDA skipped**, 210.713 s | 0 |
| Existing complete analysis suite | **115 passed**, 121.864 s | 0 |
| Final gate/artifact rerun after receipt-path SHA guard | **12 passed**, 14.408 s | 0 |
| Python AST parsing (all src, new residual tests/helper) | **15 files passed** | 0 |
| Git Bash `bash -n scripts/run_residual_gorina4.sh` | **Passed**, syntax only | 0 |
| `git diff --check` | **Passed** | 0 |

The focused tests are included in full discovery, not additional unique tests.
The only final helper refinement validates a full lowercase SHA before computing
receipt paths; its gate/artifact tests were rerun successfully afterward. The
full suite and analysis ran concurrently; times are local observations, not
scientific throughput measurements.

Logs remain under ignored `.venv-inputnorm/`: `residual-focused.log`,
`residual-infra.log`, `residual-full-tests.log`, `residual-analysis-tests.log`,
`residual-infra-final.log`. Synthetic `AUDIT PASSED` lines in these logs are
fixture tests, not evidence of real Gorina4 runs. Git Bash initially failed to
initialize inside the sandbox (Windows status -1073741502); the explicit syntax
check succeeded outside the sandbox. No shell failure was counted as a pass.

## CUDA tests unavailable locally

All six were reported as unittest skips, never local passes:

- `tests.test_evaluate.EvaluateTests.test_cuda`
- `tests.test_networks.NetworkTests.test_cuda_device_placement`
- `tests.test_sac_update.SACAgentTests.test_update_on_cuda`
- `tests.test_train_smoke.TrainScheduleTests.test_cuda_smoke`
- `tests.test_inputnorm.AgentTests.test_cuda_normalization_updates_evaluation_and_checkpoint`
- `tests.test_residual.ArchitectureTests.test_cuda_forward_backward_checkpoint_and_evaluation`

Actual server CPU/CUDA suites, idle physical GPU0/environment checks, separate
6200/26000 preflights and scientific runs remain unperformed. Mocked gate tests
and Bash syntax do not establish successful server orchestration.

## Material verified behavior

- Symbolic graphs and numeric forward hooks verify eight hidden Linears, eight
  hidden ReLUs and exactly seven identity additions before the existing ReLUs.
- Zeroing all seven hidden transformations leaves an analytic nonzero gradient
  through the identity paths; every hidden weight receives a finite gradient.
- Actor/Q shapes, bounded sampled actions, finite CPU forward/backward/SAC losses,
  independent frozen targets and updates covering the final hidden layer passed.
- Actor 468236 + Q1 466945 + Q2 466945 = **1402126**; enabled/disabled initialization,
  state dictionaries and RNG consumption are identical. No normalization/dropout
  module or trainable skip parameter is introduced.
- Strict bool validation covers TrainConfig/config loading, SACAgent, Actor and
  QNetwork; historical positional arguments are preserved.
- Disabled depths 2/8 match actual frozen networks, SAC and training-loop source
  at `6c674babc344e62566692e1a4fcf3611dc53a6cc`: forward/action/loss trajectories,
  network and optimizer states, RNG, raw replay, evaluation and schedule counters.
  Existing InputNorm reference tests were strengthened to load frozen networks
  too, and filter the newly appended flag from historical TrainConfig arguments.
- Residual checkpoint inference/optimizer roundtrip passes. Missing historical
  flags mean false. Enabled/disabled/malformed config flags and inconsistent
  actor/critic/target flags reject before modifying receiving network, optimizer,
  alpha or normalization state. Save rejects config/network disagreement too.
- Residual-only training uses no observation transform or normalizer. Evaluation
  at both 5/10 episode settings leaves the training trajectory and RNG unchanged;
  schedules, raw state/reward/termination/truncation storage and action bounds pass.
- Real HalfCheetah-v5 residual CPU smoke: 12 training steps with optimization,
  periodic/final evaluation, logging, metadata and checkpoint writes passed.
- Separate synthetic 6200/26000/1M artifact fixtures pass their expected schedules.
  Corrupt config, checkpoint flag, normalization contamination, versions/provenance,
  output schemas, nonfinite metrics/weights, missing evaluation/log/telemetry and
  nonzero exit are rejected. Failed re-audits invalidate old receipts.
- Wrong/dirty source, altered frozen config, naive/normalized/wrong-depth topology,
  unexpected LayerNorm, busy/wrong GPU, wrong Python, missing/skipped CUDA receipts,
  InputNorm receipts and unsafe receipt-path SHA values are rejected.

## Complete-diff review and remaining limits

Reviewed the complete staged implementation against starting source
`689dbee26af0ab44957bd17dafc7435d8f4a6109` and the network diff against frozen
naive source `6c674babc344e62566692e1a4fcf3611dc53a6cc`.

Existing scientific configs, requirements, raw results/analysis, InputNorm source
and launch helpers, replay, seeding, evaluation and CSV logger/metadata source
have no changes. SAC loss/update methods and training-loop schedules have no
changes. The isolated config adds residual=true and explicit input_norm=false.
No report or past experiment outputs were edited. The parent branch/PR #1 is
preserved; the new PR is stacked on its frozen source.

No unresolved local test failures remain. CUDA/server readiness and residual
scientific outcomes remain unknown. Generalization/significance/mechanism claims
and exact training continuation are unsupported. Use the full residual commit
SHA in the PR handoff as a code-review candidate; any changed SHA needs new
review/server gates. Stop here for independent code review.

## CUDA checkpoint comparison correction — 18 October 2026

The user independently tested original candidate
`c61a3aca7acabbdf19995740bbcbc1c0b39aa4ec` on Gorina4: CPU 183 tests passed
with 6 CUDA skips; CUDA 183 tests had one error in the residual checkpoint
comparison. Their recursive diagnostic found only optimizer step-counter device
differences: actor 20, critics 36, alpha 1, each CPU -> CUDA:0 with identical values.
These server observations are user-supplied evidence, not local CUDA verification.

Investigated the installed PyTorch 2.5.1 source and its official version tag:

- [Adam initialization](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/optim/adam.py):
  non-capturable, non-fused Adam initializes scalar step counters on CPU, while
  moment tensors follow parameters.
- [Optimizer loading](https://github.com/pytorch/pytorch/blob/v2.5.1/torch/optim/optimizer.py):
  `_process_value_according_to_param_policy` preserves already-loaded step tensors
  in that mode. The existing checkpoint loader uses `map_location=agent.device`,
  moving those serialized CPU counters to CUDA before optimizer restoration.

The correction changes only the failing CUDA test and this evidence record.
A comparison local to that test explicitly permits CPU -> actual indexed CUDA
relocation only at the three optimizers' `state.<parameter_id>.step` paths. It
requires scalar shape, identical dtype/value and exactly 20/36/1 such counters.
All other tensors, including model parameters/buffers, log_alpha and optimizer
moments, must have identical CUDA devices, dtype, shape and exact values. The
shared EqualityMixin is unchanged; no blanket CPU conversion is applied.

The restored CUDA agent then performs critic, actor and alpha updates. The test
requires finite losses, model/alpha parameters and all optimizer state tensors,
each counter incremented by one, and moment tensors remaining on CUDA. Existing
CUDA training/evaluation coverage remains. Production checkpoint/SAC logic,
architecture, hyperparameters, configurations and launch helpers are unchanged.

Local CUDA remains unavailable. The revised comparison and loaded-state CUDA
updates require the user's complete CPU/CUDA gate repeat on the new full commit
SHA. Previous receipts cannot validate it. No Gorina4 connection or preflight or
scientific run is performed or authorized by this correction. Stop after
implementation for independent review.

Actual local reruns for this correction (same commands/environment as above):

| Verification | Actual result | Exit |
|---|---|---:|
| Full implementation discovery | 183 discovered, **177 passed**, **6 CUDA skipped**, 370.934 s | 0 |
| Existing complete analysis suite | **115 passed**, 124.743 s | 0 |
| Final test AST and comparison against original test module | Passed; only the CUDA-specific test differs | 0 |
| `git diff --check`; protected production/shared-comparison paths | Passed; unchanged | 0 |

Logs: ignored `.venv-inputnorm/residual-cuda-fix-tests.log` and
`.venv-inputnorm/residual-cuda-fix-analysis.log`. The modified CUDA test is one
of the six skips; neither its revised assertions nor its restored-state updates
were executed locally. No unresolved local test failures remain.
