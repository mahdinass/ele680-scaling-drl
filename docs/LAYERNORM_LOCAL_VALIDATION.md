# LayerNorm-only local validation — 20 October 2026

Readiness: **CODE REVIEW ONLY**. No Gorina4 connection, package change, actual
server CUDA test, preflight or scientific LayerNorm run was performed. Existing
scientific artifacts and PRs are preserved; comparisons remain deferred.

## Actual environment

Reused the ignored `.venv-inputnorm` implementation environment and separate
`.venv` plotting environment without installing/upgrading dependencies.

| Component | Implementation runtime |
|---|---|
| Windows Python | 3.12.14 |
| PyTorch | 2.5.1+cpu |
| NumPy | 1.26.4 |
| Gymnasium | 1.1.1 |
| MuJoCo | 3.15.0 |
| TensorBoard | 2.18.0 |
| CUDA available | False |

## Commands and actual evidence

Run from repository root. Workspace-local temporary storage avoids the observed
sandbox temporary-directory ACL failures; these settings do not change code:

```powershell
$env:TEMP=(Resolve-Path .venv-inputnorm/tmp).Path
$env:TMP=$env:TEMP
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
.\.venv-inputnorm\Scripts\python.exe -B -m unittest tests.test_layernorm_gates tests.test_layernorm_audit -v
.\.venv-inputnorm\Scripts\python.exe -B -m unittest discover -s tests -t . -v
.\.venv\Scripts\python.exe -B -m unittest discover -s analysis/tests -v
```

| Verification | Actual result | Exit |
|---|---|---:|
| Initial gate/artifact suite | 12 passed, 18.075 s | 0 |
| First full implementation discovery | 213 discovered, **206 passed, 7 CUDA skipped**, 214.633 s | 0 |
| Final gate/artifact suite, including run identity/nonfinite JSON rejection | **13 passed**, 19.563 s | 0 |
| Final full implementation discovery | **214 discovered, 207 passed, 7 CUDA skipped**, 226.628 s | 0 |
| Existing complete analysis suite with workspace temp | **115 passed**, 116.125 s | 0 |
| Python AST syntax, src + new tests/helper | **15 files passed** | 0 |
| SAC loss/update function ASTs vs exact Residual parent | **Unchanged** | 0 |
| Training-loop AST vs parent, excluding only new constructor keyword | **Identical** | 0 |
| Git Bash `bash -n scripts/run_layernorm_gorina4.sh` | **Passed**, syntax only | 0 |
| Shell file LF-only byte check | **Passed** | 0 |

The final 13 gate/artifact tests were rerun after adding the JSON-finiteness guard
while full discovery was in progress; they validate the final helper source.
Final whitespace review (`git diff --check`) passed. No unresolved local failures
remain.

Focused tests are included in discovery; do not add them to the total as unique
tests. Synthetic artifact fixtures create fake 6200/26000/1M-step records without
training those budgets. Their `AUDIT PASSED` output is not remote/science evidence.

Actual failed attempts, retained rather than counted as passes:

- First LayerNorm-focused run: 18 discovered, six subtest errors, one CUDA skip,
  64.576 s, exit1. The new snapshot helper reused a Residual-era helper that
  assumed the actual naive agent had `observation_normalizer`; it predates that
  attribute. Fixed only the new LayerNorm helper to represent absent historical
  normalization as None. Subsequent full discovery passed; no production change
  was needed for this test error.
- Initial analysis run without workspace TEMP/TMP: 97 executed tests, 111 errors
  including setup/subtest failures, 28.910 s, exit1. Traces show WinError5 on
  sandbox `AC/Temp` directories. Rerun uses the same tests and source with the
  workspace temp settings above; no historical data or analysis source changed.
- Initial sandboxed Git Bash could not initialize its MSYS named object
  (`NtCreateDirectoryObject`, access denied). The explicitly escalated syntax-only
  check succeeded, exit0; it did not execute the runner or access any server.

Logs remain ignored under `.venv-inputnorm/`: `layernorm-focused.log`,
`layernorm-infra.log`, `layernorm-full-tests.log`, `layernorm-infra-final.log`,
`layernorm-full-final.log`, `layernorm-analysis-tests.log`,
`layernorm-analysis-final.log`. Tests sometimes ran concurrently; reported wall
times are local observations, not scientific throughput benchmarks.

## Verified implementation contracts

- Graph and numeric hooks check eight hidden Linear -> independent LayerNorm ->
  ReLU stages, including the first; no shortcuts or normalized output heads.
  Raw actor input and critic state-action concatenation remain unchanged.
- Actor 472332 + Q1 471041 + Q2 471041 = 1414414 online parameters; 12288 added
  affine parameters. Eight 256-scale/256-bias normalizers per network, initialized
  to one/zero; all 80 affine tensors across five networks have independent storage.
  Historical Linear weights and post-construction RNG remain identical.
- CPU actions/log-probs/Q shapes and bounds, finite forward/backward passes,
  gradients through all hidden Linears/affines, optimizer changes and exact Polyak
  updates including normalizers. Targets remain independent, initially equal,
  gradient-disabled and unchanged until explicit target updates.
- Strict bool validation and appended signatures. Mixed LayerNorm with InputNorm
  or residual rejects. Disabled mode registers no extra network modules/keys.
- Actual Git-loaded naive `6c674babc344e62566692e1a4fcf3611dc53a6cc`, InputNorm
  `689dbee26af0ab44957bd17dafc7435d8f4a6109` and Residual
  `41ffb94d03472a5ebc81b4d39fdf9c59bfbe6609` sources are used for depth-2/8
  exact comparisons: initialization, sampled actions, losses/optimizers/targets,
  raw replay, schedules, evaluations, training trajectories and Python/NumPy/Torch
  RNG. Missing frozen source is a failure, not a skipped compatibility check.
- Historical full checkpoints load in new disabled agents, and new disabled
  checkpoints load in actual historical loaders, for all three historical modes
  at depths 2/8. Enabled checkpoint affine/optimizer/inference roundtrip passes.
- Enabled/disabled, malformed/missing flag, config/agent/all-five-network mode
  disagreements and affine key/shape corruption reject before receiving-state
  mutation. Failed saves leave the existing checkpoint unchanged. Existing
  residual/InputNorm checkpoint protections continue to pass.
- LayerNorm has no running moments or batch dependence. Both 5/10-episode
  evaluation paths preserve state and policy RNG. Raw replay/rewards, random
  warm-up and update counters remain intact. Real HalfCheetah CPU smoke performs
  12 training steps with optimization, periodic/final evaluation, logging and
  checkpoint writes in a temporary directory.
- Dedicated synthetic gates/audits reject wrong/dirty SHA, config/provenance drift,
  mixed modes, wrong environment/GPU, missing/failed/skipped CUDA receipts, other
  ablation receipts, unsafe IDs, wrong topology/affines/heads, shared parameters,
  nonfinite artifacts, missing evaluation rows, bad CSV schemas/logs/exit/telemetry.
  Failed re-audits invalidate receipts. These fixtures do not prove live orchestration.

## CUDA coverage unavailable locally

Seven skips are expected on this CPU-only runtime, never local passes:

- `tests.test_evaluate.EvaluateTests.test_cuda`
- `tests.test_networks.NetworkTests.test_cuda_device_placement`
- `tests.test_sac_update.SACAgentTests.test_update_on_cuda`
- `tests.test_train_smoke.TrainScheduleTests.test_cuda_smoke`
- `tests.test_inputnorm.AgentTests.test_cuda_normalization_updates_evaluation_and_checkpoint`
- `tests.test_residual.ArchitectureTests.test_cuda_forward_backward_checkpoint_and_evaluation`
- `tests.test_layernorm.ArchitectureTests.test_cuda_forward_backward_checkpoint_optimizer_evaluation_and_training`

New CUDA coverage includes all LayerNorm gradients/updates, checkpoint roundtrip,
exact state values/devices, restored optimizer steps, deterministic evaluation
isolation and short fake-env CUDA training. Only scalar Adam step counters may
relocate CPU->CUDA (36 actor, 68 critics, 1 alpha); all other tensors remain under
strict device assertions. Global EqualityMixin and historical CUDA test are unchanged.
This branch was not executed locally and requires zero-skip Gorina4 validation.

## Protected paths and remaining limits

Existing configs, requirements, scientific/analysis data, InputNorm/Residual
helpers, observation-normalization/replay/env/evaluation/logging source remain
unchanged. SAC loss/update methods and training-loop scheduling are unchanged.
Only existing tests' exact config-field/signature expectations and projection
into the historical TrainConfig were updated for the appended field; no prior
assertion or CUDA comparison was weakened. README/policy append current context.

No server CUDA readiness or scientific outcome is claimed. Exact continuation
remains unsupported; shared GPU load and n=3 limit future interpretation. Review
the complete stacked diff before authorizing any Gorina4 action. The manual
exact-SHA test/preflight/science sequence is in `LAYERNORM_ABLATION.md`.
