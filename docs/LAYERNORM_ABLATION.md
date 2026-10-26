# LayerNorm ONLY: SAC 8x256

Status: **CODE-REVIEW CANDIDATE ONLY**, 20 October 2026. This task does not access
Gorina4 or run preflights/scientific training. Independent review of the complete
diff comes first. User-verified study status is 21/24 completed/audited 1M-step
runs; only LayerNorm seeds 0/1/2 remain. Postpone comparisons/final analysis until
they finish. Historical InputNorm/Residual results and source remain preserved.

## Frozen references and branch dependency

- Naive control: `6c674babc344e62566692e1a4fcf3611dc53a6cc`, `configs/depth_8x256.json`.
- InputNorm: `689dbee26af0ab44957bd17dafc7435d8f4a6109`, open/unmerged PR #1.
- Starting Residual: `41ffb94d03472a5ebc81b4d39fdf9c59bfbe6609`, open/unmerged PR #2.
- New `codex/layernorm-8x256` starts at that exact Residual SHA. Its stacked PR
  targets `codex/residual-8x256`, not main. Do not merge/rebase the existing PRs.
- Validate the full independently reviewed LayerNorm PR-head SHA. Changed/merged
  SHAs require new review and gates. Moving refs and synthetic merge SHAs do not
  substitute for recorded scientific source.

## Question and exact intervention

Does hidden-activation LayerNorm alone change return, sample efficiency or
between-seed variability of naive 8x256 SAC under the fixed protocol? Improvement
is a hypothesis, not an implementation-test outcome.

For actor, Q1, Q2 and both independent target critics:

```python
x = raw_observation  # critics instead use cat(raw_observation, action)
for linear, norm in zip([fc1, fc2, *hidden_extra], hidden_norms):
    x = relu(norm(linear(x)))
# unchanged actor mean/log_std heads or scalar Q head
```

Exactly eight hidden Linears of width 256, eight original ReLU positions and
eight independently parameterized `nn.LayerNorm(256, eps=1e-5,
elementwise_affine=True)` modules per network, including after the first Linear.
Each scale starts at one, bias at zero. The actor's raw 17-dimensional observation
and critic's raw 23-dimensional state-action concatenation enter the first Linear
unchanged. No normalization on raw observations/actions/rewards, output heads,
action rescaling or Q scalar; no residual links, additional hidden layers,
dropout, changed Linear initializer, SAC loss or action/log-probability formula.

| Network | Naive parameters | Added affine | LayerNorm parameters |
|---|---:|---:|---:|
| Actor | 468236 | 4096 | 472332 |
| Q1 | 466945 | 4096 | 471041 |
| Q2 | 466945 | 4096 | 471041 |
| Actor + online critics | **1402126** | **12288** | **1414414** |

Eight times (256 scales + 256 biases) = 4096/network, 12288 online total. Targets
are excluded from this count. Added affine capacity is an explicit scientific
caveat; this comparison is not parameter-matched. Target affine parameters are
independent deep copies with gradients disabled; Polyak averaging includes them.
LayerNorm has no running moments and behaves identically in train/eval modes.

Strict `layer_norm: bool = False` is appended to Actor/QNetwork/SACAgent/TrainConfig
after all historical arguments/fields; integers/strings/non-bools reject. Mixed
LayerNorm+Residual and LayerNorm+InputNorm reject rather than defining an
unreviewed combined intervention. Historical modes remain available unchanged
with LayerNorm false. Constructors retain width/depth configurability; this
scientific config and its gates strictly require 8x256.

All historical Linears are constructed in their original order; enabled norms
are constructed afterward with constant initialization and no RNG draws. Disabled
networks register no extra modules or state_dict keys and use original forward
operations. New `configs/ablation_layernorm_8x256.json` differs from the frozen
naive JSON only by explicit `layer_norm: true`, `input_norm: false` and
`residual_connections: false`.

## Frozen learning, evaluation and checkpoint contract

HalfCheetah-v5/SAC; seeds 0/1/2; 1M steps; replay 1M; batch 256; 5000 random
actions; gamma .99; tau .005; actor Adam LR .0003, critic/alpha Adam LR .001.
Critic updates retain zero-indexed `global_step > 5000`; compensated actor/alpha
frequency 2 and target frequency 1 remain. Alpha uses fresh policy samples after
actor updates. Only termination masks bootstrap; truncation still bootstraps.
Raw replay/rewards, reset/seeding/sampling and diagnostics remain intact.

Separate deterministic evaluation uses seed+10000+i for episode i: five episodes
every 25k plus separate ten-episode final evaluation. It updates no optimizer,
replay or normalizer and consumes no policy-sampling RNG. Within-evaluation SD
is population SD (ddof=0); future between-training-seed SD is sample SD (ddof=1,
n=3). Retain all seeds, including valid weak historical seeds. No comparisons
are generated here. Preflight returns never count as scientific evidence.

CSV/JSON schemas/cadences stay fixed; TensorBoard remains convenience only.
Atomic checkpoints every 100k plus final retain five networks, three optimizers,
log_alpha, completed step, config and seed. Resolved config in metadata and
checkpoint explicitly records LayerNorm mode. Save/load validate strict mode
against agent and all five networks before restoration. Historical missing flags
mean false. Existing residual/InputNorm and architecture key/shape checks remain.
No observation-normalization state is valid for LayerNorm-only. Replay/RNG/env
state remain absent: checkpoints do not support exact training continuation.

## Attribution and interpretation

LayerNorm concept: Ba, Kiros and Hinton, *Layer Normalization* (2016),
https://arxiv.org/abs/1607.06450. Uses PyTorch's existing module; no external
repository code copied. Existing CleanRL SAC, InputNorm/SimBa and Residual/He et
al. attribution remains for inherited code. This is one controlled component,
not full SimBa. Local tests cannot establish scientific efficacy, significance,
mechanism, generalization or server CUDA readiness. See local evidence separately.

## Manual Gorina4 procedure: ONLY after independent code-review authorization

The user operates the terminal **one operation at a time**. This runbook is not
permission to run jobs. Preserve historical artifacts/PRs and the existing Conda
environment. No package installs/upgrades, concurrent GPU jobs or workloads on
ssh1. Stop after any failed check and review actual evidence before proceeding.

### 1. Clean-shell Git before uenv/Conda

First inspect existing HEAD/status and ensure no active jobs. In a fresh shell:

```bash
cd /bhome/u273406/ele680-scaling-drl
test -z "${CONDA_PREFIX:-}" || { echo 'Open a fresh shell'; exit 1; }
test -z "$(git status --porcelain)" || { echo 'Dirty checkout; inspect, do not discard'; exit 1; }
EXPECTED_SHA=PASTE_INDEPENDENTLY_REVIEWED_FULL_LAYERNORM_SHA
git fetch origin codex/layernorm-8x256
git switch --detach "$EXPECTED_SHA"
test "$(git rev-parse HEAD)" = "$EXPECTED_SHA" && test -z "$(git status --porcelain)"
git log -1 --format=fuller
```

Use the full reviewed SHA, not a moving branch. Git after loading UiS Miniconda
has previously failed from OpenSSL library mismatches; use a fresh shell.

### 2. Existing environment and assigned physical GPU0

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

Require Python 3.12, Torch 2.5.1+cu121/CUDA runtime 12.1, NumPy 1.26.4, Gymnasium
1.1.1, MuJoCo 3.15.0, TensorBoard 2.18.0 and idle physical GPU0 Tesla P100.
Use the explicit Python binary; repeated uenv initialization inside active Conda
has previously selected the wrong interpreter. Do not reinstall packages.

### 3. Separate exact-commit CPU and CUDA suites

```bash
CUDA_VISIBLE_DEVICES='' "$PY" scripts/layernorm_gorina4.py tests --sha "$EXPECTED_SHA" --mode cpu
CUDA_VISIBLE_DEVICES=0 "$PY" scripts/layernorm_gorina4.py tests --sha "$EXPECTED_SHA" --mode cuda
"$PY" scripts/layernorm_gorina4.py gate --sha "$EXPECTED_SHA"
```

Inspect each output before the next operation. CUDA requires zero skips. Receipts
live in `results/layernorm_gates/<SHA>/`, experiment `layernorm_only_8x256_v1`.
InputNorm/Residual receipts cannot qualify; failed reruns invalidate old success.
The CUDA checkpoint test permits only known Adam scalar step-counter CPU->CUDA
relocation (36 actor, 68 critics, 1 alpha), checks exact values and all other
tensor devices strictly, then exercises restored optimizers and evaluation.

### 4. Distinct NON-SCIENTIFIC preflights, after server-test evidence review

Create tmux from a regular shell; inside it activate the existing environment,
set EXPECTED_SHA and OMP/MKL as above. Run 6200 first:

```bash
tmux new -s ele680_layernorm_pre6200 -c /bhome/u273406/ele680-scaling-drl
# Inside tmux after setup:
bash scripts/run_layernorm_gorina4.sh "$EXPECTED_SHA" preflight6200 0
"$PY" scripts/layernorm_gorina4.py audit --sha "$EXPECTED_SHA" --run-id EXACT_6200_RUN_ID --steps 6200 --seed 0
```

Review log, exit, timing/memory and audit before creating the separate 26000
tmux session; repeat environment/SHA setup there:

```bash
tmux new -s ele680_layernorm_pre26000 -c /bhome/u273406/ele680-scaling-drl
# Inside tmux after setup:
bash scripts/run_layernorm_gorina4.sh "$EXPECTED_SHA" preflight26000 0
"$PY" scripts/layernorm_gorina4.py audit --sha "$EXPECTED_SHA" --run-id EXACT_26000_RUN_ID --steps 26000 --seed 0
```

6200 has a final ten-episode eval; 26000 adds periodic five-episode eval at 25000.
The runner re-audits 6200 from raw artifacts. Audit checks config/source/versions,
strict mode flags, topology/parameter counts, finite reloadable actor/all critics,
complete CSV schemas/schedules, no InputNorm artifacts, completion log, zero
Python exit and numeric P100 telemetry. No return threshold. Review actual
evidence and **STOP before scientific authorization**.

### 5. Future science: separate authorization after reviewed preflights

Prepared only; not authorized by this task. Same exact SHA, tmux, idle GPU0:
seed0, independent audit/review, then seed1, independent audit/review, then seed2.

```bash
bash scripts/run_layernorm_gorina4.sh "$EXPECTED_SHA" scientific 0
"$PY" scripts/layernorm_gorina4.py audit --sha "$EXPECTED_SHA" --run-id EXACT_SCIENCE_RUN_ID --steps 1000000 --seed 0
```

Runner re-audits both preflights and the immediately previous scientific seed,
checks gates twice around launch and acquires `results/layernorm_gpu0.lock`.
The cooperative lock cannot reserve the shared GPU against other users/runners.
Unique `LAYERNORM_8x256_*` IDs avoid overwrites. External `.log`, `.exitcode` and
five-second `.gpu.csv` files live under `/bhome/u273406/`. Python and tee must
both succeed; telemetry monitor must remain alive through training completion.

At 1M, audit expects 994 finite training rows (6001,7001,...,999001), 1000 training
episodes, 40 periodic evaluations plus final, correct checkpoint and provenance.
Failed re-audits invalidate receipts. Preserve originals and archive new raw
artifacts before final comparisons. No LayerNorm scientific results exist here.
