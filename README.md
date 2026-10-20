# ELE680 — Scaling Up Networks in Deep Reinforcement Learning

Deep Reinforcement Learning project for ELE680.

Team:
- Mahdi Nasser
- Malvin Varpe

## Setup

Requires **Python 3.12**.

### Gorina4 (UiS)

```bash
uenv miniconda3-py312
conda activate /bhome/u273406/conda-envs/ele680
python -m pip install -r requirements/gorina4.txt
```

`requirements/base.txt` holds the portable dependencies; `requirements/gorina4.txt` adds the CUDA 12.1 build of PyTorch (`torch==2.5.1+cu121`) from the official PyTorch wheel index.

Group 1 uses **physical GPU 0** (Tesla P100). Run jobs with `CUDA_VISIBLE_DEVICES=0`, e.g.:

```bash
CUDA_VISIBLE_DEVICES=0 python <script>.py
```

### Smoke test

```bash
CUDA_VISIBLE_DEVICES=0 python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import gymnasium as gym; e = gym.make('HalfCheetah-v5'); e.reset(seed=0); e.step(e.action_space.sample()); print('ok')"
```

Expected: `2.5.1+cu121 True` and `ok`.

### Git on Gorina4

Perform Git operations from a **fresh Gorina4 shell before loading the UiS
Miniconda environment or activating Conda**.

The UiS `miniconda3-py312` environment can alter shared-library resolution and
cause system Git/SSH to fail with an OpenSSL version mismatch, for example:

```text
OpenSSL version mismatch. Built against 30000020, you have 30300010
```

### Activating the project environment

Every new Gorina4 shell:

```bash
uenv miniconda3-py312
conda activate /bhome/u273406/conda-envs/ele680
```

### Experiment infrastructure policy

The canonical rules for reproducible ELE680 experiments are documented in
[`docs/EXPERIMENT_INFRASTRUCTURE_POLICY.md`](docs/EXPERIMENT_INFRASTRUCTURE_POLICY.md).

## Running the tests

From the repository root, with the project environment active (the suite includes a short real `HalfCheetah-v5` run and CUDA smoke tests, and should take well under a minute):

```bash
CUDA_VISIBLE_DEVICES=0 python -m unittest discover -s tests -v
```

## Running an experiment

Experiments are launched from a JSON config plus optional overrides:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.train --config configs/baseline.json \
    --seed 0 --device cuda --output-dir results
```

Overridable flags: `--seed`, `--device`, `--total-steps`, `--output-dir`, `--run-id` (all other settings come from the config file). `CUDA_VISIBLE_DEVICES=0` selects physical GPU 0, our assigned GPU; `--device cuda` then means "that GPU", and the code never hardcodes a GPU index. To keep a console log, redirect output yourself, e.g. `... 2>&1 | tee train.log`.

> **Warning:** `configs/baseline.json` is the full **scientific baseline** (1,000,000 steps, 1,000,000-transition replay buffer, evaluations every 25,000 steps). Do not launch it by accident. For a quick check of the launcher use a tiny run, for example:
>
> ```bash
> CUDA_VISIBLE_DEVICES=0 python -m src.train --config configs/baseline.json \
>     --total-steps 300 --output-dir /tmp/ele680_smoke --run-id smoke
> ```
>
> (With the baseline config this still performs the final 10-episode evaluation, i.e. 10 x 1000 environment steps.)

### Where results are written

Each run gets its own directory `results/<run_id>/` (the default run id is `<env>_seed<seed>_<UTC timestamp>_<4 hex>`; an existing directory is never overwritten). `results/` is git-ignored.

| File | Content |
|---|---|
| `config.json` | the full config used (with the resolved `run_id`) |
| `metadata.json` | run id, status, start/end time, runtime, git commit (+ dirty flag), Python/PyTorch/CUDA/Gymnasium/MuJoCo versions, GPU name, env id, seed, device, config, parameter counts, checkpoint path, final evaluation |
| `status.json` | quick look: `running` / `completed` / `failed` (+ failure type and message) |
| `train_metrics.csv` | every `log_interval` steps: losses, alpha, mean abs Q, actor/critic gradient norms, elapsed time, steps per second |
| `episode_metrics.csv` | one row per finished training episode: return and length |
| `eval_metrics.csv` | one row per evaluation (`periodic` / `final`): number of episodes, mean, std, per-episode returns |
| `checkpoint.pt` | latest checkpoint (networks, targets, optimizers, `log_alpha`, step, config, seed) |
| `tensorboard/` | TensorBoard event files (convenience only) |

The CSV/JSON files are the source of truth; TensorBoard is only a viewer (`tensorboard --logdir results`). All `step` columns mean "environment steps completed". Actor columns in `train_metrics.csv` are blank on rows where no actor update happened.

Checkpoints do **not** include the replay buffer or RNG states, so they are for inspection/evaluation, not exact training resumption (yet).

## InputNorm ablation candidate

The 8x256 InputNorm-only implementation and manual Gorina4 validation procedure
are documented in [docs/INPUTNORM_ABLATION.md](docs/INPUTNORM_ABLATION.md).
Local verification is recorded in
[docs/INPUTNORM_LOCAL_VALIDATION.md](docs/INPUTNORM_LOCAL_VALIDATION.md).
Scientific InputNorm results are not yet available; server preflights remain required.

## Residual-only 8x256 implementation

The isolated residual-only design, checkpoint semantics and manual Gorina4 gates
are in [docs/RESIDUAL_ABLATION.md](docs/RESIDUAL_ABLATION.md), with actual local
test evidence in [docs/RESIDUAL_LOCAL_VALIDATION.md](docs/RESIDUAL_LOCAL_VALIDATION.md).
This branch is stacked on frozen InputNorm commit
`689dbee26af0ab44957bd17dafc7435d8f4a6109` (open PR #1), while its new experiment
disables InputNorm. Readiness is **CODE REVIEW ONLY**; server validation and
residual scientific results remain pending. The earlier InputNorm candidate
status above is historical; the latest Notion milestones record its completed
three-seed study.

## LayerNorm-only 8x256 implementation

The final required ablation is implemented on `codex/layernorm-8x256`, stacked
on exact reviewed Residual source `41ffb94d03472a5ebc81b4d39fdf9c59bfbe6609`.
See [docs/LAYERNORM_ABLATION.md](docs/LAYERNORM_ABLATION.md) for architecture,
compatibility and the manual exact-commit validation workflow, and
[docs/LAYERNORM_LOCAL_VALIDATION.md](docs/LAYERNORM_LOCAL_VALIDATION.md) for local
evidence. **Code-review candidate only:** no LayerNorm server validation,
preflight or scientific training has been performed by this task. Latest
user-verified status (20 October): 21/24 scientific runs audited, including all
InputNorm and Residual seeds; only LayerNorm seeds 0/1/2 remain. Earlier candidate
status text is historical. Defer comparisons until LayerNorm science completes.
