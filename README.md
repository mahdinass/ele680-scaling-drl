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
