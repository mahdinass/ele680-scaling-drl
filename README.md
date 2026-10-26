# ELE680 — Scaling Up Networks in Deep Reinforcement Learning

**Group 1: Mahdi Nasser and Malvin Varpe** · University of Stavanger, ELE680 (2026)

## Final project status

**Completed:** 24 independently trained SAC runs on Gymnasium HalfCheetah-v5 (8 architecture conditions × seeds 0,1,2 × 1,000,000 environment steps). All completed runs passed the independent **artifact audits**. The LayerNorm seeds 1 and 2 **did not meet the originally strict exclusive-GPU monitoring rule**: a separate, documented exception runner exited nonzero after recording overlapping third-party CUDA processes, despite successful training and original artifact audits. These runs are included with explicit qualification; we do not claim unaffected GPU timing or improved/damaged return.

The companion LaTeX report is at https://github.com/mahdinass/ele680-drl-report . The training code, scripts, original configs and per-run tests live here. Original historical scientific commits have been retained in Git history, including by merging the original three stacked architecture PRs (#1/#2/#3) without squash.

## Install and basic checks

Use Python 3.12. For UiS Gorina4, use the assigned module and project environment; to reproduce in a fresh compatible environment:

~~~bash
python -m pip install -r requirements/gorina4.txt
python -m unittest discover -s tests -t . -v
~~~

`requirements/base.txt` contains portable dependencies; `requirements/gorina4.txt` pins the recorded CUDA PyTorch build. Local CPU hosts can run the CPU tests; CUDA tests require CUDA hardware and are skipped when unavailable. For a quick environment smoke test:

~~~bash
python -c "import gymnasium as gym; e=gym.make('HalfCheetah-v5'); e.reset(seed=0); e.step(e.action_space.sample()); e.close(); print('MuJoCo OK')"
~~~

Do not assume GPU 0 is yours on another machine: `CUDA_VISIBLE_DEVICES=0` selects the first physically visible device and was specifically the assigned P100 on Gorina4. Ask the administrator for an exclusive allocation before launching a million-step experiment. Do not defeat the original per-run hardware gates or kill another user's process.

## Experimental matrix

| Condition | JSON config | Original scientific code SHA |
|---|---|---|
| 2×256 | `configs/baseline.json` | `1a9f2405b5fdb9cffa379b72f569a0a3a01515e4` |
| 2×512 | `configs/width_2x512.json` | `f25b3a0cb93da0a605b3dbac6254a866f83f664e` |
| 2×1024 | `configs/width_2x1024.json` | `e533a3c76ef069ffe37affc9a6a15ea2b1ae2e14` |
| 4×256 | `configs/depth_4x256.json` | `6c674babc344e62566692e1a4fcf3611dc53a6cc` |
| 8×256 naive | `configs/depth_8x256.json` | `6c674babc344e62566692e1a4fcf3611dc53a6cc` |
| 8×256 InputNorm **only** | `configs/ablation_inputnorm_8x256.json` | `689dbee26af0ab44957bd17dafc7435d8f4a6109` |
| 8×256 Residual **only** | `configs/ablation_residual_8x256.json` | `41ffb94d03472a5ebc81b4d39fdf9c59bfbe6609` |
| 8×256 LayerNorm **only** | `configs/ablation_layernorm_8x256.json` | `3a53be9ed5d68e7572e78a83dbdb0d0b7496f8e2` |

For every condition, **three independent training seeds** 0/1/2, 1M steps, periodic deterministic evaluation of five episodes every 25k and separate ten-episode final evaluation. The 2×256 baseline is counted once despite appearing in both width and depth comparisons. No preflights enter the scientific results.

## Repeat all 24 experiments (dry-run command plan)

~~~bash
python analysis/final_24run_reproduce.py verify-bundled
python analysis/final_24run_reproduce.py plan > reproduce_24_commands.sh
~~~

`plan` prints, but **never executes**, 24 exact-commit checkout + training commands, one for each manifest row. Inspect `reproduce_24_commands.sh` before use. It uses new `REPRO_...` IDs and a separate results directory. Training is never started by the analyzer. Run only sequentially on an allocated GPU and ideally in a clean clone, with the required environment already installed. Different PyTorch/GPU/driver versions and stochastic RL produce different numerical trajectories despite matching seeds; reproducibility means recreating the procedure, not promising byte-identical rewards.

To start **one** experiment from the appropriate historical source revision (example uses the current merged implementation; for exact historical execution, switch to the SHA in the matrix):

~~~bash
CUDA_VISIBLE_DEVICES=0 python -m src.train --config configs/ablation_inputnorm_8x256.json --seed 0 --device cuda --output-dir results_reproduced --run-id REPRO_InputNorm_seed0
~~~

Do not reuse the original run IDs or overwrite original output folders. The historical per-architecture control scripts and exact-SHA gates remain at `scripts/` and `docs/`, but some require the unchanged frozen scientific commit rather than today's merged `main` commit. Follow the original gate documentation when repeating historically identical runs.

## Validate original results and regenerate plots without GPU

Original completed experiments produce distinct folders `results/<run_id>/` with `config.json`, `metadata.json`, `status.json`, `checkpoint.pt`, `train_metrics.csv`, `episode_metrics.csv`, `eval_metrics.csv` and TensorBoard logs. Checkpoints do *not* capture replay buffer, environment state or RNG, so they are not sufficient to resume exactly. Raw 24-run folders are large (~740 MB) and are excluded from Git; our original archived copy is separate.

This repository includes **three versioned, measured secondary-summary CSVs** (not fabricated metrics) in `analysis/data/final_24run/`:

- `run_inventory.csv`: all 24 exact run IDs, commits, seed scores, runtime, deviation flags.
- `learning_curves.csv`: all 8×40 periodic checkpoints, per-seed 5-episode means and across-seed mean/sample SD.
- `final_summary.csv`: 3-seed final means and sample SD.

Run these without a GPU:

~~~bash
python analysis/final_24run_reproduce.py verify-bundled
python analysis/final_24run_reproduce.py plots --output-dir analysis/generated/final_24run
~~~

The second command requires Matplotlib and generates separate width, depth and deep-component mean ± **between-training-seed sample SD** learning curves. Plotting never changes the frozen source CSVs. For full independent verification after copying/unpacking the original run directories into this repository's `results/`:

~~~bash
python analysis/final_24run_reproduce.py verify-raw --results-root results
~~~

This secondary checker does not replace the historical frozen code's original full auditor. See `docs/FINAL_STUDY_REPRODUCIBILITY.md` for the original artifact and analysis provenance, archive and GPU limitations.

## Experimental design and source acknowledgments

SAC itself is from Haarnoja et al. (ICML 2018). The squashed-Gaussian actor, twin Q critics and update schedules were **guided by** the CleanRL `sac_continuous_action.py` reference. The network modifications are literature-motivated: He et al. for residual connections, Ba et al. for LayerNorm and Lee et al. (SimBa, ICLR 2025) for the broader scaling design. The method is not claimed as a new RL algorithm or a literal SimBa reproduction. Source URLs, contributions and adapted ideas are itemized in `docs/SOURCE_ATTRIBUTION.md`. Check source originality and licensing when reusing code outside this course; no statement here substitutes for a line-by-line plagiarism review.

The measured study is a **shared Group 1 submission by Mahdi Nasser and Malvin Varpe**. This repository does not infer each individual's personal contribution; both members should confirm a truthful division for oral assessment. The report discusses our critical interpretation without attributing unverified tasks to either member.
