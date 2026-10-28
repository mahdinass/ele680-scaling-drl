# Scaling Network Architectures in Deep Reinforcement Learning

**ELE680 · Deep Learning · University of Stavanger · 2026**  
**Group 1:** Mahdi Nasser and Malvin Varpe

This project investigates how **network width, depth, and architectural components** affect Soft Actor-Critic (SAC) on the continuous-control **HalfCheetah-v5** task. We trained eight actor–critic architecture conditions across three independent seeds each, for **24 one-million-step experiments**. The goal was a controlled empirical comparison—not a new reinforcement-learning algorithm or a reproduction of the full SimBa architecture.

The repository contains the training implementation, experiment configurations, tests, recorded summary data, and scripts for independently checking the reported results. The [companion LaTeX report](https://github.com/mahdinass/ele680-drl-report) contains the full methodology and discussion (access is required while that repository remains private).

## Results at a glance

Each number below is the **mean final deterministic evaluation return across three training seeds ± between-seed sample standard deviation**. Each seed's final score is itself a mean over ten evaluation episodes, after one million environment steps.

| Architecture | Final return, mean ± SD (n = 3) |
| :-- | --: |
| Baseline · 2×256 | 11,906 ± 497 |
| Wider · 2×512 | 11,645 ± 482 |
| Wider · 2×1024 | 11,084 ± 1,595 |
| Deeper · 4×256 | 11,884 ± 1,838 |
| Deeper · 8×256 | 8,247 ± 5,072 |
| 8×256 + input normalization only | 12,951 ± 481 |
| 8×256 + residual connections only | 12,674 ± 1,539 |
| 8×256 + layer normalization only | 8,153 ± 5,016 † |

The measured results do **not** establish a general scaling law or statistically significant superiority: there are only three training seeds per condition, and individual deep-network outcomes vary markedly. The 8×256 InputNorm condition had the highest observed three-seed mean in this study, but that is a descriptive result rather than a general performance guarantee.

† **Execution qualification:** LayerNorm seeds 1 and 2 completed training and passed independent artifact audits, but an external monitoring runner detected overlapping third-party GPU processes and exited nonzero. They remain in the analysis with that limitation explicitly recorded; exclusive-GPU timing is not claimed for these runs. See [scientific status and limitations](docs/FINAL_STUDY_REPRODUCIBILITY.md).

For exact, unrounded results and every seed, see [`final_summary.csv`](analysis/data/final_24run/final_summary.csv) and [`run_inventory.csv`](analysis/data/final_24run/run_inventory.csv). For the complete 40-checkpoint evaluation series, see [`learning_curves.csv`](analysis/data/final_24run/learning_curves.csv). The [analysis directory](analysis/README.md) also contains baseline and width-comparison plots.

## Repository guide

| Location | Contents |
| :-- | :-- |
| [`src/`](src/) | SAC training, networks, replay, evaluation, checkpointing, and experiment logging |
| [`configs/`](configs/) | Recorded configurations for all eight conditions |
| [`tests/`](tests/) | Unit, integration, architecture, and artifact-audit tests |
| [`analysis/`](analysis/) | Versioned result summaries, analysis scripts, and figures |
| [`docs/`](docs/) | Methodology, historical validation records, scientific status, and source attribution |
| [`scripts/`](scripts/) | Original cluster-specific ablation runners and verification gates |
| [`requirements/`](requirements/) | Portable dependencies and the original CUDA environment specification |

Historical validation documents are retained for provenance. The final outcome and qualifications are stated in [`docs/FINAL_STUDY_REPRODUCIBILITY.md`](docs/FINAL_STUDY_REPRODUCIBILITY.md), which supersedes any pre-run status in older notes.

## Set up and validate

Use **Python 3.12**. Create and activate a virtual environment first. On a CPU machine, install the portable dependencies and a compatible CPU PyTorch build:

```bash
python -m pip install -r requirements/base.txt
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m unittest discover -s tests -t . -v
```

The original Gorina4 CUDA environment used `requirements/gorina4.txt` and an assigned Tesla P100; this specification is **not** a general-purpose CPU installation command. Tests requiring CUDA are skipped on CPU-only machines. Some historical source-compatibility tests require the full Git history, so clone normally rather than using a shallow checkout.

To verify that MuJoCo/HalfCheetah loads:

```bash
python -c "import gymnasium as gym; e=gym.make('HalfCheetah-v5'); e.reset(seed=0); e.step(e.action_space.sample()); e.close(); print('MuJoCo OK')"
```

### Validate and plot the archived summaries (no GPU required)

```bash
python analysis/final_24run_reproduce.py verify-bundled
python -m pip install matplotlib
python analysis/final_24run_reproduce.py plots --output-dir analysis/generated/final_24run
```

`verify-bundled` checks all 24 run identities and the 320 architecture/checkpoint rows, including recomputing the across-seed means and sample SDs. Plotting uses recorded data and writes only to the generated-output directory. These checks validate the versioned **secondary summaries**, not the original full training artifacts.

If you have the separately archived original run directories unpacked under `results/`, you can additionally run:

```bash
python analysis/final_24run_reproduce.py verify-raw --results-root results
```

The full raw run folders (including checkpoints and TensorBoard logs) are deliberately not committed to Git. The original 24-run archive is held separately by the group and can be provided to an assessor on request; it is not available for anonymous download from this repository. See the [reproducibility record](docs/FINAL_STUDY_REPRODUCIBILITY.md) for checks, provenance, and archive details.

### Repeat training (allocated GPU required)

```bash
python analysis/final_24run_reproduce.py plan
```

`plan` **prints a dry-run command plan only**; it never starts jobs or changes the checkout. Each historical experiment was run at a recorded source commit (below). Full reproduction requires appropriate hardware access, the corresponding environment, an unshallowed Git history, and new output/run IDs. Run experiments sequentially only on a GPU allocated to you; do not assume `CUDA_VISIBLE_DEVICES=0` selects an available device on a different system. Identical seeds do not guarantee byte-identical trajectories across hardware/software versions.

For a single example with the current merged implementation, after setting up an appropriate CUDA environment and GPU allocation:

```bash
CUDA_VISIBLE_DEVICES=0 python -m src.train --config configs/baseline.json --seed 0 --device cuda --output-dir results_reproduced --run-id REPRO_baseline_seed0
```

Do not reuse the original run IDs or overwrite scientific artifacts. The original cluster runners under [`scripts/`](scripts/) may require their exact frozen historical source revision.

## Experimental design and source revisions

All conditions share the same SAC algorithm, training budget, evaluation schedule, and non-architecture hyperparameters. Periodic deterministic evaluation used five episodes every 25,000 steps; final evaluation used ten episodes. The baseline appears in both the width and depth comparisons but was trained **only three times**, not duplicated.

| Condition | Configuration | Recorded scientific source commit |
| :-- | :-- | :-- |
| 2×256 | [`baseline.json`](configs/baseline.json) | `1a9f2405b5fdb9cffa379b72f569a0a3a01515e4` |
| 2×512 | [`width_2x512.json`](configs/width_2x512.json) | `f25b3a0cb93da0a605b3dbac6254a866f83f664e` |
| 2×1024 | [`width_2x1024.json`](configs/width_2x1024.json) | `e533a3c76ef069ffe37affc9a6a15ea2b1ae2e14` |
| 4×256 | [`depth_4x256.json`](configs/depth_4x256.json) | `6c674babc344e62566692e1a4fcf3611dc53a6cc` |
| 8×256 | [`depth_8x256.json`](configs/depth_8x256.json) | `6c674babc344e62566692e1a4fcf3611dc53a6cc` |
| 8×256 + InputNorm | [`ablation_inputnorm_8x256.json`](configs/ablation_inputnorm_8x256.json) | `689dbee26af0ab44957bd17dafc7435d8f4a6109` |
| 8×256 + Residual | [`ablation_residual_8x256.json`](configs/ablation_residual_8x256.json) | `41ffb94d03472a5ebc81b4d39fdf9c59bfbe6609` |
| 8×256 + LayerNorm | [`ablation_layernorm_8x256.json`](configs/ablation_layernorm_8x256.json) | `3a53be9ed5d68e7572e78a83dbdb0d0b7496f8e2` |

The versioned [`run_inventory.csv`](analysis/data/final_24run/run_inventory.csv) is the machine-readable authority for exact per-seed run IDs, commits, scores, runtimes, and deviation flags. The current `main` branch unifies the historical architecture implementations; it is not itself the historical commit for every run.

## Research attribution and responsible reuse

- **Method:** Soft Actor-Critic (Haarnoja et al., ICML 2018).
- **Implementation guidance:** CleanRL's continuous-action SAC reference, including adapted mathematical/control-flow patterns and a credited small matching code block.
- **Architecture inspiration:** residual networks (He et al.), layer normalization (Ba et al.), and SimBa-related scaling questions (Lee et al.). These are tested as isolated interventions, not claimed as novel algorithms.

See [`docs/SOURCE_ATTRIBUTION.md`](docs/SOURCE_ATTRIBUTION.md) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for detailed code provenance and the retained CleanRL MIT notice. This is a **joint course project**. No statement here assigns unverified personal contributions to either author, and no separate license for the group's own code is currently included. Public visibility by itself does not grant permission to reuse that code.
