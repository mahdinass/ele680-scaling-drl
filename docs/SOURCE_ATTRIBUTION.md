# Code provenance and research attribution — ELE680 Group 1

This file identifies **influences and implementation decisions**, not a claim that every source line has been formally plagiarism-scanned. All group members must verify compliance with the University's academic-integrity requirements. No third-party repository should be presented as wholly original work.

## Core SAC algorithm and reference code

**Scientific method:** Haarnoja, Zhou, Abbeel & Levine (2018), *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*, ICML/PMLR: https://proceedings.mlr.press/v80/haarnoja18b.html .

**Code-guidance reference:** CleanRL contributors, `cleanrl/sac_continuous_action.py`, https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/sac_continuous_action.py , consulted October 2026. CleanRL's permissive project license must be consulted for any copied portions and retained where required by the license.

Project files explicitly citing that CleanRL reference include `src/networks.py`, `src/sac_update.py`, and `src/train.py`. The following mathematical design or control-flow choices come from the shared SAC literature/CleanRL reference rather than being proposed new algorithms: twin Q critics and target critics, tanh-squashed reparameterized Gaussian actor and bounded log standard deviation, action affine rescaling/log-Jacobian correction, automatic alpha tuning, Polyak target averaging, off-policy replay learning, warm-up and delayed/compensated actor-update schedule. The complete implementation has its own project structure, dataclass configuration, observation normalization and experiment logging/audit systems; do not infer from that fact alone that no lines resemble the cited implementation.

## Architectural motivations

- **Residual identity shortcuts:** He, Zhang, Ren & Sun (2016), *Deep Residual Learning for Image Recognition*, CVPR, DOI:10.1109/CVPR.2016.90. `src/networks.py` implements an eight-hidden-layer MLP with seven same-width shortcuts, not a reproduced ResNet architecture.
- **Layer normalization:** Ba, Kiros & Hinton (2016), *Layer Normalization*, arXiv:1607.06450 https://arxiv.org/abs/1607.06450 . The project applies learnable normalization after hidden linear transforms in actor and critics, before ReLU.
- **SimBa architecture/scaling inspiration:** Lee et al. (2025), *SimBa: Simplicity Bias for Scaling Up Parameters in Deep Reinforcement Learning*, ICLR, https://arxiv.org/abs/2410.09754 . Inspired the choice to test normalization, residuals and LayerNorm **one at a time**. The experiments are not claimed to reproduce the complete SimBa network, its benchmarks or its claimed scaling results.
- **Related depth/normalization research:** Bjorck, Gomes & Weinberger (2021), *Towards Deeper Deep Reinforcement Learning with Spectral Normalization*, NeurIPS, https://arxiv.org/abs/2106.01151 . Motivated the hypothesis that naive depth can be hard to optimize. Spectral normalization is **not** an implemented intervention in this 24-run core project.
- **Input normalization estimator:** `src/observation_normalization.py` explicitly describes its Welford moment update as an independently implemented observation-only running-statistics mechanism inspired by SimBa, not direct reproduction of SimBa combined blocks.

## Project-specific engineering contribution and boundaries

The codebase includes configurable actor/critic network scaling, isolated InputNorm/Residual/LayerNorm flags with historical default-off compatibility tests, deterministic evaluation isolation, custom per-run provenance and checkpoint metadata, CPU/CUDA verification, exact-source validation and independent auditors, preflight/science separation, metric aggregation and GPU-process anomaly records. These are implementation and experimental-engineering decisions in this project; they do not establish a new RL method. The source's own headers preserve reference attribution.

The group consists of **Mahdi Nasser and Malvin Varpe**. This document makes **no unsupported allocation of individual code-writing or review tasks**. Each member should be able to explain the code and record his actual contribution accurately for the oral assessment. AI-assisted drafting/testing does not remove the group's responsibility to inspect source origins or understand the implementation.

## Before submission: integrity checks

1. Search for copied/near-verbatim external code blocks and retain notices/licensing where required.
2. Check differences from the cited CleanRL reference, including the independently implemented actor, SAC update, replay and runner files.
3. Confirm all figures derive from recorded CSV/JSON, not online example results, smoothed illustrations or filtered best seeds.
4. Keep exact Git SHAs in raw run metadata, cite each upstream paper and code repository in the final LaTeX bibliography.
5. Make all reproducibility artifacts available to examiner under course data-access conditions.

This attribution page is a transparent provenance map, **not an automated originality certificate**.
