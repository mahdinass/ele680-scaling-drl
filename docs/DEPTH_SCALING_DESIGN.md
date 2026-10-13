# SAC depth scaling (Phase 2) - design notes

Question: how does the number of hidden layers in the SAC actor and critics affect
performance, sample efficiency, stability and compute under a fixed protocol?

| Config | Hidden layers x width | Trainable params (HalfCheetah-v5) |
|---|---|---:|
| `configs/baseline.json` (reference) | 2 x 256 | 217,870 |
| `configs/depth_4x256.json` | 4 x 256 | 612,622 |
| `configs/depth_8x256.json` | 8 x 256 | 1,402,126 |

Depth = number of fully connected hidden layers before the output head(s). Actor,
Q1, Q2 and both target critics share the same depth. Layers beyond `fc1`/`fc2` are
256->256 `Linear` + ReLU, stored in `hidden_extra` (an `nn.ModuleList`, empty for
depth 2, so depth-2 `state_dict` keys, shapes and init RNG consumption are identical
to the frozen implementation). Minimum supported depth is 2.

The depth configs differ from `baseline.json` only by `hidden_depth`. No other
hyperparameter, initialization, normalization, residual connection or optimizer
change is introduced. Historical runs/configs without `hidden_depth` mean depth 2.

## Limitation
Increasing depth at fixed width also increases the parameter count (2.8x for 4x256,
6.4x for 8x256). The study therefore measures *depth scaling at fixed width*, not a
parameter-matched, depth-only causal effect. Results are descriptive (n = 3 seeds).
