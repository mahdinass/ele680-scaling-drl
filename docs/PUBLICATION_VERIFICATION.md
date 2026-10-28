# Publication verification — 28 October 2026

This record covers housekeeping based on `6de6e56239a7873b3d0ce3659d4a3505b64456b9`
(main after PR #5). It does not replace the [scientific status and provenance
record](FINAL_STUDY_REPRODUCIBILITY.md) or claim new training results.

## Security review

- Reviewed all 34 locally reachable historical commits, including all five GitHub
  pull-request heads, and all 229 unique historical file blobs (9,387,351 bytes).
- Gitleaks 8.30.1, downloaded from its official release and checked against its
  published SHA-256 checksum, found no credential leaks in full-history diffs,
  exported historical blobs, or commit messages. Authentication filenames and
  sensitive-access/restriction keywords were also reviewed. No private keys,
  passwords, tokens, protected authentication files, or institutional publication
  restriction were identified in the reviewed material.
- Retrieved and scanned the logs of all five available pre-cleanup Actions runs:
  `40670673002`, `40668909088`, `40356880018`, `40356362921`, `40355627681`.
  All five runs succeeded; no credential leaks were found in their logs. The
  Actions artifact inventory was empty.
- The cleanup PR's successful [Python 3.12 CI run](https://github.com/mahdinass/ele680-scaling-drl/actions/runs/40682207569)
  was also downloaded and scanned without credential findings. The housekeeping
  diff contains no authentication material.
- Retained the historical infrastructure notes and original scientific metadata.
  The repository owner explicitly accepted disclosure of the hostnames and ordinary
  SSH commands; this is not a claim of separate institutional security approval.
  UiS publicly lists the gateway on its [server fingerprint page](https://www.ux.uis.no/server-ssh-fingerprint.html).
  These values are not authentication credentials. Current general analysis
  instructions use repository-relative paths and an activated Python environment.
- No tracked file exceeded 508 kB; the largest historical blob was a scientific
  PNG (507,304 bytes). No accidental checkpoint, environment, cache, or
  authentication artifact was identified in tracked history.

This is a bounded review of available repository history and Actions evidence,
not a guarantee that no secret could exist. Local untracked experiment outputs
and environments are excluded from publication. No Git history was rewritten.

## Reproducibility checks

| Check | Result |
| :-- | :-- |
| Historical reproduction plan | Passed: 24 commands, eight conditions × three seeds |
| Exact source provenance | Passed: seven valid scientific commits, all ancestors of main; each configuration exists at its recorded commit, matches the current version, and supports the printed CLI options |
| Python syntax | Passed for the source, test, script, and analysis files; CI compileall includes the added reproduction-plan test module (38 Python files total) |
| Complete CPU-compatible unit suite | 215 discovered: 208 passed, seven CUDA tests skipped; zero failures/errors in the corrected clean Linux CI run |
| Analysis suite | All 115 existing tests passed in Linux CI and locally on Windows; both added historical-plan regressions also passed locally, bringing CI discovery to 117 |
| Bundled 24-run arithmetic and identities | Passed locally and in CI: all 24 runs and 320 periodic aggregate points |
| Figure regeneration | Passed locally for final-study, baseline, and width plots: 19 PNG/PDF pairs plus summary tables, all under ignored analysis/generated/; original figures unchanged |
| MuJoCo environment | HalfCheetah-v5 reset and one step passed locally; real-environment CPU smoke tests also passed in CI |
| Documentation links and README paths | All active local Markdown links resolved; README command/config paths checked; the report path is explicitly pending and has no active link |
| Clean Python 3.12 dependency installation | Passed from a fresh Linux Actions checkout using the documented requirements and official CPU PyTorch index; pip check reported no broken requirements |
| Cleanup PR CI | Both independent summary verification and full CPU/analysis jobs passed |

The clean CI environment was Ubuntu with Python **3.12.15**, PyTorch **2.5.1+cpu**,
NumPy **1.26.4**, Gymnasium **1.1.1**, MuJoCo **3.15.0**, TensorBoard **2.18.0**,
pandas **2.2.3**, Matplotlib **3.10.7**, and tabulate **0.9.0**. Direct analysis and
simulator dependencies are pinned in the new verification requirements; these are
not a complete transitive lockfile. The original training requirements and recorded
environment metadata are preserved.

The seven skipped tests cover CUDA evaluation, network placement, SAC updates,
training smoke, and InputNorm/Residual/LayerNorm CUDA checkpoint paths. They were
not counted as passes. An initial Windows sandbox run encountered temporary-file
permission errors; the entire original suite then passed unchanged with writable
temporary storage (207 passed, seven CUDA skips, Python 3.12.14). The clean local
dependency installation also passed pip check on Python 3.12.5.

### Verification issues found and addressed

A second Linux run exposed an intermittent failure in the original LayerNorm
per-sample/batch-independence test, which had passed in the first Linux run and on
Windows. Its fixture consumed the global RNG without isolation. Float32 one-row
and batched CPU kernels can accumulate differently near zero; a local 64-fixture
probe found double-precision differences no larger than 4.5e-15. The test now uses
an isolated fixed RNG seed. **Every original assertion and float32 tolerance is
retained.** An additional three-seed float64 test checks the same property with
1e-12 absolute / 1e-11 relative tolerance. The [corrected complete CI run](https://github.com/mahdinass/ele680-scaling-drl/actions/runs/40684198774)
passed all 208 CPU tests with seven CUDA skips. This is a test-fixture correction,
not a change to scientific LayerNorm code or recorded results.

The historical command plan previously wrote to `results_reproduced/`, which the
early source revisions did not ignore. It now uses `results/reproduced/` so newly
created outputs do not by themselves mark historical run provenance as dirty.
Two added regression checks validate all 24 printed commands against recorded
source/configuration/CLI options and check that every source revision ignores the
output root. The README calls for a separate clean training clone and saving the
plan outside it before switching revisions. No training command was executed.

The workflow retains the independent bundled-summary check and adds a full-history
Python 3.12 CPU job for syntax, both test suites, dependency consistency, and final
figure generation. CUDA-specific tests must be reported as skipped on CPU builds.

## Limits and unchanged scientific record

No SAC implementation, architecture, scientific configuration, measured CSV/JSON,
checkpoint, recorded source SHA, or tracked original figure was changed. No new
scientific training was launched, no Gorina4 files were accessed, and the separate
report repository was not modified. The two LayerNorm seeds with documented
third-party GPU overlap remain qualified in the README and final scientific record.

The complete 24-run raw archive is not present in this checkout or publicly
downloadable from GitHub. Full raw-artifact verification and historical CUDA
audits were therefore not repeated. The bundled checker validates secondary
summaries, not checkpoint contents or historical GPU exclusivity. The optional
`verify-raw` convenience command checks checkpoint **presence**, not its tensor
payload, and does not replace the frozen per-condition auditors. Repeating training
requires the recorded source revisions, compatible software, allocated hardware,
and the separately archived evidence for a full historical audit. Numerical
trajectories need not be identical across hardware/software versions.

The existing licensing position is unchanged: the CleanRL MIT notice and source
attribution remain; no license for the group's original code has been added.
Public visibility does not itself grant a reuse license.

## Adding the final report

1. Commit the finalized PDF as `docs/ELE680_Final_Report.pdf`.
2. Replace the prominent pending-report sentence near the top of `README.md` with
   the PDF link supplied in the adjacent HTML comment; remove that comment.
3. Open the rendered README on GitHub and confirm that the link opens the PDF.

No final tagged release is created before the report is available.
