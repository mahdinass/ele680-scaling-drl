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
| Python syntax | Pending final check |
| Complete CPU-compatible unit suite | Pending final check |
| Analysis suite | Pending final check |
| Bundled 24-run arithmetic and identities | Pending final check |
| Figure regeneration | Pending final check |
| Documentation links and README paths | Pending final check |
| Clean Python 3.12 dependency installation | Pending final check |
| Cleanup PR CI | Pending final check |

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
