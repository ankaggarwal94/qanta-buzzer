# Code Package Checklist

CS321M Final Project -- "Does the Proxy Preserve the Decision?"

Verified: 2026-05-26

| Item | Status | Location | Notes |
|------|--------|----------|-------|
| README with setup instructions | PRESENT | `README.md` | CS321M-oriented with Environment Setup and Quickstart sections |
| Pinned requirements.txt | PRESENT | `requirements.txt` | 14 dependencies pinned with == from working environment |
| Quickstart smoke run | PRESENT | `scripts/manual-smoke.sh` | 4-stage pipeline; ~2-5 min on CPU |
| Fresh split script | PRESENT | `scripts/fresh_split.py` | v10 section 0.3 protocol, seed 789685 |
| CSLI computation | PRESENT | `scripts/compute_csli.py` | Panel across TF-IDF, SBERT, T5-small. Canonical CSLI = `max(0, acc_choices_only - 1/K)` (PAP-original); legacy gap published as `panel_question_use_gap`. K=4 invariant enforced at runtime. |
| Prefix calibration | PRESENT | `scripts/compute_prefix_calibration.py` | Platt scaling + per-bucket ECE. Producer downgrades `gate_verdict` to `warn` on degenerate (`constant`) or empty buckets and records `gate_verdict_reason`. |
| StopDFF diagnostic | PRESENT | `scripts/compute_stopdff.py` | Myopic-threshold approach. Producer downgrades `gate_verdict` to `warn` on `ceiling_effect_detected` or any unreachable bucket; records `gate_verdict_reason` and `threshold_only_verdict`. |
| Audit card generator | PRESENT | `scripts/make_audit_card.py` | Aggregates all 3 metrics. Overall verdict downgrades to WARN with a `retained-subset` qualifier when any coverage/retention gate was overridden. Cross-checks each source artifact's recorded `script_sha256` against the live script. |
| Figure regeneration | PRESENT | `scripts/regenerate_figures.py` | Rebuilds tables + figures from cached JSONs |
| Audit card JSON | PRESENT | `paper_exports/audit_card.json` | Overall verdict + per-metric summaries |
| CSLI results JSON | PRESENT | `paper_exports/csli.json` | Per-model `csli` = choices-only excess with bootstrap CI under `panel_csli`. Legacy gap under `panel_question_use_gap` with its own CI. Carries `metadata.generation.{script_sha256, git_commit, argv}`. |
| Calibration results JSON | PRESENT | `paper_exports/calibration.json` | ECE by prefix bucket |
| StopDFF results JSON | PRESENT | `paper_exports/stopdff.json` | Median absolute prefix shift |
| Audit table (LaTeX) | PRESENT | `paper_exports/audit_table.tex` | 8-column table for manuscript |
| CSLI panel figure | PRESENT | `paper_exports/csli_panel.png` | Bar chart across 3 models |
| Reliability diagram (early) | PRESENT | `paper_exports/reliability_early.png` | Calibration plot for early prefixes |
| Reliability diagram (mid) | PRESENT | `paper_exports/reliability_mid.png` | Calibration plot for mid prefixes |
| Reliability diagram (late) | PRESENT | `paper_exports/reliability_late.png` | Calibration plot for late prefixes |
| Threshold manifest | PRESENT | `threshold_manifest.json` | Frozen before test-set inspection (DATA-03) |
| Threshold manifest sidecar | PRESENT | `threshold_manifest.json.sha256` | SHA-256 integrity verification (load-time enforced via `scripts/threshold_manifest.py`) |
| StopDFF attestation report | PRESENT | `stopdff_report.json` | Timestamp + verdict + parameters |
| Attribution in README | PRESENT | `README.md` (## Attribution) | 4 contributors named; Novel vs Reused separation |
| Source-to-claim map | PRESENT | `reproducibility/source_to_claim.md` | Phase 9 deliverable |
| Test suite | PRESENT | `tests/` | 429 tests across 33 files (pytest) |
