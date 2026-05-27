# Source-to-Claim Traceability Map

Maps every substantive claim in `final_project.tex` to its backing artifact and verification command.

**Generated:** 2026-05-26
**Manuscript:** `cs321m-paper/final_project.tex`
**Artifact root:** `qanta-buzzer/paper_exports/`

> **PR #14 follow-up review (Blocker 3) rename note (2026-05-26):**
> The canonical CSLI in `csli.json` is now the PAP-original choices-only
> excess over chance (`max(0, acc_choices_only - 1/K)`), surfaced at
> `panel_csli.{mean, ci_lower, ci_upper}` with a percentile bootstrap CI.
> The legacy in-flight-manuscript CSLI (the full-minus-choices gap) is
> preserved at `panel_question_use_gap.{mean, ci_lower, ci_upper}`
> (rows 5a-5b). Verification commands using `d['panel_csli']['mean']`
> therefore now read the choices-excess; commands that need the gap
> must use `d['panel_question_use_gap']['mean']`. Numeric claims in the
> manuscript must be cross-checked against the regenerated artifacts.

## Quantitative Claims

| # | Claim | Section | Artifact | Verification |
|---|-------|---------|----------|--------------|
| 1 | Panel CSLI (choices-only excess) | Results 4.1 | `paper_exports/csli.json` | `python3 -c "import json; d=json.load(open('paper_exports/csli.json')); print(round(d['panel_csli']['mean'],4))"` |
| 2 | CSLI 95% CI (choices-only excess) | Results 4.1 | `paper_exports/csli.json` | `python3 -c "import json; d=json.load(open('paper_exports/csli.json')); print(round(d['panel_csli']['ci_lower'],4), round(d['panel_csli']['ci_upper'],4))"` |
| 3 | TF-IDF CSLI = max(0, acc_choices_only - 1/K) | Results 4.1 | `paper_exports/csli.json` | `python3 -c "import json; d=json.load(open('paper_exports/csli.json')); print(round(d['per_model']['tfidf']['csli'],4))"` |
| 4 | SBERT CSLI = max(0, acc_choices_only - 1/K) | Results 4.1 | `paper_exports/csli.json` | `python3 -c "import json; d=json.load(open('paper_exports/csli.json')); print(round(d['per_model']['sbert']['csli'],4))"` |
| 5 | T5-small CSLI = max(0, acc_choices_only - 1/K) | Results 4.1 | `paper_exports/csli.json` | `python3 -c "import json; d=json.load(open('paper_exports/csli.json')); print(round(d['per_model']['t5-small']['csli'],4))"` |
| 5a | Panel question-use gap (former in-flight CSLI) | Results 4.1 | `paper_exports/csli.json` | `python3 -c "import json; d=json.load(open('paper_exports/csli.json')); print(round(d['panel_question_use_gap']['mean'],4))"` |
| 5b | Question-use gap 95% CI | Results 4.1 | `paper_exports/csli.json` | `python3 -c "import json; d=json.load(open('paper_exports/csli.json')); print(round(d['panel_question_use_gap']['ci_lower'],4), round(d['panel_question_use_gap']['ci_upper'],4))"` |
| 6 | No model exceeds 0.30 leakage threshold | Results 4.1 | `paper_exports/csli.json` | `python3 -c "import json; d=json.load(open('paper_exports/csli.json')); print(all(not v['leakage_flag'] for v in d['per_model'].values()))"` |
| 7 | 1000 bootstrap resamples | Results 4.1 | `paper_exports/csli.json` | `python3 -c "import json; d=json.load(open('paper_exports/csli.json')); print(d['metadata']['bootstrap_resamples'])"` |
| 8 | ECE early = 0.0259 (n=2553) | Results 4.2 | `paper_exports/calibration.json` | `python3 -c "import json; d=json.load(open('paper_exports/calibration.json')); b=d['per_bucket']['early']; print(round(b['ece'],4), b['n_samples'])"` |
| 9 | ECE mid = 0.0055 (n=3316) | Results 4.2 | `paper_exports/calibration.json` | `python3 -c "import json; d=json.load(open('paper_exports/calibration.json')); b=d['per_bucket']['mid']; print(round(b['ece'],4), b['n_samples'])"` |
| 10 | ECE late = 0.0261 (n=5255) | Results 4.2 | `paper_exports/calibration.json` | `python3 -c "import json; d=json.load(open('paper_exports/calibration.json')); b=d['per_bucket']['late']; print(round(b['ece'],4), b['n_samples'])"` |
| 11 | All ECE buckets below 0.10 threshold | Results 4.2 | `paper_exports/calibration.json` | `python3 -c "import json; d=json.load(open('paper_exports/calibration.json')); print(d['gate_verdict'])"` |
| 12 | Platt scaling on validation (n=2142) | Results 4.2 | `paper_exports/calibration.json` | `python3 -c "import json; d=json.load(open('paper_exports/calibration.json')); print(d['metadata']['n_val'])"` |
| 13 | Median absolute prefix shift = 0.0 | Results 4.3 | `paper_exports/stopdff.json` | `python3 -c "import json; d=json.load(open('paper_exports/stopdff.json')); print(d['median_abs_prefix_shift'])"` |
| 14 | All 2258 items have same stopping step (ceiling-effect diagnostic null) | Results 4.3 | `paper_exports/stopdff.json` | `python3 -c "import json; d=json.load(open('paper_exports/stopdff.json')); print(d['direction_breakdown']['same_step'])"` |
| 15 | StopDFF gate verdict = warn (ceiling effect; early+mid buckets unreachable) | Results 4.3 | `paper_exports/stopdff.json` | `python3 -c "import json; d=json.load(open('paper_exports/stopdff.json')); print(d['gate_verdict'])"` |
| 16 | Overall audit verdict WARN (StopDFF ceiling + retained-MC-subset under override) | Results 4.4 | `paper_exports/audit_card.json` | `python3 -c "import json; d=json.load(open('paper_exports/audit_card.json')); print(d['overall_verdict'])"` |
| 17 | n=2258 test questions (retained MC subset) | Results 4.1 | `paper_exports/csli.json` | `python3 -c "import json; d=json.load(open('paper_exports/csli.json')); print(d['metadata']['n_questions'])"` |
| 18 | Seed 789685 | Data 2 | `paper_exports/csli.json` | `python3 -c "import json; d=json.load(open('paper_exports/csli.json')); print(d['metadata']['seed'])"` |
| 19 | 3-model panel (TF-IDF, SBERT, T5-small) | Methods 3 | `paper_exports/csli.json` | `python3 -c "import json; d=json.load(open('paper_exports/csli.json')); print(d['metadata']['models'])"` |
| 20 | K=4 answer options | Methods 3 | `paper_exports/csli.json` | `python3 -c "import json; d=json.load(open('paper_exports/csli.json')); print(d['metadata']['K'])"` |

## Structural Claims

| # | Claim | Section | Artifact | Verification |
|---|-------|---------|----------|--------------|
| 21 | NeurIPS 2025 template with `final` option | Format | `cs321m-paper/final_project.tex` | `grep "usepackage\[final\]{neurips_2025}" cs321m-paper/final_project.tex` |
| 22 | Code link to qanta-buzzer | Reproducibility 3.4 | `cs321m-paper/final_project.tex` | `grep "github.com/ankaggarwal94/qanta-buzzer" cs321m-paper/final_project.tex` |
| 23 | Three AI disclosures present | AI Disclosures | `cs321m-paper/final_project.tex` | `grep -c "Disclosure [123]" cs321m-paper/final_project.tex` |
| 24 | Thresholds frozen before test inspection | Data 2 | `threshold_manifest.json` + `.sha256` | `ls qanta-buzzer/threshold_manifest.json qanta-buzzer/threshold_manifest.json.sha256` |
| 25 | Item-clustered 60/20/20 split | Data 2 | `paper_exports/csli.json` | `python3 -c "import json; d=json.load(open('paper_exports/csli.json')); print(d['metadata']['test_split_seed'])"` |
| 26 | Calibration model: all-MiniLM-L6-v2 | Results 4.2 | `paper_exports/calibration.json` | `python3 -c "import json; d=json.load(open('paper_exports/calibration.json')); print(d['metadata']['model'])"` |
| 27 | StopDFF uses myopic_threshold policy | Results 4.3 | `paper_exports/stopdff.json` | `python3 -c "import json; d=json.load(open('paper_exports/stopdff.json')); print(d['metadata']['stopping_policy'])"` |
| 28 | Diagnostic-only (not ex-ante DP) | Results 4.3 | `paper_exports/stopdff.json` | `python3 -c "import json; d=json.load(open('paper_exports/stopdff.json')); print(d['metadata']['metric_type'])"` |

## Summary

- **Total claims traced:** 28
- **Quantitative claims:** 20 (all backed by JSON artifacts in `paper_exports/`)
- **Structural claims:** 8 (backed by source files and configuration)
- **Unverifiable claims:** 0
- **Source artifacts:** `csli.json`, `calibration.json`, `stopdff.json`, `audit_card.json`, `threshold_manifest.json`
