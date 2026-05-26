---
title: Scientific Metric Edge-Case Guards
date: 2026-05-26
category: logic-errors
module: qanta-buzzer scientific metrics
problem_type: logic_error
component: tooling
symptoms:
  - "CSLI coverage counted matched rows instead of unique question IDs"
  - "Prefix calibration could crash or become undefined for empty and single-class buckets"
  - "StopDFF reachability assumed cosine=1.0 was always the maximum calibrated probability"
  - "Package artifacts could remain technically correct but stale or non-portable"
root_cause: logic_error
resolution_type: code_fix
severity: high
related_components:
  - development_workflow
  - documentation
tags:
  - qanta-buzzer
  - scientific-metrics
  - csli
  - prefix-calibration
  - stopdff
  - pr-review
---

# Scientific Metric Edge-Case Guards

## Problem

During PR #14 babysitting, live review comments exposed real correctness risks in the CS321M metric scripts even though the branch was mergeable and CI was green. The failures were edge-case logic bugs in scientific audit code: coverage cardinality, calibration degeneracy, and threshold reachability semantics.

These bugs mattered because the scripts generate manuscript-facing artifacts. A green build alone did not prove that the reported CSLI, calibration, and StopDFF metadata preserved the intended measurement contract.

## Symptoms

- `scripts/compute_csli.py` computed MC coverage from matched row count. Duplicate MC rows for one `qid` could inflate coverage and hide missing test questions.
- `scripts/compute_prefix_calibration.py` warned about single-class validation buckets but still read `labels[0]` and still fit `LogisticRegression`, which fails on empty or single-class buckets.
- The calibration reporting path always called `predict_proba`, which fails on empty test buckets even after successful model fitting.
- `scripts/compute_stopdff.py` treated cosine similarity `1.0` as the maximum calibrated score for every Platt model, which is false when the fitted coefficient is negative.
- Package docs/artifacts carried stale state: an absolute path in `artifacts/smoke/run_metadata.json` and a `PENDING` source-to-claim checklist row after the source-to-claim map existed.

## What Didn't Work

- Treating CI as sufficient did not catch the bugs; the failures lived in semantic invariants rather than syntax or broad test coverage.
- Printing warnings without changing control flow was not enough. The calibration script could warn about a degenerate bucket and then immediately take a path that requires non-empty, two-class labels.
- Counting records was the wrong proxy for CSLI coverage. The measurement question is whether each test `qid` is represented, not whether enough rows happened to match.
- Assuming the positive end of a bounded domain is always the maximum ignored the sign of the Platt coefficient.

Session-history search found no relevant prior sessions beyond the current PR #14 babysitting work.

## Solution

### Count CSLI Coverage by Unique Question ID

Separate matched rows from matched unique test questions. Keep the row count for downstream compute visibility, but gate coverage on unique `qid`s.

```python
def _filter_test_mc_questions(
    mc_questions: list[dict],
    test_qids: set[str],
) -> tuple[list[dict], dict[str, Any]]:
    questions = [q for q in mc_questions if str(q["qid"]) in test_qids]
    matched_qids = {str(q["qid"]) for q in questions}
    missing_qids = test_qids - matched_qids
    coverage = len(matched_qids) / max(1, len(test_qids))
    return questions, {
        "matched_test_mc_questions": len(questions),
        "matched_test_mc_qids": len(matched_qids),
        "missing_qids": missing_qids,
        "coverage_rate": coverage,
    }
```

The artifact metadata now records both `matched_test_mc_questions` and `matched_test_mc_qids`, making duplicate-row inflation visible.

### Use Explicit Calibration Fallbacks

Model degenerate calibration buckets as degenerate calibrators instead of letting them fall into sklearn paths that require two classes.

```python
class ConstantCalibrationModel:
    def __init__(self, probability: float, reason: str) -> None:
        self.probability = float(probability)
        self.reason = reason

    def predict_proba(self, raw_scores: np.ndarray) -> np.ndarray:
        positive = np.full(len(raw_scores), self.probability, dtype=float)
        return np.column_stack([1.0 - positive, positive])
```

`_fit_bucket_calibrator()` now returns a constant model for empty validation buckets and single-class buckets. `_calibrate_bucket_scores()` returns an empty array for empty test buckets before calling `predict_proba`.

The output JSON records `platt_model_type`, `platt_fallback_reason`, and `platt_constant_probability` per bucket, so a future manuscript or audit card can distinguish true logistic calibration from fallback behavior. Downstream StopDFF loading converts constant fallback buckets into an equivalent zero-slope Platt form, preventing the metric consumer from crashing on `null` coefficients.

### Evaluate StopDFF Reachability Over the Full Domain

The Platt transform is monotone in `coef * x + intercept`. When `coef < 0`, the maximum over cosine similarity in `[-1, 1]` occurs at `x = -1`, not `x = 1`.

```python
max_raw_score = 1.0 if coef >= 0 else -1.0
max_cal = calibrate_score(max_raw_score, coef, intercept)
cal_at_sim_1 = calibrate_score(1.0, coef, intercept)
cal_at_sim_neg_1 = calibrate_score(-1.0, coef, intercept)
```

The StopDFF artifact now records `max_calibrated_probability`, `max_calibrated_raw_score`, `calibrated_at_sim_1`, and `calibrated_at_sim_neg_1`.

### Keep Package Artifacts Portable

Review package artifacts as part of metric-script fixes. In PR #14, `artifacts/smoke/run_metadata.json` was changed from an absolute local path to `artifacts/smoke/train_dataset.json`, and `CODE_CHECKLIST.md` was updated to mark `reproducibility/source_to_claim.md` as present.

## Why This Works

Each fix replaces an incidental proxy with the invariant the audit actually depends on:

- CSLI coverage is a unique-question invariant, so it must be computed over unique `qid`s.
- Calibration bucket degeneracy is real information; constant fallback models preserve a defined output while recording the limitation.
- StopDFF reachability is a bounded-domain optimization problem; evaluating the correct endpoint based on coefficient sign gives the true maximum for a linear-logistic transform.
- Portability fields and checklist status are part of the reproducibility contract, not cosmetic metadata.

The fixes also made the contract testable. `tests/test_pr14_review_regressions.py` encodes the exact edge cases reviewers found so future changes cannot silently regress them.

## Prevention

- Add regression tests for reviewer-discovered scientific edge cases before patching the scripts.
- Assert the measurement invariant directly: unique qids for coverage, defined calibrators for degenerate buckets, and bounded-domain extrema for reachability.
- Regenerate manuscript-facing artifacts from committed code after metric-script changes.
- Re-query both GitHub review surfaces before marking a PR done: inline review threads and top-level review bodies.
- Treat package artifacts as part of the reproducibility surface; path portability and checklist accuracy matter for submission packages.

## Related Issues

- PR #14: `https://github.com/ankaggarwal94/qanta-buzzer/pull/14`
- Relevant commits:
  - `75c2483` — harden metric edge-case guards
  - `748d3df` — regenerate metric artifacts after guard fixes
  - `5c6efd3` — clean remaining checklist artifacts
- Regression coverage: `tests/test_pr14_review_regressions.py`
