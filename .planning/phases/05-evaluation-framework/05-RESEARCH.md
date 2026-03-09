# Phase 5: Evaluation Framework - Research

**Researched:** 2026-02-26
**Domain:** Evaluation metrics and visualization for RL buzzer agents
**Confidence:** HIGH

## Summary

This is NOT a research-from-scratch phase. Phase 4 already implemented substantial evaluation infrastructure: `evaluation/metrics.py` (S_q, ECE, Brier), `evaluation/controls.py` (3 control experiments), `evaluation/plotting.py` (entropy, calibration, comparison tables), and `scripts/evaluate_all.py` (orchestration). The research task is to AUDIT what exists against EVAL-01 through EVAL-07 requirements and identify specific GAPS that remain.

**Critical finding:** Most evaluation requirements are ALREADY SATISFIED. The main gaps are:
1. **Per-category accuracy breakdown (EVAL-07)** — completely missing, needs new function
2. **Comparison table completeness (EVAL-06)** — current implementation only includes 4 metrics, needs baseline sweep results added
3. **S_q computation edge cases (EVAL-01)** — implementation correct but lacks explicit tests for edge cases

**Primary recommendation:** Focus Phase 5 on filling the 3 specific gaps above. Do NOT rebuild existing infrastructure. Add per-category accuracy function, enhance comparison table with baseline metrics, and add tests for S_q edge cases.

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|-----------------|
| EVAL-01 | S_q metric computation: system score = Σ(b_t × g_t) per episode | SATISFIED by evaluation/metrics.py system_score() — implementation matches specification exactly |
| EVAL-02 | Calibration metrics: ECE (expected calibration error) and Brier score | SATISFIED by evaluation/metrics.py expected_calibration_error() and brier_score() |
| EVAL-03 | Control experiment: choices-only (remove clues, verify ~25% random baseline) | SATISFIED by evaluation/controls.py run_choices_only_control() using logistic regression on surface features |
| EVAL-04 | Control experiment: shuffle (permute option order, verify no position bias) | SATISFIED by evaluation/controls.py run_shuffle_control() |
| EVAL-05 | Control experiment: alias substitution (swap answer text, verify robustness) | SATISFIED by evaluation/controls.py run_alias_substitution_control() |
| EVAL-06 | Comparison plots: calibration curves, entropy vs clue index, agent comparison tables | PARTIAL — entropy/calibration plots exist in evaluation/plotting.py, but comparison table (save_comparison_table) only includes 4 agents (softmax, shuffle, alias, ppo) and limited metrics. Missing baseline sweep results. |
| EVAL-07 | Per-category accuracy breakdown with summary statistics | GAP — completely missing. Need new function to group results by MCQuestion.category and compute accuracy/S_q per category |
</phase_requirements>

## Standard Stack

### Core (Already Installed)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| NumPy | <2.0.0 | Array operations | Universal scientific computing standard, used for metric computation |
| pandas | latest | Data manipulation | Standard for tabular data, used in comparison table generation |
| matplotlib | latest | Plotting | De facto Python visualization library for academic figures |
| seaborn | latest | Statistical plots | High-level interface over matplotlib, better defaults |
| scikit-learn | latest | Control experiments | Logistic regression for choices-only control, TfidfVectorizer |

### Supporting
No new dependencies required. All libraries already installed in Phase 4.

**Installation:**
Already complete — Phase 4 installed all required packages via `pip install stable-baselines3 matplotlib seaborn pandas`.

## Architecture Patterns

### Existing Architecture (Do NOT Rebuild)
```
evaluation/
├── metrics.py          # COMPLETE: system_score, ECE, Brier, summarize_buzz_metrics, calibration_at_buzz
├── controls.py         # COMPLETE: 3 control experiments + bootstrap_ci
├── plotting.py         # MOSTLY COMPLETE: entropy, calibration, comparison table (needs enhancement)
└── __init__.py

scripts/
└── evaluate_all.py     # ORCHESTRATION: loads data, runs controls, generates plots
```

### Pattern 1: Metric Computation (Existing)
**What:** Pure functions that take traces/results and return scalar metrics
**Implementation:** `evaluation/metrics.py` already implements this pattern
**Example:**
```python
def system_score(c_trace: list[float], g_trace: list[float]) -> float:
    """Compute S_q = sum_t b_t * g_t where b_t = c_t * prod_{i<t} (1 - c_i)."""
    c = np.array(c_trace, dtype=np.float64)
    g = np.array(g_trace, dtype=np.float64)
    b = np.zeros_like(c)
    survival = 1.0
    for t in range(len(c)):
        b[t] = c[t] * survival
        survival *= (1.0 - c[t])
    return float(np.sum(b * g))
```
**Status:** COMPLETE for S_q, ECE, Brier

### Pattern 2: Control Experiments (Existing)
**What:** Transform questions, evaluate with same evaluator function, compare results
**Implementation:** `evaluation/controls.py` implements 3 controls
**Pattern:**
```python
def run_control(questions, evaluator, transform_fn):
    transformed = [transform_fn(q) for q in questions]
    return evaluator(transformed)
```
**Status:** COMPLETE for 3 controls (choices-only, shuffle, alias)

### Pattern 3: Per-Group Analysis (NEW — EVAL-07)
**What:** Group results by a category field, compute metrics per group
**Where to add:** New function in `evaluation/metrics.py`
**Pattern:**
```python
def per_category_accuracy(results: list[Any]) -> dict[str, dict[str, float]]:
    """Group results by category and compute accuracy, S_q, buzz_step per category.

    Returns dict[category_name, {accuracy, mean_sq, mean_buzz_step, n}]
    """
    from collections import defaultdict
    by_category = defaultdict(list)
    for r in results:
        category = r.get("category", "unknown")
        by_category[category].append(r)

    return {
        cat: summarize_buzz_metrics(rows)
        for cat, rows in by_category.items()
    }
```
**Status:** NEEDS IMPLEMENTATION

### Pattern 4: Comparison Table Enhancement (EXISTING, NEEDS EXPANSION)
**What:** Aggregate all agent results into single comparison CSV/markdown
**Current:** `evaluation/plotting.py save_comparison_table()` exists but only includes 4 agents
**Enhancement needed:** Add baseline sweep results (threshold, softmax_profile at multiple thresholds)
**Where:** Modify `scripts/evaluate_all.py` to load `baseline_summary.json` and add rows to table

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Category grouping | Custom loops with if/else | `collections.defaultdict` or `pandas.groupby` | Edge cases (missing categories, empty groups) already handled |
| Confidence intervals | Manual bootstrap resampling | `evaluation/controls.py bootstrap_ci()` already exists | Reproducible seeding, quantile computation tested |
| Binned calibration | Manual histogram bucketing | `evaluation/metrics.py expected_calibration_error()` already exists | Handles edge cases (empty bins, boundary conditions) |
| Surface feature extraction | Ad-hoc string parsing | `evaluation/controls.py _option_scalar_features()` already exists | Tested feature set (length, parens, commas, capitalization) |

**Key insight:** Phase 4 built most utilities needed. Don't reimplement, extend what exists.

## Common Pitfalls

### Pitfall 1: Category Field Not Always Present
**What goes wrong:** MCQuestion.category may be empty string or None for some questions
**Why it happens:** CSV data may have missing category values
**How to avoid:** Default to "unknown" category in per_category_accuracy, handle empty groups
**Warning signs:** KeyError or empty dict returned from grouping operation

### Pitfall 2: Comparison Table Overwrites Existing Columns
**What goes wrong:** Adding baseline sweep results may duplicate column names (e.g., "accuracy")
**Why it happens:** Each agent dict has overlapping metric names
**How to avoid:** Prefix metrics with agent name (e.g., "threshold_0.5_accuracy") or use nested structure
**Warning signs:** DataFrame has fewer columns than expected, values overwritten

### Pitfall 3: Per-Category Metrics on Small Samples
**What goes wrong:** Categories with 2-3 questions have 0% or 100% accuracy (high variance)
**Why it happens:** Stratified splits preserve proportions, but some categories are rare
**How to avoid:** Report n per category, add confidence intervals, flag categories with n < 10
**Warning signs:** Category accuracies are all 0.0 or 1.0, no intermediate values

### Pitfall 4: Episode Results Missing Category Field
**What goes wrong:** per_category_accuracy() receives episode results without category attribute
**Why it happens:** Episode results are dicts/dataclasses that don't include question metadata
**How to avoid:** Modify agent.run_episode() to include q.category in returned dict, or join results with original questions
**Warning signs:** All results grouped under "unknown" category

### Pitfall 5: S_q Edge Cases Not Tested
**What goes wrong:** S_q computation fails on empty traces or all-zero confidence
**Why it happens:** Current implementation lacks explicit tests for edge cases
**How to avoid:** Add test cases for len(c_trace)==0, all c==0, all c==1, single-step episodes
**Warning signs:** Division by zero, NaN results, incorrect scores on trivial cases

## Code Examples

Verified patterns from existing codebase:

### Per-Category Accuracy (NEW)
```python
# Source: To be added to evaluation/metrics.py
from collections import defaultdict
from typing import Any

def per_category_accuracy(
    results: list[Any],
    questions: list[MCQuestion],
) -> dict[str, dict[str, float]]:
    """Compute accuracy and S_q metrics grouped by question category.

    Joins results with questions to extract category field, then groups
    and computes summarize_buzz_metrics per category.

    Parameters
    ----------
    results : list[Any]
        Episode results from agent evaluation (dicts or dataclasses).
        Must have qid field for joining.
    questions : list[MCQuestion]
        Original questions with category field.

    Returns
    -------
    dict[str, dict[str, float]]
        Mapping from category name to metrics dict with keys:
        n, buzz_accuracy, mean_buzz_step, mean_sq, mean_reward_like.
    """
    # Build qid -> category lookup
    qid_to_category = {q.qid: q.category if q.category else "unknown" for q in questions}

    # Group results by category
    by_category = defaultdict(list)
    for r in results:
        r_dict = _to_dict(r)
        qid = r_dict.get("qid", "")
        category = qid_to_category.get(qid, "unknown")
        by_category[category].append(r)

    # Compute metrics per category
    return {
        cat: summarize_buzz_metrics(rows)
        for cat, rows in by_category.items()
    }
```

### Enhanced Comparison Table (MODIFY)
```python
# Source: Modify scripts/evaluate_all.py main()
# After loading baseline_summary.json:
baseline_summary = load_json(baseline_summary_path) if baseline_summary_path.exists() else {}

# Build table rows including baseline sweep
table_rows = [
    {"agent": "softmax_profile", **{k: v for k, v in full_eval.items() if k != "runs"}},
    {"agent": "shuffle_control", **{k: v for k, v in shuffle_eval.items() if k != "runs"}},
    {"agent": "alias_control", **{k: v for k, v in alias_eval.items() if k != "runs"}},
]

# Add baseline sweep results (threshold at multiple values)
if "threshold" in baseline_summary:
    for threshold_str, metrics in baseline_summary["threshold"].items():
        table_rows.append({"agent": f"threshold_{threshold_str}", **metrics})

# Add softmax_profile sweep results
if "softmax_profile" in baseline_summary:
    for threshold_str, metrics in baseline_summary["softmax_profile"].items():
        table_rows.append({"agent": f"softmax_{threshold_str}", **metrics})

if ppo_summary:
    table_rows.append({"agent": "ppo", **ppo_summary})

save_comparison_table(table_rows, out_dir / "plots" / "comparison.csv")
```

### S_q Edge Case Tests (NEW)
```python
# Source: To be added to tests/test_metrics.py
def test_system_score_empty_trace():
    """S_q should return 0.0 for empty traces."""
    assert system_score([], []) == 0.0

def test_system_score_all_zero_confidence():
    """S_q should return 0.0 when agent never considers buzzing."""
    c_trace = [0.0, 0.0, 0.0]
    g_trace = [1.0, 1.0, 1.0]  # All correct but agent doesn't buzz
    assert system_score(c_trace, g_trace) == 0.0

def test_system_score_all_correct_immediate_buzz():
    """S_q should equal first g_trace value when agent buzzes immediately."""
    c_trace = [1.0, 0.0, 0.0]  # Buzz on step 0
    g_trace = [1.0, 1.0, 1.0]
    expected = 1.0 * 1.0  # b_0 = c_0 * 1.0 = 1.0, survival after = 0
    assert abs(system_score(c_trace, g_trace) - expected) < 1e-9

def test_system_score_gradual_confidence():
    """S_q should accumulate survival-weighted correctness."""
    c_trace = [0.3, 0.5, 1.0]
    g_trace = [0.0, 0.0, 1.0]  # Only correct at final step
    # b_0 = 0.3 * 1.0 = 0.3, survival = 0.7
    # b_1 = 0.5 * 0.7 = 0.35, survival = 0.7 * 0.5 = 0.35
    # b_2 = 1.0 * 0.35 = 0.35
    # S_q = 0.3*0 + 0.35*0 + 0.35*1 = 0.35
    expected = 0.35
    assert abs(system_score(c_trace, g_trace) - expected) < 1e-9
```

## State of the Art

Phase 4 already implements current best practices:

| Aspect | Phase 4 Implementation | Standard |
|--------|----------------------|----------|
| S_q computation | Explicit survival probability tracking | Matches quiz bowl RL literature (qb-rl reference) |
| ECE | Uniform binning with 10 bins | Standard calibration metric (Guo et al. 2017) |
| Brier score | Mean squared error | Original Brier (1950) formulation |
| Control experiments | 3 types (choices-only, shuffle, alias) | Best practice for artifact detection |
| Bootstrap CI | 1000 resamples, 95% CI | Standard non-parametric confidence intervals |
| Matplotlib Agg backend | Non-interactive for headless | Standard for CI/server environments |

**Deprecated/outdated:**
- Manual histogram binning: Use `expected_calibration_error()` instead
- String-based category filtering: Use pandas groupby for efficiency

## Validation Architecture

> workflow.nyquist_validation is false in .planning/config.json — skip test requirements

Since `workflow.nyquist_validation` is not enabled, test infrastructure is optional. However, given the critical nature of S_q metric computation, recommend adding basic unit tests for edge cases.

### Recommended Test Coverage (Optional)
- `tests/test_metrics.py` — Unit tests for S_q, ECE, Brier edge cases (empty, all-zero, boundary)
- `tests/test_per_category.py` — Test per_category_accuracy with synthetic data (various category distributions)

### Quick Validation Commands
```bash
# Run evaluation on smoke dataset
python scripts/evaluate_all.py --smoke

# Check per-category breakdown in generated JSON
python -c "import json; print(json.load(open('artifacts/smoke/evaluation_report.json'))['per_category'])"

# Verify comparison table includes all agents
wc -l artifacts/smoke/plots/comparison.csv  # Should be > 4 (header + 4 existing agents + baseline sweep)
```

## Open Questions

1. **Should per-category breakdown include confidence intervals?**
   - What we know: bootstrap_ci() function exists in controls.py
   - What's unclear: Whether CS234 writeup needs statistical rigor at category level
   - Recommendation: Start without CI, add if time permits and categories have reasonable sample sizes (n > 10)

2. **How to handle categories with n < 5 questions?**
   - What we know: Stratified splits preserve proportions, some categories rare
   - What's unclear: Whether to merge small categories or flag as insufficient data
   - Recommendation: Report all categories with n, add warning footnote for n < 5

3. **Should comparison table show confidence intervals?**
   - What we know: bootstrap_ci() function exists, Phase 4 doesn't use it in table
   - What's unclear: Whether additional columns would clutter comparison table
   - Recommendation: Defer to Phase 6 stretch goal, core table shows point estimates only

## Sources

### Primary (HIGH confidence)
- Phase 4 implementation files: evaluation/metrics.py (lines 1-247), evaluation/controls.py (lines 1-351), evaluation/plotting.py (lines 1-191), scripts/evaluate_all.py (lines 1-319)
- Phase 4 SUMMARY.md: Confirmed existing infrastructure works end-to-end in smoke test
- REQUIREMENTS.md: EVAL-01 through EVAL-07 specifications
- MCQuestion dataclass (qb_data/mc_builder.py): Confirmed category field exists (inherited from TossupQuestion)

### Secondary (MEDIUM confidence)
- qb-rl reference implementation: Original source for S_q computation, control experiments
- Calibration metrics literature: ECE binning standard (10 bins uniform), Brier score formulation

## Metadata

**Confidence breakdown:**
- Existing infrastructure: HIGH - Phase 4 files read directly, smoke test confirmed working
- Gaps identification: HIGH - Systematic audit of EVAL-01 through EVAL-07 against implemented functions
- Per-category implementation: MEDIUM - Pattern is standard (groupby), but category field handling needs care
- Comparison table enhancement: HIGH - Straightforward modification to existing code

**Research date:** 2026-02-26
**Valid until:** 2026-03-05 (7 days — stable codebase, no external dependencies)

**Key finding:** This is a GAP-FILLING phase, not new development. Focus on 3 specific gaps rather than rebuilding.
