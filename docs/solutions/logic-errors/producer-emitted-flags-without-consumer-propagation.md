---
title: Producer-Emitted Fallback Flags Without Consumer-Side Propagation
date: 2026-05-26
last_updated: 2026-05-27
category: logic-errors
module: qanta-buzzer audit pipeline
problem_type: logic_error
component: tooling
symptoms:
  - "Audit card reports overall_verdict PASS on a StopDFF run where every test question times out to the final prefix (median_abs_prefix_shift = 0.0 with no statistical power)"
  - "Audit card reports calibration PASS when a degenerate validation bucket falls back to ConstantCalibrationModel and ECE collapses to 0.0"
  - "Producer JSON artifacts carry the right fallback flags (ceiling_effect_detected, platt_model_type=constant, threshold_reachable=False) but the audit-card consumer ignores them"
  - "Verdict-cell markdown shows a bare PASS even though the producer detected and labeled a degenerate run"
root_cause: logic_error
resolution_type: code_fix
severity: high
related_components:
  - testing_framework
  - documentation
tags:
  - qanta-buzzer
  - audit-card
  - consumer-side-propagation
  - producer-consumer-contract
  - silent-pass
  - verdict-gate
  - stopdff
  - pr-review
related:
  - logic-errors/scientific-metric-edge-case-guards.md
  - architecture-patterns/cryptographic-artifact-provenance-with-runtime-verification.md
  - CS321M/docs/solutions/workflow-issues/pre-submission-artifact-consistency-audit.md
---

# Producer-Emitted Fallback Flags Without Consumer-Side Propagation

> Sibling to [`scientific-metric-edge-case-guards.md`](./scientific-metric-edge-case-guards.md).
> That doc is the producer-side post-mortem (detecting and labeling degenerate runs).
> This doc is the consumer-side post-mortem (acting on the labels rather than silently
> dropping them). Both halves are required for a defensive metric pipeline.

## Problem

The CS321M audit card consumer (`scripts/make_audit_card.py`) read only headline numerics and stored verdicts from upstream metric scripts, ignoring producer-emitted degeneracy flags. When `compute_stopdff.py` emitted `ceiling_effect_detected: true` (every question timed out to the final prefix so `median_abs_prefix_shift` was mechanically `0.0`) and `compute_prefix_calibration.py` emitted `platt_model_type: "constant"` (degenerate validation bucket where `compute_ece` returned `0.0`), the card still rendered an unqualified PASS verdict — exactly the "scientifically misleading PASS" case the producer-side fix had been written to prevent.

## Symptoms

- `paper_exports/audit_card.md` line 7 currently renders the qualifier the fix introduced:

```7:7:paper_exports/audit_card.md
| Diagnostic StopDFF (Median Abs Prefix Shift) | 0.0 | 1.0 | PASS (ceiling effect — diagnostic null; unreachable bucket(s): early, mid) |
```

  Pre-fix, the same row read just `PASS` even though the producer's ceiling detector had set `ceiling_effect_detected = True` ("no question in either condition stopped before the final prefix step" — `scripts/compute_stopdff.py:756`) and `reachability` flagged the `early` and `mid` buckets as unreachable.

- For calibration, an empty validation bucket flows through `_fit_bucket_calibrator` → `ConstantCalibrationModel` with metadata `{"platt_model_type": "constant", "platt_fallback_reason": "empty_validation_bucket"}`. `compute_ece` returns `0.0` for an empty array. The producer correctly stamps `platt_model_type: "constant"` and `n_samples: 0` into the per-bucket JSON. **The consumer ignored both fields and inherited `gate_verdict: "pass"`.**

- `_compute_overall_verdict` therefore returned `PASS` on the same run that two producers had flagged as degenerate.

- The card had no Data Provenance section, so a reviewer could not see at a glance that CSLI, calibration, and StopDFF agreed on what counted as a defensible retained-subset audit. After the fix, `paper_exports/audit_card.md` includes a per-metric per-split table showing coverage 100% / retention 72.7% with `overridden=yes` for all three metrics.

## What Didn't Work

The earlier compound — [`scientific-metric-edge-case-guards.md`](./scientific-metric-edge-case-guards.md) — fixed the producer side. The Platt fallback emits `platt_model_type: "constant"`; the StopDFF ceiling detector emits `ceiling_effect_detected: true`; the reachability check is correct in both coefficient signs. That doc even named the principle in its own Prevention section — *"Printing warnings without changing control flow was not enough."* — but it scoped the principle to the producer. The audit-card consumer was outside the field of view, so the very same anti-pattern recurred one layer up.

The deciding moment is captured in the prior session's own rationale (from the PR #14 babysit that landed `c1cd95c`):

> *"Since these code changes alter guard behavior and diagnostic metadata but not the already-reported numeric metrics, I'll avoid another full artifact regeneration unless verification shows a committed artifact must change."*

Reading the producer's degeneracy flags as *"diagnostic metadata only"* — rather than as a contract the audit card had to satisfy — is the move that left the gap live. The compound that followed that session focused on producer-side hardening; the audit-card consumer was never grepped for the new field names, so the silent-PASS path stayed open until the ChatGPT-5.5 Pro follow-up review caught it.

Three concrete temptations that fall short and were considered and rejected during the fix:

1. **Patch the headline numeric.** Tempting to write `if median_abs_prefix_shift == 0: verdict = "warn"` inside `_evaluate_stopdff`. This misses the cause — the metric is documented as `diagnostic_only` / `myopic_threshold`, so the gate definition `median_abs_prefix_shift <= threshold` is correct. What fails is interpretability, and the right gate is on the producer-emitted degeneracy flag, not the numeric.
2. **Producer-side schema test as "good enough".** A green producer-test on a silent consumer is the classic "schema-only" smell. `test_prefix_calibration_uses_constant_model_for_empty_val_bucket` pins the producer's return value but says nothing about whether `make_audit_card.py` *reads* the field — so the producer's regression was locked while the consumer's was wide open.
3. **Invert the verdict instead of qualifying.** Tempting to flip the StopDFF verdict from `"pass"` to `"fail"` when the ceiling fires. The implementation deliberately keeps the verdict and adds a qualifier — inverting would invalidate the already-submitted Phase 07 manuscript without changing the underlying metric semantics. Phase 06 documented StopDFF as a diagnostic-only metric and the null result is scientifically valid; what was missing was visibility, not invalidation.

A partial consumer fix in the same prior session (`scripts/regenerate_figures.py::_generate_audit_table`) created an asymmetry that made the gap easier to miss: figures got dynamic thresholds, but the JSON-of-record (`paper_exports/audit_card.json`) didn't get the qualifier fields, so a casual diff showed *"consumers updated"* without surfacing that `make_audit_card.py` was untouched.

## Solution

The fix wires the consumer to BOTH the headline numeric AND the producer-emitted degeneracy flags, renders a visible qualifier in the user-facing markdown, surfaces a Data Provenance section, and locks the propagation with consumer-side regression tests.

### 1. Consumer reads producer-emitted flags (calibration)

`_evaluate_calibration` scans per-bucket fallback metadata and force-overrides to `warn` when any bucket is degenerate:

```203:207:scripts/make_audit_card.py
    if fallback_buckets or empty_buckets:
        # Force WARN even if threshold-based ECE passes, because the
        # ECE is computed against a degenerate calibrator and/or
        # empty test bucket.
        computed_verdict = "warn"
```

The per-bucket scan reads `bucket.get("platt_model_type") == "constant"` and `bucket.get("n_samples") == 0` straight off the producer schema — the consumer now BRANCHES on the producer's contract.

**Round-11 tightening (commits `996718b`, `3b53f4c`):** the calibration producer (`compute_prefix_calibration.py`) now itself downgrades `gate_verdict` to `"warn"` on degenerate buckets and records a `gate_verdict_reason` field. The consumer's force-override above is preserved as the legacy-artifact fallback — when `stored_reason is not None` the consumer defers to the producer's verdict instead. The same producer-as-primary-source pattern was applied to `compute_stopdff.py` (downgrades `gate_verdict` on `ceiling_effect_detected` or any unreachable bucket, with matching `gate_verdict_reason`). This collapses the producer/consumer asymmetry: a single source of truth for the scientific verdict, with the audit card as the surface that renders qualifiers and propagates overall verdict downgrades.

### 2. Consumer reads producer-emitted flags (StopDFF)

`_evaluate_stopdff` reads `ceiling_effect_detected` and per-bucket `threshold_reachable` and constructs a `verdict_qualifier`:

```279:286:scripts/make_audit_card.py
    qualifier_parts = []
    if ceiling_effect:
        qualifier_parts.append("ceiling effect — diagnostic null")
    if unreachable_buckets:
        qualifier_parts.append(
            f"unreachable bucket(s): {', '.join(sorted(unreachable_buckets))}"
        )
    verdict_qualifier = "; ".join(qualifier_parts) if qualifier_parts else None
```

The `unreachable_buckets` list is computed by filtering `reachability.items()` on `info.get("threshold_reachable") is False` — again, a direct read of the producer's per-bucket contract.

### 3. Visible qualifier rendered in the markdown card

`_render_verdict_cell` folds the qualifier into the verdict column so the limitation surfaces in the headline, not buried in `details`:

```466:472:scripts/make_audit_card.py
def _render_verdict_cell(m: dict) -> str:
    """Render a metric's verdict cell, including any PR-14-B2 qualifier."""
    base = m["verdict"].upper()
    qualifier = m.get("verdict_qualifier")
    if qualifier:
        return f"{base} ({qualifier})"
    return base
```

The resulting cell at `paper_exports/audit_card.md:7` is the user-visible artifact cited in Symptoms above.

### 4. Provenance section surfaces gate state per metric per split

`_extract_data_provenance` walks each metric's `mc_coverage` and `mc_retention_gate` blocks — handling both the legacy flat CSLI shape and the new nested calibration/StopDFF shape. `_render_data_provenance_md` emits the per-metric per-split table now visible in `paper_exports/audit_card.md`, so reviewers can see all three metrics ran the same gate (current run: coverage 100% / retention 72.7% / `overridden=yes`).

### 5. Regression tests pin the CONSUMER, not the producer

Each propagation field gets a synthetic-payload test that constructs the field set and asserts the consumer reacts. The Blocker 4 calibration test is the canonical pattern:

```632:639:tests/test_pr14_review_regressions.py
    metric = make_audit_card._evaluate_calibration(cal_data, threshold=0.10)

    assert metric["verdict"] == "warn"
    assert metric["details"]["fallback_buckets"][0]["bucket"] == "mid"
    assert metric["details"]["fallback_buckets"][0]["reason"] == (
        "empty_validation_bucket"
    )
    assert "mid" in metric["details"]["empty_buckets"]
```

The B2 mirror ("must add a qualifier when `ceiling_effect_detected`") lives at `tests/test_pr14_review_regressions.py:225`, and the markdown-render lock ("card cell reads `PASS (ceiling effect — diagnostic null)`") at `tests/test_pr14_review_regressions.py:314`. All three are constructed against synthetic upstream payloads so the consumer's decision logic is tested in isolation from the upstream compute path.

## Why This Works

The fix doesn't just patch B2 and B4 individually; it encodes a general producer/consumer contract principle for scientific audit pipelines:

- **Every defensive metadata field emitted by a producer is a CONTRACT with downstream consumers.** When a producer adds `platt_model_type: "constant"` to its schema, that field exists to be *read*, not just *stored*.
- **The producer's job is to detect and label the degeneracy.**
- **The consumer's job is to BRANCH on the label, not just on the headline metric.** Without consumer-side propagation, the producer's defensive flag is invisible to the gate — a silent regression where the producer correctly screams "degenerate!" into a log line that gets discarded by every downstream tool.
- **Regression tests must pin the contract at the consumer layer, not just the producer layer.** Tests at the producer prove schema correctness; tests at the consumer prove decision correctness. The PR-14 babysit closed only after both layers were covered.

This principle generalizes to all six PR-14 follow-up blockers, every one of which is the same shape — "make the producer's safety contract visible at the consumer":

- **B1** surfaces both CSLI flavors in `details` (pre-fix, the card hid the PAP-original choices-only excess behind the gap flavor).
- **B3** surfaces shared coverage/retention provenance per metric per split (pre-fix, a reviewer could not tell whether all three metrics agreed on what counted as a defensible retained-subset audit).
- **B5** raises an actionable `KeyError` ("re-run build_mc_dataset.py") at the consumer when the producer wrote TOSSUP-only rows instead of `TypeError`-ing on dict iteration.
- **B6** raises explicitly on empty/NaN belief-math input instead of silently returning NaN.

This is also the design pattern that [`CS321M/docs/solutions/design-patterns/distinguishable-defensive-fallbacks-2026-05-18.md`](../../../../../CS321M/final_project/docs/solutions/design-patterns/distinguishable-defensive-fallbacks-2026-05-18.md) cataloged at the *value level* (encoded offset, side-channel, re-raise, producer-side round-trip). The audit-card recurrence is a fifth shape: the producer emits a structured *flag* alongside the value, and the consumer must read the flag to decide. Both shapes share the same root: a defensive fallback at one layer must be *distinguishable* at every downstream layer that makes a decision on top of it.

## Prevention

Concrete checklist for next time:

1. **Grep every downstream consumer when adding a metadata field.** When `compute_prefix_calibration.py` gained `platt_model_type`, the right next step is `rg 'platt_model_type' scripts/ tests/` — if no downstream `if`/`elif`/`switch` on the *value* exists, the field is silent. This is the cheap mechanical check that would have caught the original gap.
2. **Treat the consumer's verdict logic as the LOAD-BEARING gate, not the producer's.** Producer tests prove schema correctness; consumer tests prove decision correctness. Both layers must be green before the contract is real.
3. **Wire a qualifier-rendering path when emitting `*_detected: bool` or `*_model_type: "constant"`.** Every defensive flag should map to a corresponding qualifier in the user-facing artifact, not just a JSON field. The pattern is `_render_verdict_cell` (`scripts/make_audit_card.py:466`): inspect the flag, decorate the headline.
4. **Use the "schema-only" smell check.** If a PR adds a new field without a test that asserts a downstream verdict *change* (not just shape/presence), the field is almost certainly silent.
5. **Surface gate state in a visible Data Provenance section.** Fold producer-emitted coverage, retention, and fallback markers into a card-level table so reviewers can verify at a glance that all metrics ran the same gate.
6. **Cross-audit downstream consumers when a producer-side compound lands.** Whenever `ce-compound` documents a producer-side hardening, scan every downstream consumer for fields the new doc names and assert that each one is read. Use the rationale "altered diagnostic metadata only" as a smell, not as a license to skip the consumer audit — that exact phrase preceded this gap.

### Code example: consumer-side propagation test pattern

The template a new metadata field must satisfy before merge. First, construct a synthetic upstream payload with the degeneracy flag set:

```616:622:tests/test_pr14_review_regressions.py
            "mid": {
                "ece": 0.0,
                "n_samples": 0,
                "platt_model_type": "constant",
                "platt_fallback_reason": "empty_validation_bucket",
                "platt_constant_probability": 0.0,
            },
```

Then assert the consumer's *decision* flips on the flag:

```632:639:tests/test_pr14_review_regressions.py
    metric = make_audit_card._evaluate_calibration(cal_data, threshold=0.10)

    assert metric["verdict"] == "warn"
    assert metric["details"]["fallback_buckets"][0]["bucket"] == "mid"
    assert metric["details"]["fallback_buckets"][0]["reason"] == (
        "empty_validation_bucket"
    )
    assert "mid" in metric["details"]["empty_buckets"]
```

The test isolates the propagation contract from the upstream compute path: no real producer is run, no real ECE is computed, only the consumer's reaction to the producer-emitted field is exercised. This is the right shape for *any* future `*_detected` / `*_model_type: constant` / `*_fallback_reason` field added to a producer JSON.

## Related

- Producer-side sibling (this doc's primary cross-link): [`scientific-metric-edge-case-guards.md`](./scientific-metric-edge-case-guards.md). That doc documents the producer-side hardening (Platt fallback, ceiling detection, reachability check); this doc documents the consumer-side that was previously omitted. Read both together.
- Third-axis sibling (artifact-source integrity, not flag semantics): [`../architecture-patterns/cryptographic-artifact-provenance-with-runtime-verification.md`](../architecture-patterns/cryptographic-artifact-provenance-with-runtime-verification.md). This doc covers *semantic* propagation (the producer's flag must be read by the consumer); that doc covers *source* propagation (the consumer must verify the producer artifact came from the producer script currently on disk via embedded `script_sha256` + `git_commit`). The two together form the full producer/consumer trust contract for a multi-script scientific audit pipeline.
- Design-pattern parent at the value level: `CS321M/docs/solutions/design-patterns/distinguishable-defensive-fallbacks-2026-05-18.md`. The audit-card recurrence is the "fifth shape" of that pattern (producer emits a structured flag, consumer must read it).
- Producer-consumer round-trip verification precedent: `CS321M/docs/solutions/runtime-errors/silent-ncf-head-load-failure-via-full-module-pickle-2026-05-17.md`.
- Audit-all-paths-not-just-except prevention precedent: `CS321M/docs/solutions/runtime-errors/labeling-py-nan-fallback-bomb-2026-05-17.md`.
- Sibling propagation gap at the manuscript/package layer (not the JSON layer): `CS321M/docs/solutions/workflow-issues/pre-submission-artifact-consistency-audit.md`. Treat the two docs as a pair: when a metric script changes, audit the JSON-card consumer (`make_audit_card.py`) AND the prose/figure/checksum bundle.
- Orchestration parallel — the Evidence-Lane Review pattern that surfaced this gap is the CoVe v2/v3 family applied to a code-review document: `CS321M/docs/solutions/architecture-patterns/cove-three-wave-validation-pipeline-2026-05-17.md`, `cove-v3-iterative-re-review-and-self-containment-2026-05-18.md`.

## References

- PR: [qanta-buzzer #14](https://github.com/ankaggarwal94/qanta-buzzer/pull/14)
- Trigger commits: `41e15b7` (B1-B6 code), `6d33c8e` (regression tests), `eb8e337` (regenerated artifacts + audit card)
- Producer-side precursor commits: `bcd99ff` (Platt class-balance check), `4d169c2` (StopDFF reachability check), `15097cd` (`scientific-metric-edge-case-guards.md`)
