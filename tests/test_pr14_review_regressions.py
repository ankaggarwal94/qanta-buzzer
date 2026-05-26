"""Regression tests for active PR #14 review findings."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.modules.pop("evaluation.controls", None)
sys.modules.pop("scripts.compute_csli", None)

from scripts import compute_csli, compute_prefix_calibration
from scripts.compute_stopdff import (
    check_threshold_reachability,
    load_platt_coefficients,
)

_filter_test_mc_questions = getattr(compute_csli, "_filter_test_mc_questions")
_fit_bucket_calibrator = getattr(
    compute_prefix_calibration,
    "_fit_bucket_calibrator",
)
_calibrate_bucket_scores = getattr(
    compute_prefix_calibration,
    "_calibrate_bucket_scores",
)


def test_csli_coverage_uses_unique_matched_qids_not_row_count() -> None:
    """Duplicate MC rows for one qid must not hide another missing qid."""
    mc_questions = [
        {"qid": "q1", "question": "first copy"},
        {"qid": "q1", "question": "duplicate copy"},
        {"qid": "outside-test", "question": "not in test"},
    ]

    questions, coverage = _filter_test_mc_questions(mc_questions, {"q1", "q2"})

    assert len(questions) == 2
    assert coverage["matched_test_mc_questions"] == 2
    assert coverage["matched_test_mc_qids"] == 1
    assert coverage["missing_qids"] == {"q2"}
    assert coverage["coverage_rate"] == 0.5


def test_prefix_calibration_uses_constant_model_for_empty_val_bucket() -> None:
    """Empty validation buckets should not read labels[0] or fit Platt."""
    model, metadata = _fit_bucket_calibrator(
        "early",
        np.array([], dtype=float),
        np.array([], dtype=int),
    )

    calibrated = _calibrate_bucket_scores(model, np.array([0.1, 0.2]))

    assert metadata["platt_model_type"] == "constant"
    assert metadata["platt_fallback_reason"] == "empty_validation_bucket"
    assert calibrated.tolist() == [0.0, 0.0]


def test_prefix_calibration_uses_constant_model_for_single_class_bucket() -> None:
    """Single-class validation buckets should not call LogisticRegression.fit."""
    model, metadata = _fit_bucket_calibrator(
        "mid",
        np.array([0.1, 0.2, 0.3], dtype=float),
        np.array([1, 1, 1], dtype=int),
    )

    calibrated = _calibrate_bucket_scores(model, np.array([0.4, 0.5]))

    assert metadata["platt_model_type"] == "constant"
    assert metadata["platt_fallback_reason"] == "single_class_validation_bucket"
    assert calibrated.tolist() == [1.0, 1.0]


def test_prefix_calibration_empty_test_bucket_returns_empty_scores() -> None:
    """Empty test buckets should skip predict_proba and produce no samples."""
    model, metadata = _fit_bucket_calibrator(
        "late",
        np.array([0.1, 0.2, 0.3], dtype=float),
        np.array([0, 1, 1], dtype=int),
    )

    calibrated = _calibrate_bucket_scores(model, np.array([], dtype=float))

    assert metadata["platt_model_type"] == "logistic"
    assert calibrated.shape == (0,)


def test_stopdff_reachability_uses_negative_raw_bound_for_negative_coef() -> None:
    """For negative Platt coefficients, max probability occurs at cosine=-1."""
    reachability = check_threshold_reachability(
        {"early": (-2.0, 0.0)},
        threshold=0.7,
    )

    early = reachability["early"]
    assert early["max_calibrated_raw_score"] == -1.0
    assert early["max_calibrated_probability"] > 0.7
    assert early["threshold_reachable"] is True
    assert -1.0 <= early["required_raw_score"] <= 1.0


def test_stopdff_loads_constant_calibration_bucket(tmp_path: Path) -> None:
    """StopDFF should consume constant calibration fallback buckets safely."""
    calibration_path = tmp_path / "calibration.json"
    calibration_path.write_text(
        """
{
  "per_bucket": {
    "early": {
      "platt_coef": null,
      "platt_intercept": null,
      "platt_model_type": "constant",
      "platt_constant_probability": 1.0
    }
  }
}
""".strip(),
        encoding="utf-8",
    )

    params = load_platt_coefficients(calibration_path)

    assert params["early"] == (0.0, 500.0)


# ============================================================================
# PR #14 follow-up review (ChatGPT-5.5 Pro): Blockers 1-6
# ============================================================================
#
# These tests cover the verdict-propagation and defensive-guard gaps the
# follow-up review surfaced after the first PR babysit cycle closed.
# Together with the existing tests above, they pin both the producer-side
# schema and the audit-card-consumer-side propagation that the reviewer
# argued were the load-bearing missing pieces.

import pytest

from agents._math import bayesian_update, softmax_belief
from scripts import make_audit_card
from scripts._audit_gates import (
    build_coverage_metadata,
    build_retention_metadata,
    coverage_gate_decision,
    filter_mc_questions_to_split,
    load_mc_build_metadata,
    metadata_retention_threshold,
    retention_gate_decision,
)
from scripts._common import iter_split_questions, load_mc_questions


# ----- Blocker 1: CSLI dual-flavor reporting ----------------------------------


def test_audit_card_csli_surfaces_both_gap_and_excess_flavors() -> None:
    """The audit card details must report both gap and PAP-original excess.

    Reviewer claim (Blocker 1): the audit card only displayed the gap
    flavor, hiding the PAP-original "choices-only excess over chance"
    interpretation. The fix surfaces both in ``details``, with explicit
    definition notes so a reader can interpret the audit card without
    cross-referencing csli.json.
    """
    csli_data = {
        "panel_csli": {
            "mean": 0.10,
            "ci_lower": 0.05,
            "ci_upper": 0.15,
        },
        "per_model": {
            "tfidf": {"acc_choices_only": 0.260, "leakage_flag": False},
            "sbert": {"acc_choices_only": 0.244, "leakage_flag": False},
            "t5-small": {"acc_choices_only": 0.214, "leakage_flag": False},
        },
    }

    metric = make_audit_card._evaluate_csli(csli_data, threshold=0.30)

    details = metric["details"]
    assert "panel_csli_gap" in details
    assert "panel_csli_choices_excess" in details
    assert "per_model_csli_choices_excess" in details
    # TF-IDF: max(0, 0.260 - 0.25) = 0.010; SBERT: 0; T5-small: 0
    excess = details["per_model_csli_choices_excess"]
    assert excess["tfidf"] == pytest.approx(0.010, abs=1e-6)
    assert excess["sbert"] == pytest.approx(0.0, abs=1e-6)
    assert excess["t5-small"] == pytest.approx(0.0, abs=1e-6)
    # Definition notes must be present so a reader can disambiguate
    assert "panel_csli_gap_definition" in details
    assert "panel_csli_choices_excess_definition" in details
    assert "definition_note" in details


def test_audit_card_csli_reads_excess_from_artifact_when_present() -> None:
    """When csli.json already carries the excess block, the card prefers it."""
    csli_data = {
        "panel_csli": {"mean": 0.10, "ci_lower": 0.05, "ci_upper": 0.15},
        "panel_csli_choices_excess": {
            "mean_from_per_model_avg": 0.0333,
            "definition": "...",
            "K": 4,
            "chance": 0.25,
        },
        "per_model": {
            "tfidf": {"acc_choices_only": 0.260, "leakage_flag": False},
            "sbert": {"acc_choices_only": 0.244, "leakage_flag": False},
            "t5-small": {"acc_choices_only": 0.214, "leakage_flag": False},
        },
    }

    metric = make_audit_card._evaluate_csli(csli_data, threshold=0.30)

    assert metric["details"]["panel_csli_choices_excess"] == pytest.approx(
        0.0333, abs=1e-4
    )


# ----- Blocker 2: StopDFF ceiling-effect qualifier ----------------------------


def test_audit_card_stopdff_adds_qualifier_when_ceiling_detected() -> None:
    """When ceiling_effect_detected, the verdict must carry a qualifier.

    Reviewer claim (Blocker 2): a degenerate StopDFF run where every
    question times out to the final prefix produces median_abs_prefix_shift
    = 0, which mechanically passes the threshold. Without a qualifier, the
    audit card reports "PASS" with no power. The fix keeps the verdict
    (the metric is documented as diagnostic_only / myopic_threshold) but
    adds a verdict_qualifier so the card renders ``PASS (ceiling effect
    -- diagnostic null)`` in the verdict column.
    """
    stopdff_data = {
        "median_abs_prefix_shift": 0.0,
        "gate_verdict": "pass",
        "direction_breakdown": {
            "mc_stops_earlier": 0,
            "nonmc_stops_earlier": 0,
            "same_step": 2258,
        },
        "ceiling_effect_detected": True,
        "reachability": {
            "early": {"threshold_reachable": False, "max_calibrated_probability": 0.51},
            "mid": {"threshold_reachable": False, "max_calibrated_probability": 0.41},
            "late": {"threshold_reachable": True, "max_calibrated_probability": 0.82},
        },
        "metadata": {"metric_type": "diagnostic_only"},
    }

    metric = make_audit_card._evaluate_stopdff(stopdff_data, threshold=1)

    # Verdict itself is preserved (the diagnostic test passes its threshold)
    assert metric["verdict"] == "pass"
    # But a qualifier exists and mentions both the ceiling effect and the
    # unreachable buckets so the card reader can see the limitation
    assert metric["verdict_qualifier"] is not None
    assert "ceiling effect" in metric["verdict_qualifier"]
    assert "early" in metric["verdict_qualifier"]
    assert "mid" in metric["verdict_qualifier"]
    # Details surface the raw flags for downstream consumers
    assert metric["details"]["ceiling_effect_detected"] is True
    assert set(metric["details"]["unreachable_buckets"]) == {"early", "mid"}


def test_audit_card_stopdff_no_qualifier_when_no_ceiling() -> None:
    """Healthy StopDFF runs should not get a qualifier."""
    stopdff_data = {
        "median_abs_prefix_shift": 0.5,
        "gate_verdict": "pass",
        "direction_breakdown": {
            "mc_stops_earlier": 100,
            "nonmc_stops_earlier": 100,
            "same_step": 2058,
        },
        "ceiling_effect_detected": False,
        "reachability": {
            "early": {"threshold_reachable": True, "max_calibrated_probability": 0.82},
            "mid": {"threshold_reachable": True, "max_calibrated_probability": 0.82},
            "late": {"threshold_reachable": True, "max_calibrated_probability": 0.82},
        },
        "metadata": {"metric_type": "diagnostic_only"},
    }

    metric = make_audit_card._evaluate_stopdff(stopdff_data, threshold=1)

    assert metric["verdict"] == "pass"
    assert metric["verdict_qualifier"] is None


def test_audit_card_stopdff_legacy_json_no_qualifier() -> None:
    """Older stopdff.json without ceiling/reachability fields stays uncluttered."""
    stopdff_data = {
        "median_abs_prefix_shift": 0.0,
        "gate_verdict": "pass",
        "direction_breakdown": {
            "mc_stops_earlier": 0,
            "nonmc_stops_earlier": 0,
            "same_step": 100,
        },
        "metadata": {"metric_type": "diagnostic_only"},
    }

    metric = make_audit_card._evaluate_stopdff(stopdff_data, threshold=1)

    assert metric["verdict"] == "pass"
    assert metric["verdict_qualifier"] is None
    assert metric["details"]["ceiling_effect_detected"] is False
    assert metric["details"]["unreachable_buckets"] == []


def test_audit_card_md_renders_stopdff_qualifier_in_verdict_cell() -> None:
    """The markdown card must render the qualifier alongside the verdict."""
    metric = {
        "verdict": "pass",
        "verdict_qualifier": "ceiling effect — diagnostic null",
    }

    cell = make_audit_card._render_verdict_cell(metric)

    assert cell == "PASS (ceiling effect — diagnostic null)"


# ----- Blocker 3: shared coverage/retention gates -----------------------------


def test_audit_gates_filter_mc_questions_to_split_counts_unique_qids() -> None:
    """Shared helper must mirror compute_csli._filter_test_mc_questions."""
    mc_questions = [
        {"qid": "q1", "question": "first copy"},
        {"qid": "q1", "question": "duplicate copy"},
        {"qid": "outside-test", "question": "not in test"},
    ]

    questions, coverage = filter_mc_questions_to_split(
        mc_questions, {"q1", "q2"}
    )

    assert len(questions) == 2
    assert coverage["target_qids"] == 2
    assert coverage["matched_questions"] == 2
    assert coverage["matched_qids"] == 1
    assert coverage["missing_qids"] == 1
    assert coverage["missing_qids_set"] == {"q2"}
    assert coverage["coverage_rate"] == 0.5


def test_audit_gates_coverage_decision_distinguishes_pass_and_override() -> None:
    """coverage_gate_decision must report passed/overridden separately."""
    passed = coverage_gate_decision(
        0.99, threshold=0.98, override=False
    )
    assert passed["passed"] is True
    assert passed["overridden"] is False
    assert passed["effective_pass"] is True

    overridden = coverage_gate_decision(
        0.72, threshold=0.98, override=True
    )
    assert overridden["passed"] is False
    assert overridden["overridden"] is True
    assert overridden["effective_pass"] is True

    failed = coverage_gate_decision(
        0.72, threshold=0.98, override=False
    )
    assert failed["passed"] is False
    assert failed["overridden"] is False
    assert failed["effective_pass"] is False


def test_audit_gates_retention_decision_mirrors_coverage_shape() -> None:
    """retention_gate_decision must return the same key shape as coverage."""
    decision = retention_gate_decision(
        0.727, threshold=0.98, override=True
    )
    assert set(decision.keys()) == {
        "retention_rate",
        "threshold",
        "passed",
        "overridden",
        "effective_pass",
        "override_flag",
    }
    assert decision["passed"] is False
    assert decision["overridden"] is True


def test_audit_gates_load_mc_build_metadata_handles_missing_file(
    tmp_path: Path,
) -> None:
    """Missing build_metadata.json must return status='missing', not raise."""
    summary = load_mc_build_metadata(tmp_path)

    assert summary["status"] == "missing"
    assert summary["splits"] is None
    assert summary["retention_thresholds"] is None


def test_audit_gates_load_mc_build_metadata_parses_full_file(
    tmp_path: Path,
) -> None:
    """Well-formed build_metadata.json must populate the summary."""
    payload = {
        "splits": {
            "train": {
                "raw_count": 14264,
                "retained_count": 10037,
                "dropped_count": 4227,
                "retention_rate": 0.7036,
                "drop_reasons": {"missing_distractor": 100},
            },
            "val": {
                "raw_count": 3039,
                "retained_count": 2142,
                "dropped_count": 897,
                "retention_rate": 0.7048,
            },
            "test": {
                "raw_count": 3104,
                "retained_count": 2258,
                "dropped_count": 846,
                "retention_rate": 0.7274,
            },
        },
        "retention_thresholds": {"full": 0.98, "smoke": 0.50},
    }
    metadata_path = tmp_path / "build_metadata.json"
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    summary = load_mc_build_metadata(tmp_path)

    assert summary["status"] == "loaded"
    assert summary["source_sha256"] is not None
    assert summary["retention_thresholds"] == {"full": 0.98, "smoke": 0.50}
    assert summary["splits"]["test"]["retained_count"] == 2258
    assert summary["splits"]["val"]["retention_rate"] == pytest.approx(0.7048)


def test_audit_gates_metadata_retention_threshold_precedence(
    tmp_path: Path,
) -> None:
    """Explicit > metadata > 0.98 default precedence must hold."""
    full_metadata = {
        "retention_thresholds": {"full": 0.95, "smoke": 0.55},
    }
    assert metadata_retention_threshold(
        full_metadata, smoke=False, explicit_threshold=None
    ) == pytest.approx(0.95)
    assert metadata_retention_threshold(
        full_metadata, smoke=True, explicit_threshold=None
    ) == pytest.approx(0.55)
    assert metadata_retention_threshold(
        full_metadata, smoke=False, explicit_threshold=0.80
    ) == pytest.approx(0.80)
    assert metadata_retention_threshold(
        {"retention_thresholds": None}, smoke=False, explicit_threshold=None
    ) == pytest.approx(0.98)


def test_audit_gates_build_retention_metadata_marks_not_applicable() -> None:
    """When build_metadata is missing, the retention block must flag applies=False."""
    block = build_retention_metadata(
        {"status": "missing", "splits": None},
        split="test",
        smoke=False,
        explicit_threshold=None,
        override=False,
    )
    assert block["applies"] is False
    assert block["passed"] is None
    assert block["retention_rate"] is None


def test_audit_card_data_provenance_pulls_from_each_metric_artifact() -> None:
    """Audit card must surface coverage + retention provenance per metric.

    Reviewer claim (Blocker 3): calibration and StopDFF lacked CSLI's
    coverage/retention gate, and the audit card silently propagated their
    PASS verdict without recording whether a defensible retained-subset
    audit was actually run. The fix surfaces each metric's coverage and
    retention provenance in audit_card.json's ``data_provenance`` block
    so a reviewer can see at a glance which artifacts ran the gate and
    which (if any) were produced under override.
    """
    csli_data = {
        "panel_csli": {"mean": 0.10, "ci_lower": 0.05, "ci_upper": 0.15},
        "per_model": {
            "tfidf": {"acc_choices_only": 0.20, "leakage_flag": False}
        },
        # CSLI's flat (legacy) coverage block — single test split
        "mc_coverage": {
            "coverage_rate": 1.0,
            "threshold": 0.98,
            "passed": True,
            "overridden": False,
        },
        "mc_retention_gate": {
            "applies": True,
            "retention_rate": 0.727,
            "threshold": 0.98,
            "passed": False,
            "overridden": True,
        },
    }
    cal_data = {
        # New (nested) coverage/retention block — val + test
        "mc_coverage": {
            "val": {
                "coverage_rate": 1.0,
                "threshold": 0.98,
                "passed": True,
                "overridden": False,
            },
            "test": {
                "coverage_rate": 1.0,
                "threshold": 0.98,
                "passed": True,
                "overridden": False,
            },
        },
        "mc_retention_gate": {
            "val": {
                "applies": True,
                "retention_rate": 0.705,
                "threshold": 0.98,
                "passed": False,
                "overridden": True,
            },
            "test": {
                "applies": True,
                "retention_rate": 0.727,
                "threshold": 0.98,
                "passed": False,
                "overridden": True,
            },
        },
    }
    stopdff_data = {
        # Legacy artifact -- no coverage/retention reported
    }

    provenance = make_audit_card._extract_data_provenance(
        csli_data, cal_data, stopdff_data
    )

    # CSLI's legacy flat block is normalized into {test: ...} shape
    assert provenance["csli"]["coverage"]["test"]["rate"] == pytest.approx(1.0)
    assert provenance["csli"]["retention"]["test"]["overridden"] is True
    # Calibration carries both val and test
    assert provenance["calibration"]["coverage"]["val"]["passed"] is True
    assert provenance["calibration"]["retention"]["test"]["overridden"] is True
    # StopDFF legacy artifact reports not_reported
    assert provenance["stopdff"]["coverage"] == "not_reported"
    assert provenance["stopdff"]["retention"] == "not_reported"


def test_audit_card_md_data_provenance_section_renders_all_metrics() -> None:
    """The markdown card must render a Data Provenance section for the reviewer."""
    provenance = {
        "csli": {
            "coverage": {
                "test": {
                    "rate": 1.0,
                    "threshold": 0.98,
                    "passed": True,
                    "overridden": False,
                }
            },
            "retention": {
                "test": {
                    "rate": 0.727,
                    "threshold": 0.98,
                    "passed": False,
                    "overridden": True,
                }
            },
        },
        "calibration": {"coverage": "not_reported", "retention": "not_reported"},
        "stopdff": {"coverage": "not_reported", "retention": "not_reported"},
    }

    lines = make_audit_card._render_data_provenance_md(provenance)

    text = "\n".join(lines)
    assert "## Data Provenance — MC Coverage and Retention" in text
    assert "csli" in text and "calibration" in text and "stopdff" in text
    # Empty / not-reported metrics get a placeholder row, not silently dropped
    assert text.count("not reported") >= 2


# ----- Blocker 4: empty/constant calibration bucket verdict --------------------


def test_audit_card_calibration_warns_on_constant_fallback_bucket() -> None:
    """Constant-calibrator buckets must downgrade verdict to warn.

    Reviewer claim (Blocker 4): compute_ece returns 0.0 for empty buckets
    so the threshold gate falsely PASSes when the calibrator was a
    constant fallback. The fix reads platt_model_type from per-bucket
    metadata and forces verdict='warn' when any bucket is degenerate,
    while preserving the threshold-based PASS for healthy runs.
    """
    cal_data = {
        "max_ece": 0.03,
        "gate_verdict": "pass",
        "per_bucket": {
            "early": {
                "ece": 0.03,
                "n_samples": 100,
                "platt_model_type": "logistic",
                "platt_fallback_reason": None,
            },
            "mid": {
                "ece": 0.0,
                "n_samples": 0,
                "platt_model_type": "constant",
                "platt_fallback_reason": "empty_validation_bucket",
                "platt_constant_probability": 0.0,
            },
            "late": {
                "ece": 0.025,
                "n_samples": 200,
                "platt_model_type": "logistic",
                "platt_fallback_reason": None,
            },
        },
    }

    metric = make_audit_card._evaluate_calibration(cal_data, threshold=0.10)

    assert metric["verdict"] == "warn"
    assert metric["details"]["fallback_buckets"][0]["bucket"] == "mid"
    assert metric["details"]["fallback_buckets"][0]["reason"] == (
        "empty_validation_bucket"
    )
    assert "mid" in metric["details"]["empty_buckets"]


def test_audit_card_calibration_passes_when_all_buckets_logistic() -> None:
    """Healthy calibration runs (all logistic) must still PASS."""
    cal_data = {
        "max_ece": 0.03,
        "gate_verdict": "pass",
        "per_bucket": {
            "early": {
                "ece": 0.026,
                "n_samples": 2553,
                "platt_model_type": "logistic",
                "platt_fallback_reason": None,
            },
            "mid": {
                "ece": 0.006,
                "n_samples": 3316,
                "platt_model_type": "logistic",
                "platt_fallback_reason": None,
            },
            "late": {
                "ece": 0.026,
                "n_samples": 5255,
                "platt_model_type": "logistic",
                "platt_fallback_reason": None,
            },
        },
    }

    metric = make_audit_card._evaluate_calibration(cal_data, threshold=0.10)

    assert metric["verdict"] == "pass"
    assert metric["details"]["fallback_buckets"] == []
    assert metric["details"]["empty_buckets"] == []


def test_audit_card_calibration_legacy_json_no_fallback_fields() -> None:
    """Older calibration.json without fallback fields stays at threshold verdict."""
    cal_data = {
        "max_ece": 0.03,
        "gate_verdict": "pass",
        "per_bucket": {
            "early": {"ece": 0.026},
            "mid": {"ece": 0.006},
            "late": {"ece": 0.026},
        },
    }

    metric = make_audit_card._evaluate_calibration(cal_data, threshold=0.10)

    # No constant/empty signal => threshold-based PASS
    assert metric["verdict"] == "pass"


# ----- Blocker 5: load_mc_questions dual-shape ---------------------------------


def test_load_mc_questions_accepts_plain_list_shape(tmp_path: Path) -> None:
    """Plain-list payloads must round-trip through load_mc_questions."""
    row = _make_mc_row("q1")
    path = tmp_path / "mc.json"
    path.write_text(json.dumps([row]), encoding="utf-8")

    questions = load_mc_questions(path)

    assert len(questions) == 1
    assert questions[0].qid == "q1"


def test_load_mc_questions_accepts_wrapped_shape_with_mc_rows(
    tmp_path: Path,
) -> None:
    """Wrapped payloads carrying MC rows must round-trip too.

    Reviewer claim (Blocker 5): the convenience loader did naive
    iteration so wrapped payloads were silently iterated as dict keys.
    The fix delegates to iter_split_questions.
    """
    row = _make_mc_row("q1")
    path = tmp_path / "wrapped.json"
    payload = {"metadata": {"seed": 789685}, "questions": [row]}
    path.write_text(json.dumps(payload), encoding="utf-8")

    questions = load_mc_questions(path)

    assert len(questions) == 1
    assert questions[0].qid == "q1"


def test_load_mc_questions_surfaces_producer_mismatch_on_tossup_only(
    tmp_path: Path,
) -> None:
    """TOSSUP-only wrapped split must raise an actionable KeyError.

    save_splits writes TOSSUP-only rows (no options/gold_index); calling
    load_mc_questions on that file used to TypeError on dict iteration
    and now raises a producer-mismatch KeyError that tells the operator
    to re-run build_mc_dataset.py.
    """
    tossup_row = {
        "qid": "q1",
        "question": "Test question?",
        "tokens": ["Test", "question"],
        "answer_primary": "answer",
        "clean_answers": ["answer"],
        "run_indices": [0, 1],
        "cumulative_prefixes": ["Test", "Test question"],
    }
    path = tmp_path / "tossup_only.json"
    path.write_text(
        json.dumps({"metadata": {}, "questions": [tossup_row]}),
        encoding="utf-8",
    )

    with pytest.raises(KeyError) as excinfo:
        load_mc_questions(path)

    assert "build_mc_dataset.py" in str(excinfo.value)


def _make_mc_row(qid: str) -> dict:
    """Build a synthetic but schema-complete MC question dict for tests."""
    return {
        "qid": qid,
        "question": "Test question?",
        "tokens": ["Test", "question"],
        "answer_primary": "answer",
        "clean_answers": ["answer"],
        "run_indices": [0, 1],
        "human_buzz_positions": None,
        "category": "Test",
        "cumulative_prefixes": ["Test", "Test question"],
        "options": ["answer", "wrong1", "wrong2", "wrong3"],
        "gold_index": 0,
        "option_profiles": [
            {"answer_text": "answer"},
            {"answer_text": "wrong1"},
            {"answer_text": "wrong2"},
            {"answer_text": "wrong3"},
        ],
        "option_answer_primary": ["answer", "wrong1", "wrong2", "wrong3"],
        "distractor_strategy": "test",
    }


def test_iter_split_questions_still_handles_both_shapes() -> None:
    """Smoke-check that the underlying helper hasn't regressed."""
    plain = [{"qid": "q1"}]
    wrapped = {"metadata": {}, "questions": [{"qid": "q1"}]}

    assert iter_split_questions(plain) == [{"qid": "q1"}]
    assert iter_split_questions(wrapped) == [{"qid": "q1"}]


# ----- Blocker 6: belief math NaN guards --------------------------------------


def test_softmax_belief_raises_on_empty_input() -> None:
    """Empty input must raise an actionable ValueError, not crash inside np.max."""
    with pytest.raises(ValueError, match="non-empty 1D"):
        softmax_belief(np.array([]), beta=1.0)


def test_softmax_belief_returns_uniform_on_all_neg_inf() -> None:
    """All-(-inf) scores must degrade to uniform belief, not silent NaN."""
    belief = softmax_belief(np.array([-np.inf, -np.inf, -np.inf, -np.inf]), beta=5.0)
    expected = np.full(4, 0.25, dtype=np.float32)
    np.testing.assert_allclose(belief, expected)
    assert np.all(np.isfinite(belief))


def test_softmax_belief_returns_uniform_on_nan_input() -> None:
    """NaN-containing scores must degrade to uniform belief, not propagate NaN."""
    belief = softmax_belief(np.array([0.5, np.nan, 0.5, 0.5]), beta=5.0)
    expected = np.full(4, 0.25, dtype=np.float32)
    np.testing.assert_allclose(belief, expected)
    assert np.all(np.isfinite(belief))


def test_softmax_belief_handles_large_beta_without_nan() -> None:
    """Large beta on bounded cosine scores must not overflow into NaN."""
    belief = softmax_belief(np.array([0.5, 0.4, 0.3, 0.2]), beta=1000.0)
    assert np.all(np.isfinite(belief))
    assert belief.sum() == pytest.approx(1.0, abs=1e-5)
    # Largest score should dominate under huge beta
    assert int(np.argmax(belief)) == 0


def test_softmax_belief_normal_path_unchanged() -> None:
    """Normal bounded inputs must produce the same output as before the guards."""
    scores = np.array([0.9, 0.7, 0.4, 0.1])
    belief = softmax_belief(scores, beta=5.0)
    assert np.all(np.isfinite(belief))
    assert belief.sum() == pytest.approx(1.0, abs=1e-5)
    assert int(np.argmax(belief)) == 0


def test_bayesian_update_raises_on_empty_scores() -> None:
    """Empty scores must raise ValueError before the math runs."""
    with pytest.raises(ValueError, match="non-empty 1D"):
        bayesian_update(np.array([0.25, 0.25, 0.25, 0.25]), np.array([]), beta=1.0)


def test_bayesian_update_raises_on_shape_mismatch() -> None:
    """Mismatched prior/scores shapes must raise, not produce silent broadcasting."""
    with pytest.raises(ValueError, match="matching shapes"):
        bayesian_update(
            np.array([0.25, 0.25, 0.25, 0.25]),
            np.array([0.9, 0.7, 0.4]),
            beta=1.0,
        )


def test_bayesian_update_returns_uniform_on_neg_inf_scores() -> None:
    """All-(-inf) scores must degrade to uniform belief."""
    prior = np.array([0.25, 0.25, 0.25, 0.25])
    posterior = bayesian_update(
        prior, np.array([-np.inf, -np.inf, -np.inf, -np.inf]), beta=5.0
    )
    np.testing.assert_allclose(posterior, np.full(4, 0.25, dtype=np.float32))
    assert np.all(np.isfinite(posterior))


def test_bayesian_update_returns_uniform_on_nan_prior() -> None:
    """A NaN-bearing prior (e.g., from a prior bad update) must not propagate."""
    prior = np.array([0.25, np.nan, 0.25, 0.25])
    posterior = bayesian_update(prior, np.array([0.5, 0.5, 0.5, 0.5]), beta=5.0)
    np.testing.assert_allclose(posterior, np.full(4, 0.25, dtype=np.float32))
    assert np.all(np.isfinite(posterior))


def test_bayesian_update_normal_path_unchanged() -> None:
    """Normal finite inputs must produce the same output as before the guards."""
    prior = np.array([0.4, 0.3, 0.2, 0.1])
    posterior = bayesian_update(
        prior, np.array([0.9, 0.7, 0.4, 0.1]), beta=5.0
    )
    assert np.all(np.isfinite(posterior))
    assert posterior.sum() == pytest.approx(1.0, abs=1e-5)
    # Largest combined prior*likelihood should be index 0
    assert int(np.argmax(posterior)) == 0


# ----- Build coverage/retention metadata serialization -----------------------


def test_build_coverage_metadata_emits_serializer_compatible_shape() -> None:
    """build_coverage_metadata must produce a JSON-serializable dict shape."""
    coverage = {
        "target_qids": 100,
        "mc_questions_total": 95,
        "matched_questions": 95,
        "matched_qids": 95,
        "missing_qids": 5,
        "missing_qids_set": {"q96", "q97", "q98", "q99", "q100"},
        "coverage_rate": 0.95,
    }
    metadata = build_coverage_metadata(
        coverage, threshold=0.98, override=True
    )

    # The on-disk schema (consumed by audit card) does not need
    # missing_qids_set (which is a Python set, not JSON-serializable).
    assert "missing_qids_set" not in metadata
    assert metadata["coverage_rate"] == pytest.approx(0.95)
    assert metadata["passed"] is False
    assert metadata["overridden"] is True
    assert metadata["override_flag"] == "--allow-incomplete-mc-coverage"
    # Validate the result is actually JSON-serializable end-to-end
    json.dumps(metadata)
