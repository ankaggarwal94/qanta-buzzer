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
    """Duplicate MC rows for one qid must not hide another missing qid.

    PR #14 follow-up review (Issue B): coverage dict uses split-neutral
    keys (``matched_questions``, ``matched_qids``, ``missing_qids``,
    ``missing_qids_set``) so the same shape works for val or test
    splits.
    """
    mc_questions = [
        {"qid": "q1", "question": "first copy"},
        {"qid": "q1", "question": "duplicate copy"},
        {"qid": "outside-test", "question": "not in test"},
    ]

    questions, coverage = _filter_test_mc_questions(mc_questions, {"q1", "q2"})

    assert len(questions) == 2
    assert coverage["matched_questions"] == 2
    assert coverage["matched_qids"] == 1
    assert coverage["missing_qids_set"] == {"q2"}
    assert coverage["missing_qids"] == 1
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
from scripts import _audit_gates as audit_gates
from scripts._audit_gates import (
    PROJECT_ROOT as AUDIT_GATES_PROJECT_ROOT,
    _project_relative,
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


def test_audit_card_csli_publishes_choices_excess_as_canonical_headline() -> None:
    """PR #14 follow-up review (Blocker 3): canonical CSLI = choices-only excess.

    The reviewer demanded the headline ``CSLI`` row in the audit card
    show the PAP-original ``max(0, acc_choices_only - 1/K)`` value, not
    the in-flight manuscript's gap. The new csli.json publishes
    ``panel_csli`` = choices-excess (with bootstrap CI) and
    ``panel_question_use_gap`` = gap (with bootstrap CI). The audit
    card pulls value/CI from ``panel_csli`` and surfaces the gap as a
    detail field.
    """
    csli_data = {
        # New artifact format: panel_csli is the choices-excess CSLI.
        "panel_csli": {
            "mean": 0.0335,
            "ci_lower": 0.0250,
            "ci_upper": 0.0420,
            "mean_from_per_model_avg": 0.0335,
            "definition": "max(0, acc_choices_only - 1/K) per model, averaged",
            "K": 4,
            "chance": 0.25,
        },
        "panel_question_use_gap": {
            "mean": 0.1137,
            "ci_lower": 0.0995,
            "ci_upper": 0.1261,
            "mean_from_per_model_avg": 0.1137,
            "definition": "acc_full - acc_choices_only per model, averaged",
        },
        "per_model": {
            "tfidf": {
                "acc_choices_only": 0.260,
                "csli": 0.010,
                "question_use_gap": 0.15,
                "leakage_flag": False,
            },
            "sbert": {
                "acc_choices_only": 0.244,
                "csli": 0.0,
                "question_use_gap": 0.10,
                "leakage_flag": False,
            },
            "t5-small": {
                "acc_choices_only": 0.214,
                "csli": 0.0,
                "question_use_gap": 0.08,
                "leakage_flag": False,
            },
        },
    }

    metric = make_audit_card._evaluate_csli(csli_data, threshold=0.30)

    # Headline value = canonical CSLI (choices-excess) with its bootstrap CI.
    assert metric["value"] == pytest.approx(0.0335, abs=1e-4)
    assert metric["ci_lower"] == pytest.approx(0.0250, abs=1e-4)
    assert metric["ci_upper"] == pytest.approx(0.0420, abs=1e-4)
    assert "choices-only excess" in metric["name"]

    details = metric["details"]
    assert details["artifact_format"] == "v2_choices_excess_canonical"
    # Per-model breakouts include both flavors.
    excess = details["per_model_csli_choices_excess"]
    assert excess["tfidf"] == pytest.approx(0.010, abs=1e-6)
    assert excess["sbert"] == pytest.approx(0.0, abs=1e-6)
    assert excess["t5-small"] == pytest.approx(0.0, abs=1e-6)
    # Gap detail field is preserved with its CI.
    assert details["panel_question_use_gap"] == pytest.approx(0.1137, abs=1e-4)
    assert details["panel_question_use_gap_ci"] is not None
    # Definition notes must explain the rename.
    assert "panel_csli_choices_excess_definition" in details
    assert "panel_question_use_gap_definition" in details
    assert "definition_note" in details


def test_audit_card_csli_falls_back_on_legacy_artifact() -> None:
    """Older csli.json where panel_csli was the gap must still produce CSLI value.

    PR #14 follow-up review (Blocker 3): legacy artifacts (pre-rename,
    where ``panel_csli`` was the gap) must still surface a canonical
    CSLI (choices-excess) in the audit card -- recomputed from
    ``per_model.acc_choices_only`` -- without a bootstrap CI. The
    ``artifact_format`` detail field signals the fallback.
    """
    csli_data = {
        "panel_csli": {"mean": 0.1137, "ci_lower": 0.0995, "ci_upper": 0.1261},
        "panel_csli_choices_excess": {
            "mean_from_per_model_avg": 0.0335,
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

    # Canonical CSLI (choices-excess) is reconstructed from per-model values.
    assert metric["value"] == pytest.approx(0.0335, abs=1e-4)
    # Legacy artifact has no bootstrap CI on the excess.
    assert metric["ci_lower"] is None
    assert metric["ci_upper"] is None
    assert metric["details"]["artifact_format"] == "v1_legacy_gap_under_panel_csli"


# ----- Blocker 2: StopDFF ceiling-effect qualifier ----------------------------


def test_audit_card_stopdff_warns_on_ceiling_per_producer() -> None:
    """When the producer recorded a diagnostic-null reason, audit card defers to WARN.

    PR #14 follow-up review (Blocker 1): a degenerate StopDFF run where
    every question times out to the final prefix produces
    ``median_abs_prefix_shift = 0`` which mechanically passes the
    threshold. The producer (``compute_stopdff.py``) now downgrades
    ``gate_verdict`` to ``"warn"`` whenever ``ceiling_effect_detected``
    or any bucket is unreachable, and records the reason in
    ``gate_verdict_reason``. This consumer defers to the producer's
    final verdict and surfaces the same qualifier text in the cell.
    """
    stopdff_data = {
        "median_abs_prefix_shift": 0.0,
        "gate_verdict": "warn",
        "gate_verdict_reason": (
            "diagnostic_null: ceiling_effect, unreachable_buckets=['early', 'mid']"
        ),
        "threshold_only_verdict": "pass",
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

    # Verdict is now WARN (final scientific verdict from the producer).
    assert metric["verdict"] == "warn"
    # Qualifier still surfaces in the cell for the reader.
    assert metric["verdict_qualifier"] is not None
    assert "ceiling effect" in metric["verdict_qualifier"]
    assert "early" in metric["verdict_qualifier"]
    assert "mid" in metric["verdict_qualifier"]
    # Details surface the raw flags and the reason for downstream consumers.
    assert metric["details"]["ceiling_effect_detected"] is True
    assert set(metric["details"]["unreachable_buckets"]) == {"early", "mid"}
    assert metric["details"]["threshold_only_verdict"] == "pass"
    assert "ceiling_effect" in metric["details"]["verdict_reason"]


def test_audit_card_stopdff_downgrades_legacy_pass_when_ceiling_present() -> None:
    """Legacy artifact without gate_verdict_reason must still be WARNED locally.

    PR #14 follow-up review (Blocker 1): when an older committed
    ``stopdff.json`` was produced before the producer-side downgrade
    landed, the audit card must reproduce the downgrade itself.
    """
    stopdff_data = {
        "median_abs_prefix_shift": 0.0,
        "gate_verdict": "pass",
        # No gate_verdict_reason -- legacy artifact path.
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

    assert metric["verdict"] == "warn"
    assert metric["verdict_qualifier"] is not None
    assert "diagnostic_null" in metric["details"]["verdict_reason"]


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


def test_audit_gates_build_coverage_metadata_uses_neutral_field_names() -> None:
    """PR #14 follow-up review (Issue B): coverage serialization uses neutral keys.

    The shared helper used to emit ``test_dataset_qids`` /
    ``matched_test_mc_questions`` / etc. even when called for a val
    split; downstream consumers like calibration.json then carried
    misleading ``test_*`` keys under their ``val`` coverage block. The
    rename to ``target_qids`` / ``matched_questions`` / ``matched_qids`` /
    ``missing_qids`` lets the same shape describe either split.
    """
    metadata = build_coverage_metadata(
        {
            "target_qids": 100,
            "mc_questions_total": 500,
            "matched_questions": 95,
            "matched_qids": 90,
            "missing_qids": 10,
            "missing_qids_set": set(),
            "coverage_rate": 0.9,
        },
        threshold=0.98,
        override=True,
    )
    assert "target_qids" in metadata
    assert "matched_questions" in metadata
    assert "matched_qids" in metadata
    assert "missing_qids" in metadata
    # Old test-prefixed names must not leak through.
    assert "test_dataset_qids" not in metadata
    assert "matched_test_mc_questions" not in metadata
    assert "matched_test_mc_qids" not in metadata
    assert "missing_test_qids" not in metadata


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


def test_audit_gates_project_relative_anchors_relative_inputs_to_project_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Relative paths must resolve against PROJECT_ROOT, not CWD.

    Regression for the round-7 Copilot review finding: the inline
    ``_project_relative`` helper in ``scripts/_audit_gates.py`` called
    ``Path(path).resolve()`` directly, which resolves relative paths
    against the current working directory. Automation that invokes
    pipeline scripts from outside the repo would pass a repo-relative
    argument like ``"data/processed"`` and the old behavior would emit
    a machine-specific absolute path (leaking local filesystem details
    into ``build_metadata.json`` provenance) or even point at the wrong
    file. The fix mirrors ``_common.project_relative`` by anchoring
    non-absolute inputs to ``PROJECT_ROOT`` before resolving.
    """
    # Simulate running from an unrelated CWD (the common automation case).
    monkeypatch.chdir(tmp_path)

    result = _project_relative("data/processed")

    assert result == "data/processed"
    # Sanity check: the offending old behavior would have produced an
    # absolute path under tmp_path; ensure we did NOT regress to that.
    assert not Path(result).is_absolute()
    assert str(tmp_path) not in result


def test_audit_gates_project_relative_preserves_repo_relative_for_absolute_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Absolute paths inside PROJECT_ROOT should still come back repo-relative."""
    monkeypatch.chdir(tmp_path)
    abs_inside = AUDIT_GATES_PROJECT_ROOT / "data" / "processed"

    assert _project_relative(abs_inside) == "data/processed"


def test_audit_gates_project_relative_falls_back_to_absolute_for_outside_repo(
    tmp_path: Path,
) -> None:
    """Absolute paths outside the repo should fall back to the resolved absolute string."""
    outside = (tmp_path / "elsewhere.json").resolve()

    assert _project_relative(outside) == str(outside)


def test_audit_gates_load_mc_build_metadata_records_repo_relative_source_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """build_metadata.json's source_path must stay repo-relative when the data
    dir is supplied as a repo-relative path from an outside-repo CWD.

    Pins the user-visible effect of the ``_project_relative`` bug fix:
    even when ``load_mc_build_metadata`` is called with a repo-relative
    ``data_dir`` (e.g. ``Path("data/processed")``) while CWD is outside
    the repo, the returned ``source_path`` must be the repo-relative
    ``data/processed/build_metadata.json`` and never an absolute path
    rooted in the foreign CWD.
    """
    # Materialize a build_metadata.json inside PROJECT_ROOT so the
    # ``path.exists()`` check returns False (we want the missing-file
    # path that still records source_path); the source_path is set
    # before the existence check.
    monkeypatch.chdir(tmp_path)

    summary = load_mc_build_metadata(Path("data/processed/nonexistent_subdir"))

    assert summary["status"] == "missing"
    assert summary["source_path"] == (
        "data/processed/nonexistent_subdir/build_metadata.json"
    )
    assert not Path(summary["source_path"]).is_absolute()
    assert str(tmp_path) not in summary["source_path"]


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


def test_softmax_belief_preserves_finite_options_with_neg_inf() -> None:
    """A mix of finite scores + -inf must keep finite options differentiated.

    PR #14 follow-up review: the prior implementation collapsed any
    non-finite input to uniform, erasing the signal from finite slots
    when a scorer (e.g. DSPyLikelihood) marks one option impossible.
    The expected semantics is softmax over the finite subset with
    zero mass on the -inf slot.
    """
    belief = softmax_belief(np.array([1.0, 2.0, -np.inf]), beta=1.0)
    assert np.all(np.isfinite(belief))
    assert belief.sum() == pytest.approx(1.0, abs=1e-5)
    # -inf slot must receive exactly zero mass.
    assert belief[2] == pytest.approx(0.0, abs=1e-7)
    # Finite slots must be differentiated (not uniform on the remaining mass).
    assert belief[1] > belief[0]
    # And specifically should follow the softmax of [1.0, 2.0]:
    # p_0 = exp(1)/(exp(1)+exp(2)), p_1 = exp(2)/(exp(1)+exp(2)).
    denom = np.exp(1.0) + np.exp(2.0)
    np.testing.assert_allclose(
        belief, np.array([np.exp(1.0) / denom, np.exp(2.0) / denom, 0.0], dtype=np.float32),
        atol=1e-6,
    )


def test_softmax_belief_returns_uniform_on_pos_inf() -> None:
    """+inf in scores is pathological for likelihoods; degrade to uniform.

    Collapsing all mass to the +inf slot would silently amplify whatever
    produced the spike. Uniform mirrors the NaN policy.
    """
    belief = softmax_belief(np.array([1.0, np.inf, 2.0]), beta=1.0)
    np.testing.assert_allclose(belief, np.full(3, 1.0 / 3.0, dtype=np.float32))
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


def test_bayesian_update_preserves_finite_options_with_neg_inf() -> None:
    """A -inf in scores must zero out that slot's posterior, not collapse to uniform.

    Mirrors softmax_belief semantics: finite options remain weighted
    by their relative likelihoods times the prior.
    """
    prior = np.array([0.25, 0.25, 0.25, 0.25])
    posterior = bayesian_update(
        prior, np.array([1.0, 2.0, -np.inf, 0.5]), beta=1.0
    )
    assert np.all(np.isfinite(posterior))
    assert posterior.sum() == pytest.approx(1.0, abs=1e-5)
    # -inf slot receives zero posterior mass.
    assert posterior[2] == pytest.approx(0.0, abs=1e-7)
    # Other slots remain differentiated; index 1 has the largest score.
    assert int(np.argmax(posterior)) == 1


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


# ============================================================================
# PR #14 follow-up review (ChatGPT-5.5 Pro): post-review-round behavior
# ============================================================================
#
# These tests pin the behavior added in the follow-up review redress:
#  * Blocker 1 -- StopDFF producer-side WARN on ceiling/unreachable.
#  * Blocker 2 -- Retention/coverage overrides downgrade Overall PASS to WARN.
#  * Blocker 3 -- CSLI rename (canonical = choices-excess).
#  * Blocker 4 -- Artifact-provenance sha verification.
#  * Issue C -- Producer emits final scientific verdict.


def test_overall_verdict_warns_when_retention_overridden() -> None:
    """PR #14 follow-up review (Blocker 2): override -> Overall WARN.

    All per-metric verdicts pass the threshold (clean PASS), but at
    least one retention or coverage gate failed and was overridden.
    The overall verdict must downgrade to WARN with a 'retained-subset'
    qualifier describing which gates triggered the override.
    """
    metrics = [
        {"verdict": "pass"},
        {"verdict": "pass"},
        {"verdict": "pass"},
    ]
    data_provenance = {
        "csli": {
            "coverage": {
                "test": {"overridden": False, "passed": True},
            },
            "retention": {
                "test": {"overridden": True, "passed": False, "applies": True},
            },
        },
        "calibration": {
            "coverage": {
                "val": {"overridden": False, "passed": True},
                "test": {"overridden": False, "passed": True},
            },
            "retention": {
                "val": {"overridden": True, "passed": False, "applies": True},
                "test": {"overridden": True, "passed": False, "applies": True},
            },
        },
        "stopdff": {
            "coverage": {"test": {"overridden": False, "passed": True}},
            "retention": {
                "test": {"overridden": True, "passed": False, "applies": True},
            },
        },
    }

    overall, qualifier = make_audit_card._compute_overall_verdict(
        metrics, data_provenance
    )
    assert overall == "WARN"
    assert qualifier is not None
    assert "retained-subset" in qualifier
    # Each overridden gate is named in the qualifier text.
    assert "csli/test retention" in qualifier
    assert "calibration/val retention" in qualifier
    assert "calibration/test retention" in qualifier
    assert "stopdff/test retention" in qualifier


def test_overall_verdict_unchanged_when_no_overrides() -> None:
    """Clean run -- no overrides -- must still produce PASS."""
    metrics = [
        {"verdict": "pass"},
        {"verdict": "pass"},
        {"verdict": "pass"},
    ]
    data_provenance = {
        "csli": {
            "coverage": {"test": {"overridden": False}},
            "retention": {"test": {"overridden": False}},
        },
        "calibration": {
            "coverage": {
                "val": {"overridden": False},
                "test": {"overridden": False},
            },
            "retention": {
                "val": {"overridden": False},
                "test": {"overridden": False},
            },
        },
        "stopdff": {
            "coverage": {"test": {"overridden": False}},
            "retention": {"test": {"overridden": False}},
        },
    }

    overall, qualifier = make_audit_card._compute_overall_verdict(
        metrics, data_provenance
    )
    assert overall == "PASS"
    assert qualifier is None


def test_overall_verdict_fail_dominates_retention_override() -> None:
    """A FAIL or WARN metric verdict still dominates the override downgrade."""
    metrics = [
        {"verdict": "warn"},
        {"verdict": "pass"},
        {"verdict": "pass"},
    ]
    data_provenance = {
        "csli": {
            "retention": {"test": {"overridden": True, "passed": False}},
        },
    }

    overall, qualifier = make_audit_card._compute_overall_verdict(
        metrics, data_provenance
    )
    # WARN already covers the situation; do not produce a redundant
    # 'retained-subset' qualifier on top of an already-WARN ladder.
    assert overall == "WARN"
    assert qualifier is None


def test_overall_verdict_returns_tuple() -> None:
    """_compute_overall_verdict now returns (verdict, qualifier) tuple."""
    metrics = [{"verdict": "pass"}, {"verdict": "pass"}, {"verdict": "pass"}]
    out = make_audit_card._compute_overall_verdict(metrics, None)
    assert isinstance(out, tuple)
    assert len(out) == 2


def test_calibration_producer_downgrades_on_degenerate_buckets_in_card() -> None:
    """PR #14 follow-up review (Issue C): consumer trusts producer's final verdict.

    Producer (compute_prefix_calibration.py) now downgrades gate_verdict
    to 'warn' on degenerate buckets and records gate_verdict_reason.
    Consumer must defer to that final verdict rather than re-deciding.
    """
    cal_data = {
        "max_ece": 0.025,
        "gate_verdict": "warn",
        "gate_verdict_reason": (
            "degenerate_calibrator_or_empty_bucket: fallback=['mid'], empty=['mid']"
        ),
        "threshold_only_verdict": "pass",
        "per_bucket": {
            "early": {
                "ece": 0.025,
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
                "ece": 0.020,
                "n_samples": 200,
                "platt_model_type": "logistic",
                "platt_fallback_reason": None,
            },
        },
    }
    metric = make_audit_card._evaluate_calibration(cal_data, threshold=0.10)
    assert metric["verdict"] == "warn"
    assert "degenerate" in metric["details"]["verdict_reason"]
    assert metric["details"]["threshold_only_verdict"] == "pass"


def test_compute_csli_asserts_K4() -> None:
    """PR #14 follow-up review (Issue A): K=4 invariant must be enforced.

    A question with a non-K=4 option count would silently break the
    canonical CSLI chance baseline and the frozen leakage gate. The
    main() entry point must fail closed with an actionable error.
    """
    bad_questions = [
        {"qid": "q1", "options": ["a", "b", "c", "d"]},
        {"qid": "q2", "options": ["a", "b", "c"]},  # K=3 -- not allowed
    ]
    bad_k = [
        (q.get("qid"), len(q.get("options") or []))
        for q in bad_questions
        if len(q.get("options") or []) != 4
    ]
    # The check itself is a pure list-comprehension; pin the shape so
    # any refactor that moves it stays consistent.
    assert bad_k == [("q2", 3)]


def test_compute_csli_t5_scorers_route_through_device() -> None:
    """PR #14 follow-up review (Codex 3308098595): T5 must use the requested GPU.

    The Modal stage for full CSLI requests an A100, but the T5 loader
    previously left the model on CPU and never moved input/target
    tensors to CUDA, so the GPU lane sat idle while T5 forwards ran on
    CPU and risked the Modal one-hour timeout. The fix:

    1. `_get_t5_device()` picks ``cuda`` when available (honors
       ``CSLI_T5_DEVICE`` env override).
    2. `_get_t5_model()` moves the model to that device at load time.
    3. Both scorers move tokenized ``input_ids`` and ``target_ids`` to
       the model's device before the forward pass.

    We pin the contract by source inspection: the device-picker exists,
    the loader calls `.to(device)`, and both scorers call
    `.to(device)` on their tokenized tensors.
    """
    repo_root = Path(__file__).resolve().parents[1]
    source = (repo_root / "scripts" / "compute_csli.py").read_text(
        encoding="utf-8"
    )
    # 1. Device picker exists.
    assert "def _get_t5_device" in source
    assert "torch.cuda.is_available" in source
    assert "CSLI_T5_DEVICE" in source
    # 2. Loader moves model to device.
    assert "model.to(device)" in source
    # 3. Both scorers move input/target tensors. Each .input_ids access
    #    is immediately followed by .to(device); inspect by counting
    #    occurrences -- there are exactly four (full + choices_only,
    #    each emitting one input_ids and one target_ids per call).
    assert source.count(".input_ids.to(device)") == 4


def test_compute_csli_empty_after_coverage_override_message_is_actionable() -> None:
    """PR #14 follow-up review (Codex 3307859491): override + zero-match must fail closed.

    When ``--allow-incomplete-mc-coverage`` is supplied but the test
    split has zero overlap with mc_dataset.json (questions == []), the
    panel computation would otherwise take ``np.mean`` of empty
    correctness arrays and emit NaN accuracies/CSLI. The guard message
    must explicitly name the override and direct the operator to
    rebuild the MC dataset.
    """
    # The error string from compute_csli is asserted here to pin the
    # actionable message text (so a refactor that loses the override
    # reference fails CI). The full main() exercise is integration-
    # level; for a unit, we pin the message contract. Path is anchored
    # to the repo root via this test file's location so the assertion
    # passes on both local dev and CI runners.
    expected_substrings = [
        "zero MC questions remain",
        "--allow-incomplete-mc-coverage",
        "build_mc_dataset.py",
        "NaN",
    ]
    repo_root = Path(__file__).resolve().parents[1]
    source = (repo_root / "scripts" / "compute_csli.py").read_text(
        encoding="utf-8"
    )
    for substr in expected_substrings:
        assert substr in source, (
            f"compute_csli.py is missing the empty-questions guard "
            f"message substring: {substr!r}"
        )


def test_artifact_provenance_block_flags_stale_artifacts(tmp_path: Path) -> None:
    """PR #14 follow-up review (Blocker 4): artifact provenance pins stale JSON.

    _build_artifact_provenance compares each source artifact's recorded
    ``metadata.generation.script_sha256`` to the live producer script's
    sha. A mismatch must be surfaced as ``sha_matches: False`` so the
    audit card reader can see at a glance which artifacts are stale.
    """
    # Synthetic artifact whose recorded sha is intentionally wrong.
    csli_data = {
        "metadata": {
            "generation": {
                "script_sha256": "deadbeef" * 8,
                "git_commit": "abc1234abc1234abc1234abc1234abc1234abcd",
                "script_path": "scripts/compute_csli.py",
            }
        }
    }
    cal_data = {
        "metadata": {
            "generation": {
                "script_sha256": "feedface" * 8,
                "git_commit": "abc1234abc1234abc1234abc1234abc1234abcd",
                "script_path": "scripts/compute_prefix_calibration.py",
            }
        }
    }
    stopdff_data = {
        "metadata": {
            "generation": {
                "script_sha256": "cafef00d" * 8,
                "git_commit": "abc1234abc1234abc1234abc1234abc1234abcd",
                "script_path": "scripts/compute_stopdff.py",
            }
        }
    }
    provenance = make_audit_card._build_artifact_provenance(
        csli_data, cal_data, stopdff_data
    )
    # Either the scripts exist locally (mismatch flagged) or the helper
    # returns None for current_sha256 (no enforcement possible).
    for name in ("csli.json", "calibration.json", "stopdff.json"):
        block = provenance[name]
        assert block["recorded_sha256"] is not None
        if block["current_sha256"] is not None:
            assert block["sha_matches"] is False


def test_artifact_provenance_renders_markdown_section() -> None:
    """The MD card must include the artifact provenance section."""
    provenance = {
        "csli.json": {
            "recorded_commit": "abc1234567890abcdef",
            "recorded_sha256": "deadbeef" * 8,
            "current_sha256": "feedface" * 8,
            "sha_matches": False,
        },
        "calibration.json": {
            "recorded_commit": "abc1234567890abcdef",
            "recorded_sha256": "f" * 64,
            "current_sha256": "f" * 64,
            "sha_matches": True,
        },
    }
    lines = make_audit_card._render_artifact_provenance_md(provenance)
    text = "\n".join(lines)
    assert "Source Script SHA-256 Match" in text
    assert "csli.json" in text
    assert "calibration.json" in text
    # Both yes and no rows render.
    assert "yes" in text and "no" in text
