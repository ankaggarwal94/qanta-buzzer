"""RED suite — strict profile schema + typed ingress.

Covers: R-001, R-002, R-003, R-004, R-020, R-029, R-031, R-032.
Spec: .correctless/specs/camera-ready-aims-evidence.md
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from reproducibility.colm_aims_2026 import schema
from tests._colm_aims_helpers import (  # noqa: F401  (colm_no_network is an autouse fixture)
    SEMANTIC_BLOCK,
    STRICT_PROFILE_ID,
    OBSERVED_PROFILE_ID,
    colm_no_network,
    expected_estimand_digest,
    make_idealized_arm,
    make_llm_involvement,
    make_profile,
    make_record,
    standard_records,
)


# ---------------------------------------------------------------------------
# R-001: pinned semantic block
# ---------------------------------------------------------------------------


def test_valid_strict_profile_passes_validation():
    # Tests R-001 [unit]: the fully pinned semantic block validates cleanly.
    schema.validate_profile(make_profile())


@pytest.mark.parametrize("field", sorted(SEMANTIC_BLOCK))
def test_missing_semantic_field_rejected(field):
    # Tests R-001 [unit]: a missing key inside the semantic block is rejected.
    profile = make_profile()
    del profile["semantic"][field]
    with pytest.raises(schema.SchemaValidationError):
        schema.validate_profile(profile)


@pytest.mark.parametrize("field", sorted(SEMANTIC_BLOCK))
def test_renamed_semantic_field_rejected(field):
    # Tests R-001 [unit]: a renamed key (old name gone, new unknown name
    # present) is rejected.
    profile = make_profile()
    profile["semantic"][field + "_renamed"] = profile["semantic"].pop(field)
    with pytest.raises(schema.SchemaValidationError):
        schema.validate_profile(profile)


@pytest.mark.parametrize(
    "field,bad_value",
    [
        ("trajectory_source", "observed_reference"),
        ("observed_open_ended", True),
        ("observed_open_ended_answers", True),
        ("observed_open_ended_stopping_actions", True),
        ("pairing_unit", "observed_sessions"),
        ("pairing_is_observed_sessions", True),
        ("supports", "actual_decision_preservation"),
        ("does_not_support", "nothing"),
    ],
)
def test_altered_semantic_value_rejected(field, bad_value):
    # Tests R-001 [unit]: altering any pinned value fails validation.
    profile = make_profile()
    profile["semantic"][field] = bad_value
    with pytest.raises(schema.SchemaValidationError):
        schema.validate_profile(profile)


def test_unknown_key_inside_semantic_block_rejected():
    # Tests R-001 [unit]: unknown keys inside the semantic block are rejected.
    profile = make_profile()
    profile["semantic"]["extra_semantic_claim"] = "observed"
    with pytest.raises(schema.SchemaValidationError):
        schema.validate_profile(profile)


def test_historical_format_qa_never_substitutes_for_semantic_layer():
    # Tests R-001 [unit]: a payload carrying the historical format="QA"
    # identifier but no semantic block does not pass strict validation.
    profile = make_profile()
    del profile["semantic"]
    profile["format"] = "QA"
    with pytest.raises(schema.SchemaValidationError):
        schema.validate_profile(profile)


# ---------------------------------------------------------------------------
# R-002: constructed-as-observed rejection + reserved observed identifier
# ---------------------------------------------------------------------------


def test_constructed_reference_asserting_observed_fails():
    # Tests R-002 [unit]: observed_open_ended: true fails validation on a
    # constructed-reference artifact.
    profile = make_profile()
    profile["semantic"]["observed_open_ended"] = True
    with pytest.raises(schema.SchemaValidationError):
        schema.validate_profile(profile)


def test_reserved_observed_profile_identifier_exists_and_differs():
    # Tests R-002 [unit]: a distinct future profile identifier is reserved
    # for genuinely observed studies.
    reserved = schema.RESERVED_OBSERVED_PROFILE_ID
    assert isinstance(reserved, str) and reserved
    assert reserved != STRICT_PROFILE_ID
    assert reserved == OBSERVED_PROFILE_ID


def test_constructed_reference_validator_never_accepts_observed_profile_id():
    # Tests R-002 [unit]: the constructed-reference validator never accepts
    # the reserved observed-profile identifier.
    profile = make_profile()
    profile["profile_id"] = OBSERVED_PROFILE_ID
    with pytest.raises(schema.SchemaValidationError):
        schema.validate_profile(profile)


# ---------------------------------------------------------------------------
# R-003: per-arm identification
# ---------------------------------------------------------------------------

ARM_REQUIRED_FIELDS = [
    "construction",
    "cardinality",
    "selector",
    "scorer",
    "candidate_pool_role",
    "correctness_assignment",
    "calibration_role",
    "continuation_role",
    "seed_contract",
    "reporting_eligibility",
]


@pytest.mark.parametrize("field", ARM_REQUIRED_FIELDS)
def test_arm_missing_identification_field_rejected(field):
    # Tests R-003 [unit]: every arm identifies construction, scalar-vs-K-way
    # status, selector/scorer, pool role, correctness assignment,
    # calibration/continuation role, seed contract, reporting eligibility.
    profile = make_profile()
    del profile["arms"][0][field]
    with pytest.raises(schema.SchemaValidationError):
        schema.validate_profile(profile)


def test_idealized_arm_declared_k_way_rejected():
    # Tests R-003 [unit]: a payload declaring all arms K-way while containing
    # an idealized (scalar prefix-to-gold cosine, oracle correctness) arm fails.
    profile = make_profile()
    bad_ref = make_idealized_arm("arm-ref")
    bad_ref["cardinality"] = "k_way"  # misdeclaration: idealized is scalar
    profile["arms"] = [profile["arms"][0], bad_ref]
    with pytest.raises(schema.SchemaValidationError):
        schema.validate_profile(profile)


def test_idealized_arm_correctly_declared_scalar_accepted():
    # Tests R-003 [unit]: the correctly declared idealized scalar arm passes.
    profile = make_profile()
    assert profile["arms"][1]["construction"] == "idealized"
    assert profile["arms"][1]["cardinality"] == "scalar"
    schema.validate_profile(profile)


# ---------------------------------------------------------------------------
# R-004: lossless roundtrip + allow_nan=False
# ---------------------------------------------------------------------------


def test_encode_decode_roundtrip_is_lossless():
    # Tests R-004 [unit]: decode(encode(p)) == p (equal value).
    profile = make_profile()
    assert schema.decode_profile(schema.encode_profile(profile)) == profile


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_writer_rejects_non_finite_floats(tmp_path: Path, bad):
    # Tests R-004 [unit]: writers reject non-finite floats at write time
    # (allow_nan=False semantics) — both encode and the create-once writer.
    profile = make_profile()
    profile["cells"][0]["rates"]["rate_both_finite"] = bad
    with pytest.raises((ValueError, schema.ColmAimsError)):
        schema.encode_profile(profile)
    with pytest.raises((ValueError, schema.ColmAimsError)):
        schema.write_profile(tmp_path / "profile.json", profile)
    assert not (tmp_path / "profile.json").exists()


def test_encoded_bytes_contain_no_nan_token():
    # Tests R-004 [unit]: valid profiles encode to strict JSON (parseable
    # with plain json.loads, no NaN/Infinity tokens).
    data = schema.encode_profile(make_profile())
    parsed = json.loads(data.decode("utf-8"))  # strict JSON parse
    assert not any(
        isinstance(v, float) and not math.isfinite(v)
        for v in parsed["cells"][0]["rates"].values()
    )


# ---------------------------------------------------------------------------
# R-020: typed ingress
# ---------------------------------------------------------------------------


def _write(tree: Path, name: str, data: bytes) -> Path:
    tree.mkdir(parents=True, exist_ok=True)
    p = tree / name
    p.write_bytes(data)
    return p


def test_malformed_json_produces_typed_error_naming_file(tmp_path: Path):
    # Tests R-020 [unit]: malformed bytes produce a typed error naming the
    # file by tree-relative path — no partial semantic processing.
    tree = tmp_path / "tree"
    path = _write(tree, "profile.json", b'{"schema_version": 1, "trunc')
    with pytest.raises(schema.TypedIngressError) as exc:
        schema.load_artifact(path, tree_root=tree)
    msg = str(exc.value)
    assert "profile.json" in msg
    # Relative identification (the form the R-026 sentinel test accepts):
    # no local absolute path in the message.
    assert str(tmp_path) not in msg


def test_truncated_records_line_produces_typed_error(tmp_path: Path):
    # Tests R-020 [unit]: a truncated JSONL line is a typed error naming the file.
    tree = tmp_path / "tree"
    good = json.dumps(standard_records()[0])
    path = _write(tree, "records.jsonl", (good + "\n{\"item_key\": \"itm-").encode())
    with pytest.raises(schema.TypedIngressError) as exc:
        schema.load_artifact(path, tree_root=tree)
    assert "records.jsonl" in str(exc.value)


def test_unknown_top_level_key_produces_typed_error_naming_field(tmp_path: Path):
    # Tests R-020 [unit]: unknown-keyed files produce a typed error naming
    # the file and the field — no silent key-dropping.
    tree = tmp_path / "tree"
    profile = make_profile()
    profile["surprise_field"] = 1
    path = _write(tree, "profile.json", json.dumps(profile).encode())
    with pytest.raises(schema.TypedIngressError) as exc:
        schema.load_artifact(path, tree_root=tree)
    msg = str(exc.value)
    assert "profile.json" in msg
    assert "surprise_field" in msg


def test_schema_version_checked_before_unknown_keys(tmp_path: Path):
    # Tests R-020 [unit]: schema_version is validated before any other check;
    # an unsupported version yields the version error (artifact version,
    # supported range, verifier revision) — never a generic unknown-key error.
    tree = tmp_path / "tree"
    profile = make_profile()
    profile["schema_version"] = 99
    profile["surprise_field"] = 1  # also present: must NOT win
    path = _write(tree, "profile.json", json.dumps(profile).encode())
    with pytest.raises(schema.TypedIngressError) as exc:
        schema.load_artifact(path, tree_root=tree)
    msg = str(exc.value)
    assert "99" in msg
    assert "schema_version" in msg
    assert "support" in msg.lower()  # names the supported range
    assert "surprise_field" not in msg
    # Names the matching verifier revision.
    assert "revision" in msg.lower() or "verifier" in msg.lower()


def test_missing_schema_version_is_typed_error(tmp_path: Path):
    # Tests R-020 [unit]: a file without schema_version fails typed ingress.
    tree = tmp_path / "tree"
    profile = make_profile()
    del profile["schema_version"]
    path = _write(tree, "profile.json", json.dumps(profile).encode())
    with pytest.raises(schema.TypedIngressError) as exc:
        schema.load_artifact(path, tree_root=tree)
    assert "schema_version" in str(exc.value)


def test_valid_artifact_loads_to_typed_value(tmp_path: Path):
    # Tests R-020 [unit]: well-formed artifact bytes load to an equal value.
    # Encodes via the REAL encoder (no fallback — audit ADV-9): while
    # encode_profile is a stub this fails RED on NotImplementedError.
    tree = tmp_path / "tree"
    profile = make_profile()
    path = _write(tree, "profile.json", schema.encode_profile(profile))
    loaded = schema.load_artifact(path, tree_root=tree)
    assert loaded["profile_id"] == STRICT_PROFILE_ID
    assert loaded["semantic"] == SEMANTIC_BLOCK


# ---------------------------------------------------------------------------
# R-029: llm_involvement block
# ---------------------------------------------------------------------------


def test_profile_missing_llm_involvement_rejected():
    # Tests R-029 [unit]: the validator rejects artifacts missing the block.
    profile = make_profile()
    del profile["llm_involvement"]
    with pytest.raises(schema.SchemaValidationError):
        schema.validate_profile(profile)


@pytest.mark.parametrize(
    "axis", ["reference_construction", "data_plot_creation", "evaluation"]
)
def test_llm_involvement_missing_axis_rejected(axis):
    # Tests R-029 [unit]: `none` is an explicit value, never an absent field —
    # a missing axis is rejected.
    profile = make_profile()
    del profile["llm_involvement"][axis]
    with pytest.raises(schema.SchemaValidationError):
        schema.validate_profile(profile)


def test_llm_involvement_non_none_axis_requires_tool_note():
    # Tests R-029 [unit]: a non-none axis requires a free-text tool/version note.
    profile = make_profile()
    profile["llm_involvement"] = make_llm_involvement(
        data_plot_creation="assisted"
    )
    with pytest.raises(schema.SchemaValidationError):
        schema.validate_profile(profile)
    profile["llm_involvement"]["tool_version_note"] = "example-llm v1.2 for plots"
    schema.validate_profile(profile)


def test_llm_involvement_all_none_is_valid():
    # Tests R-029 [unit]: explicit `none` on every axis validates.
    schema.validate_profile(make_profile())


# ---------------------------------------------------------------------------
# R-031: non-reversible per-item records
# ---------------------------------------------------------------------------


def test_record_with_free_text_field_rejected():
    # Tests R-031 [unit]: a record with a free-text field FAILS (fixture from
    # the spec, RF-01). The synthetic text below is not quizbowl content.
    record = make_record("itm-0001", 1, 3, question_text="free text leaks here")
    with pytest.raises(schema.RecordValidationError):
        schema.validate_record(record)


def test_record_with_unlisted_string_field_rejected():
    # Tests R-031 [unit]: any string field outside the enumerated identifier
    # allowlist is rejected.
    record = make_record("itm-0001", 1, 3, note="operator remark")
    with pytest.raises(schema.RecordValidationError):
        schema.validate_record(record)


def test_clean_numeric_record_accepted():
    # Tests R-031 [unit]: opaque key + numeric fields validate.
    schema.validate_record(make_record("itm-0001", 1, 3))


def test_excluded_record_with_enumerated_reason_accepted():
    # Tests R-031 [unit]: enumerated categorical fields are allowed.
    record = make_record(
        "itm-0009", None, 3, excluded=True, exclusion_reason="MALFORMED_STOP"
    )
    schema.validate_record(record)


def test_excluded_record_with_free_text_reason_rejected():
    # Tests R-031 [unit]: exclusion_reason must be an enumerated categorical,
    # not free text.
    record = make_record(
        "itm-0009",
        None,
        3,
        excluded=True,
        exclusion_reason="the parser hiccuped on this one somehow",
    )
    with pytest.raises(schema.RecordValidationError):
        schema.validate_record(record)


# ---------------------------------------------------------------------------
# R-032: maximum admissible tolerance
# ---------------------------------------------------------------------------


def test_max_admissible_tolerance_is_pinned_and_small():
    # Tests R-032 [unit]: the schema pins a maximum admissible numerical
    # tolerance. DECISION: it must be positive and at most 1e-3 (joint-class
    # rates sum to 1; anything looser is meaningless).
    max_tol = schema.MAX_ADMISSIBLE_TOLERANCE
    assert 0 < max_tol <= 1e-3


def _set_tolerance(profile, value) -> None:
    """Set the declared tolerance and recompute the estimand digest so
    tolerance tests cannot pass/fail for digest-staleness reasons (ADV-3)."""
    profile["numerical_tolerance"] = value
    cell = profile["cells"][0]
    cell["estimand"]["numerical_tolerance"] = value
    cell["estimand_digest"] = expected_estimand_digest(cell["estimand"])


def test_oversized_declared_tolerance_fails_validation():
    # Tests R-032 [unit]: a declared tolerance above the pinned maximum fails
    # (digest recomputed, so the tolerance gate is the only defect).
    profile = make_profile()
    _set_tolerance(profile, 0.25)
    with pytest.raises(schema.SchemaValidationError):
        schema.validate_profile(profile)


def test_tolerance_at_maximum_boundary_passes():
    # Tests R-032 [unit]: red/green both sides of the pinned boundary
    # (digest recomputed on both sides — ADV-3).
    max_tol = schema.MAX_ADMISSIBLE_TOLERANCE
    profile = make_profile()
    _set_tolerance(profile, max_tol)
    schema.validate_profile(profile)
    _set_tolerance(profile, max_tol * 2)
    with pytest.raises(schema.SchemaValidationError):
        schema.validate_profile(profile)
