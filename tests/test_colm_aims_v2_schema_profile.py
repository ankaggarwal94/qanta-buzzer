"""Schema/profile rules: R-001..R-004, R-029, R-031, R-032, R-057, R-058,
R-063 (profile-side closed maps + AST), R-034, R-028 (import scans), R-016
(create-once publish primitives).

GREENFIELD RED: imports the production namespace at module scope — this file
FAILS AT COLLECTION until the v2 reimplementation exists (expected).
"""
from __future__ import annotations

import ast
import json
import re

import pytest

from reproducibility.colm_aims_2026 import schema

from tests._colm_aims_v2_helpers import (
    AMBIGUOUS_TERMINAL_SENTINEL,
    ANALYSIS_PROVENANCE_D7B,
    EVENT_FINITE,
    EVENT_NEVER,
    FAMILY_STOP_VOCAB,
    OBSERVED_PROFILE_ID,
    SANCTIONED_OBSERVED_CLAIM_OUTPUT,
    SEMANTIC_BLOCK,
    STRICT_PROFILE_ID,
    VERIFIER_REVISION,
    colm_no_network,  # noqa: F401 - autouse fixture
    make_arm,
    make_idealized_arm,
    make_llm_involvement,
    make_profile_v2,
    make_record_v2,
    namespace_py_files,
)


# ---------------------------------------------------------------------------
# R-058: one canonical revision constant set; NAMESPACE_REVISION deleted
# ---------------------------------------------------------------------------


class TestRevisionConstants:
    def test_constant_set_exact(self):
        assert schema.SCHEMA_VERSION == 2
        assert schema.SUPPORTED_SCHEMA_VERSION_MIN == 2
        assert schema.SUPPORTED_SCHEMA_VERSION_MAX == 2
        assert schema.VERIFIER_REVISION == VERIFIER_REVISION

    def test_schema_version_is_real_int_not_bool(self):
        assert type(schema.SCHEMA_VERSION) is int

    def test_namespace_revision_deleted_or_deprecated_alias(self):
        # R-058: NAMESPACE_REVISION is DELETED; a temporary alias must be
        # exactly `NAMESPACE_REVISION = VERIFIER_REVISION` (never an
        # independently maintained second literal).
        if hasattr(schema, "NAMESPACE_REVISION"):
            assert schema.NAMESPACE_REVISION == schema.VERIFIER_REVISION
            src = (schema.__file__ and open(schema.__file__).read()) or ""
            tree = ast.parse(src)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for tgt in node.targets:
                        if (
                            isinstance(tgt, ast.Name)
                            and tgt.id == "NAMESPACE_REVISION"
                        ):
                            assert isinstance(node.value, ast.Name), (
                                "NAMESPACE_REVISION must alias"
                                " VERIFIER_REVISION by NAME, not carry its"
                                " own literal (R-058)"
                            )
                            assert node.value.id == "VERIFIER_REVISION"

    def test_no_second_literal_revision_token_in_namespace(self):
        # No per-surface version-constant forks: the ONLY revision-shaped
        # string literal in the namespace is the canonical token.
        pattern = re.compile(r"reproducibility\.colm_aims_2026[: ]r\d+")
        legacy_pattern = re.compile(r"colm_aims_2026\.r\d+")
        for path in namespace_py_files():
            tree = ast.parse(path.read_text("utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(
                    node.value, str
                ):
                    for match in pattern.findall(node.value):
                        assert match == VERIFIER_REVISION, (
                            f"{path.name}: second revision literal"
                            f" {match!r} (R-058)"
                        )
                    assert not legacy_pattern.search(node.value), (
                        f"{path.name}: legacy v1 revision token in"
                        f" {node.value!r} (R-058)"
                    )

    def test_verifier_revision_assigned_exactly_once(self):
        assignments = 0
        for path in namespace_py_files():
            tree = ast.parse(path.read_text("utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for tgt in node.targets:
                        if (
                            isinstance(tgt, ast.Name)
                            and tgt.id == "VERIFIER_REVISION"
                            and isinstance(node.value, ast.Constant)
                        ):
                            assignments += 1
        assert assignments == 1


# ---------------------------------------------------------------------------
# R-001: strict v2 profile shape + pinned semantic block
# ---------------------------------------------------------------------------


class TestStrictProfile:
    def test_canonical_profile_validates(self):
        schema.validate_profile(make_profile_v2())

    def test_profile_id_constant_pinned(self):
        assert schema.STRICT_PROFILE_ID == STRICT_PROFILE_ID

    @pytest.mark.parametrize("field", sorted(SEMANTIC_BLOCK))
    def test_semantic_field_missing_rejected(self, field):
        profile = make_profile_v2()
        del profile["semantic"][field]
        with pytest.raises(schema.SchemaValidationError):
            schema.validate_profile(profile)

    @pytest.mark.parametrize("field", sorted(SEMANTIC_BLOCK))
    def test_semantic_field_altered_rejected(self, field):
        profile = make_profile_v2()
        value = profile["semantic"][field]
        profile["semantic"][field] = (
            (not value) if isinstance(value, bool) else value + "_altered"
        )
        with pytest.raises(schema.SchemaValidationError):
            schema.validate_profile(profile)

    def test_semantic_field_renamed_rejected(self):
        profile = make_profile_v2()
        profile["semantic"]["trajectory_src"] = profile["semantic"].pop(
            "trajectory_source"
        )
        with pytest.raises(schema.SchemaValidationError):
            schema.validate_profile(profile)

    def test_semantic_unknown_key_rejected(self):
        profile = make_profile_v2()
        profile["semantic"]["extra_axis"] = "surprise"
        with pytest.raises(schema.SchemaValidationError):
            schema.validate_profile(profile)

    @pytest.mark.parametrize("key", ["grid", "inference", "cells", "arms"])
    def test_required_top_level_key_missing_rejected(self, key):
        profile = make_profile_v2()
        del profile[key]
        with pytest.raises(schema.SchemaValidationError):
            schema.validate_profile(profile)

    def test_unknown_top_level_key_rejected(self):
        profile = make_profile_v2()
        profile["bonus_block"] = {"x": 1}
        with pytest.raises(schema.SchemaValidationError):
            schema.validate_profile(profile)

    def test_calibration_identity_must_be_map_with_exact_keys(self):
        # D1: scalar -> map migration; the v1 scalar shape is REJECTED.
        profile = make_profile_v2()
        profile["provenance"]["calibration_identity"] = "cal-0001"
        with pytest.raises(schema.SchemaValidationError):
            schema.validate_profile(profile)

    def test_calibration_identity_map_extra_key_rejected(self):
        profile = make_profile_v2()
        profile["provenance"]["calibration_identity"]["pooled"] = "cal-x"
        with pytest.raises(schema.SchemaValidationError):
            schema.validate_profile(profile)

    def test_calibration_identity_map_missing_key_rejected(self):
        profile = make_profile_v2()
        del profile["provenance"]["calibration_identity"]["shared"]
        with pytest.raises(schema.SchemaValidationError):
            schema.validate_profile(profile)

    def test_legacy_format_qa_identifier_never_substitutes(self):
        # R-001: a `format: "QA"` identifier cannot stand in for the
        # semantic block in the strict path.
        profile = make_profile_v2()
        del profile["semantic"]
        profile["format"] = "QA"
        with pytest.raises(schema.SchemaValidationError):
            schema.validate_profile(profile)


# ---------------------------------------------------------------------------
# R-002: observed-profile firewall
# ---------------------------------------------------------------------------


class TestObservedProfileFirewall:
    def test_reserved_observed_id_is_distinct_constant(self):
        assert schema.RESERVED_OBSERVED_PROFILE_ID == OBSERVED_PROFILE_ID
        assert schema.RESERVED_OBSERVED_PROFILE_ID != schema.STRICT_PROFILE_ID

    def test_observed_open_ended_true_rejected(self):
        profile = make_profile_v2()
        profile["semantic"]["observed_open_ended"] = True
        with pytest.raises(schema.SchemaValidationError):
            schema.validate_profile(profile)

    def test_observed_profile_id_rejected(self):
        profile = make_profile_v2(profile_id=OBSERVED_PROFILE_ID)
        with pytest.raises(schema.SchemaValidationError):
            schema.validate_profile(profile)

    def test_observed_id_with_observed_semantics_still_rejected(self):
        # The constructed-reference validator never accepts the reserved
        # observed profile even in its "honest" observed form.
        profile = make_profile_v2(profile_id=OBSERVED_PROFILE_ID)
        profile["semantic"]["observed_open_ended"] = True
        profile["semantic"]["observed_open_ended_answers"] = True
        profile["semantic"]["observed_open_ended_stopping_actions"] = True
        with pytest.raises(schema.SchemaValidationError):
            schema.validate_profile(profile)

    def test_sanctioned_observed_claim_output_token_pinned(self):
        assert (
            schema.OBSERVED_PAIRED_CLAIM_OUTPUT
            == SANCTIONED_OBSERVED_CLAIM_OUTPUT
        )


# ---------------------------------------------------------------------------
# R-003: arm identities, per-family closed vocabularies
# ---------------------------------------------------------------------------

ARM_REQUIRED_FIELDS = (
    "arm_id",
    "family",
    "stop_semantics",
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
)


class TestArms:
    @pytest.mark.parametrize("field", ARM_REQUIRED_FIELDS)
    def test_arm_missing_field_rejected(self, field):
        profile = make_profile_v2()
        del profile["arms"][0][field]
        with pytest.raises(schema.SchemaValidationError):
            schema.validate_profile(profile)

    def test_all_kway_claim_with_idealized_arm_rejected(self):
        # A payload declaring all arms K-way while containing an idealized
        # arm fails (idealized is scalar prefix-to-gold cosine).
        profile = make_profile_v2()
        for arm in profile["arms"]:
            arm["cardinality"] = "k_way"
        with pytest.raises(schema.SchemaValidationError):
            schema.validate_profile(profile)

    def test_idealized_arm_must_be_oracle_assigned(self):
        profile = make_profile_v2()
        for arm in profile["arms"]:
            if arm["construction"] == "idealized":
                arm["correctness_assignment"] = "option_match"
        with pytest.raises(schema.SchemaValidationError):
            schema.validate_profile(profile)

    @pytest.mark.parametrize(
        ("family", "wrong_family"),
        [
            ("constructed_reference", "myopic"),
            ("fixed_threshold", "learned_continuation"),
            ("myopic", "constructed_reference"),
            ("learned_continuation", "fixed_threshold"),
        ],
    )
    def test_cross_family_stop_vocabulary_confusion_rejected(
        self, family, wrong_family
    ):
        # No overloaded global stop integer/vocabulary across families:
        # family X carrying family Y's stop-semantics token fails.
        profile = make_profile_v2()
        arm = make_arm("confused-arm", family=family)
        arm["stop_semantics"] = FAMILY_STOP_VOCAB[wrong_family]
        profile["arms"].append(arm)
        with pytest.raises(schema.SchemaValidationError):
            schema.validate_profile(profile)

    @pytest.mark.parametrize("family", sorted(FAMILY_STOP_VOCAB))
    def test_each_family_vocabulary_accepted(self, family):
        # Nearest-true positive per family.
        profile = make_profile_v2()
        arm = (
            make_idealized_arm("extra-idealized")
            if family == "constructed_reference"
            else make_arm(f"extra-{family}", family=family)
        )
        profile["arms"].append(arm)
        schema.validate_profile(profile)

    def test_unknown_family_rejected(self):
        profile = make_profile_v2()
        profile["arms"][0]["family"] = "global_stop_integer"
        with pytest.raises(schema.SchemaValidationError):
            schema.validate_profile(profile)


# ---------------------------------------------------------------------------
# R-004: lossless roundtrip; non-finite floats rejected at write time
# ---------------------------------------------------------------------------


class TestRoundtrip:
    def test_encode_decode_roundtrip_equal_value(self):
        profile = make_profile_v2()
        assert schema.decode_profile(schema.encode_profile(profile)) == profile

    def test_writer_rejects_nan_at_write_time(self):
        profile = make_profile_v2()
        profile["cells"][0]["headline_summary"]["mean_signed_shift"] = float(
            "nan"
        )
        with pytest.raises((schema.ColmAimsError, ValueError)):
            schema.encode_profile(profile)

    def test_writer_rejects_infinity_at_write_time(self):
        profile = make_profile_v2()
        profile["cells"][0]["interval"]["ci"][1] = float("inf")
        with pytest.raises((schema.ColmAimsError, ValueError)):
            schema.encode_profile(profile)


# ---------------------------------------------------------------------------
# R-029: llm_involvement block
# ---------------------------------------------------------------------------


class TestLlmInvolvement:
    def test_explicit_none_axes_accepted(self):
        schema.validate_profile(make_profile_v2())

    def test_missing_block_rejected(self):
        profile = make_profile_v2()
        del profile["llm_involvement"]
        with pytest.raises(schema.SchemaValidationError):
            schema.validate_profile(profile)

    @pytest.mark.parametrize(
        "axis", ["reference_construction", "data_plot_creation", "evaluation"]
    )
    def test_absent_axis_rejected_none_must_be_explicit(self, axis):
        profile = make_profile_v2()
        del profile["llm_involvement"][axis]
        with pytest.raises(schema.SchemaValidationError):
            schema.validate_profile(profile)

    def test_non_none_axis_requires_tool_note(self):
        profile = make_profile_v2(
            llm_involvement=make_llm_involvement(evaluation="assisted")
        )
        with pytest.raises(schema.SchemaValidationError):
            schema.validate_profile(profile)

    def test_non_none_axis_with_tool_note_accepted(self):
        block = make_llm_involvement(evaluation="assisted")
        block["tool_version_note"] = "tiny-scorer 0.0.1 via local CLI"
        profile = make_profile_v2(llm_involvement=block)
        schema.validate_profile(profile)


# ---------------------------------------------------------------------------
# R-032: maximum admissible numerical tolerance
# ---------------------------------------------------------------------------


class TestTolerance:
    def test_max_admissible_constant_exists(self):
        assert 0 < schema.MAX_ADMISSIBLE_TOLERANCE < 1

    def test_tolerance_above_max_rejected(self):
        profile = make_profile_v2(
            numerical_tolerance=schema.MAX_ADMISSIBLE_TOLERANCE * 10
        )
        with pytest.raises(schema.SchemaValidationError):
            schema.validate_profile(profile)

    def test_tolerance_at_max_accepted(self):
        profile = make_profile_v2(
            numerical_tolerance=schema.MAX_ADMISSIBLE_TOLERANCE
        )
        for cell in profile["cells"]:
            cell["estimand"]["numerical_tolerance"] = (
                schema.MAX_ADMISSIBLE_TOLERANCE
            )
            # keep the recorded digest consistent with the mutated estimand
            from tests._colm_aims_v2_helpers import expected_estimand_digest

            cell["estimand_digest"] = expected_estimand_digest(
                cell["estimand"]
            )
        schema.validate_profile(profile)


# ---------------------------------------------------------------------------
# R-031: non-reversible per-item records (v2 event allowlist)
# ---------------------------------------------------------------------------


class TestRecordAllowlist:
    def test_canonical_finite_record_valid(self):
        schema.validate_record(make_record_v2("itm-0001", 2, 3))

    def test_canonical_never_stopped_record_valid(self):
        schema.validate_record(make_record_v2("itm-0001", None, 4))

    def test_free_text_field_rejected(self):
        rec = make_record_v2("itm-0001", 2, 3, note="this stop felt late")
        with pytest.raises(schema.RecordValidationError):
            schema.validate_record(rec)

    def test_string_outside_identifier_allowlist_rejected(self):
        rec = make_record_v2("itm-0001", 2, 3)
        rec["source_text"] = "What president..."
        with pytest.raises(schema.RecordValidationError):
            schema.validate_record(rec)

    def test_event_status_outside_enum_rejected(self):
        rec = make_record_v2("itm-0001", 2, 3)
        rec["mc_event_status"] = "TIMED_OUT"
        with pytest.raises(schema.RecordValidationError):
            schema.validate_record(rec)

    def test_terminal_imputation_outside_enum_rejected(self):
        rec = make_record_v2("itm-0001", None, 3)
        rec["mc_terminal_imputation"] = "LAST_OBSERVED"
        with pytest.raises(schema.RecordValidationError):
            schema.validate_record(rec)

    def test_ambiguous_terminal_sentinel_is_enum_member(self):
        # R-047: spec-pinned new member of the exclusion-reason enum.
        assert AMBIGUOUS_TERMINAL_SENTINEL in schema.EXCLUSION_REASONS

    def test_event_status_enum_closed_pair(self):
        assert schema.EVENT_STATUSES == frozenset({EVENT_FINITE, EVENT_NEVER})


# ---------------------------------------------------------------------------
# R-057: analysis-provenance discriminator token
# ---------------------------------------------------------------------------


class TestAnalysisProvenance:
    def test_exact_token_pinned_at_schema(self):
        assert schema.ANALYSIS_PROVENANCE_D7B == ANALYSIS_PROVENANCE_D7B

    def test_inference_block_missing_discriminator_rejected(self):
        profile = make_profile_v2()
        del profile["inference"]["analysis_provenance"]
        with pytest.raises(schema.SchemaValidationError):
            schema.validate_profile(profile)

    def test_inference_block_wrong_discriminator_rejected(self):
        # A recovered/original-analysis claim is not a valid discriminator.
        profile = make_profile_v2()
        profile["inference"]["analysis_provenance"] = (
            "recovered_original_2019_analysis"
        )
        with pytest.raises(schema.SchemaValidationError):
            schema.validate_profile(profile)


# ---------------------------------------------------------------------------
# R-063: closed maps (profile-side) + AST no-defaulted-get on trusted keys
# ---------------------------------------------------------------------------

TRUSTED_BLOCK_KEYS = {
    "anchor",
    "bindings",
    "grid",
    "inference",
    "held_fixed",
    "ledger_path",
    "ledger_sha256",
    "source_commit",
    "seed",
    "seed_derivation",
    "resample_matrix_digest",
    "item_keys_sha256",
    "pairing_population_keyset_sha256",
    "canonical_item_order_digest",
    "record_files",
    "reference_ids",
    "calibration_ids",
    "cell_ids",
    "external_claim_ids",
    "rights_inventory",
    "tree_files",
}


class TestClosedMaps:
    @pytest.mark.parametrize(
        ("key", "derivation", "expected"),
        [
            ("itm-0123456789abcdef", schema.ITEM_KEY_DERIVATION, True),
            ("itm-0123", schema.ITEM_KEY_DERIVATION, False),
            ("10015", schema.PHASE4_ITEM_KEY_DERIVATION, True),
            ("010015", schema.PHASE4_ITEM_KEY_DERIVATION, False),
            ("itm-0123456789abcdef", schema.PHASE4_ITEM_KEY_DERIVATION, False),
        ],
    )
    def test_item_keys_conform_to_exact_declared_scheme(
        self, key, derivation, expected
    ):
        assert schema.item_key_conforms_to_derivation(key, derivation) is expected

    @pytest.mark.parametrize(
        "block_path",
        [
            ("item_key_derivation",),
            ("grid",),
            ("inference",),
            ("grid", "held_fixed"),
        ],
    )
    def test_unknown_key_in_trusted_block_rejected(self, block_path):
        profile = make_profile_v2()
        target = profile
        for part in block_path:
            target = target[part]
        target["misspelled_extra"] = "typo"
        with pytest.raises(schema.SchemaValidationError):
            schema.validate_profile(profile)

    def test_estimand_timeout_parameters_unknown_key_rejected(self):
        profile = make_profile_v2()
        profile["cells"][0]["estimand"]["timeout_parameters"][
            "wall_clock_cap"
        ] = 10
        with pytest.raises(schema.SchemaValidationError):
            schema.validate_profile(profile)

    def test_event_representation_unknown_key_rejected(self):
        profile = make_profile_v2()
        profile["cells"][0]["estimand"]["event_representation"][
            "sentinal_convention"  # the typo class R-063 exists to catch
        ] = "timeout_coded_as_horizon"
        with pytest.raises(schema.SchemaValidationError):
            schema.validate_profile(profile)

    @pytest.mark.parametrize(
        "parts",
        [
            ("arms", 0),
            ("arms", 0, "seed_contract"),
            ("cells", 0),
            ("cells", 0, "counts"),
            ("cells", 0, "rates"),
            ("cells", 0, "headline_summary"),
            ("cells", 0, "finite_only_summary"),
            ("cells", 0, "interval"),
            ("provenance",),
            ("provenance", "dirty_state"),
            ("provenance", "splits"),
            ("provenance", "splits", "eval"),
            ("provenance", "pre_package_retention"),
            ("provenance", "mc_build"),
            ("provenance", "model"),
            ("provenance", "model", "numerical_settings"),
        ],
    )
    def test_every_nested_trusted_map_rejects_unknown_keys(self, parts):
        profile = make_profile_v2()
        target = profile
        for part in parts:
            target = target[part]
        target["misspelled_extra"] = "typo"
        with pytest.raises(schema.SchemaValidationError):
            schema.validate_profile(profile)

    def test_provenance_model_must_be_object(self):
        profile = make_profile_v2()
        profile["provenance"]["model"] = None
        with pytest.raises(schema.SchemaValidationError, match="model"):
            schema.validate_profile(profile)

    def test_null_revision_requires_byte_digest_manifest(self):
        profile = make_profile_v2()
        profile["provenance"]["model"]["revision"] = None
        with pytest.raises(schema.SchemaValidationError, match="byte_digest"):
            schema.validate_profile(profile)

    def test_source_commit_must_be_native_string(self):
        profile = make_profile_v2()
        profile["provenance"]["dirty_state"]["source_commit"] = int("1" * 40)
        with pytest.raises(schema.SchemaValidationError, match="source_commit"):
            schema.validate_profile(profile)

    def test_no_defaulted_dict_get_on_trusted_keys(self):
        # The v1 R1 lesson: anchor.get("ledger_path", ...) silently
        # defaulted on a typo. No namespace code may call
        # <obj>.get("<trusted key>", <default>).
        offenders: list[str] = []
        for path in namespace_py_files():
            tree = ast.parse(path.read_text("utf-8"))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "get"
                    and len(node.args) == 2
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value in TRUSTED_BLOCK_KEYS
                ):
                    offenders.append(
                        f"{path.name}:{node.lineno} .get({node.args[0].value!r}, ...)"
                    )
        assert not offenders, f"defaulted get on trusted keys: {offenders}"


# ---------------------------------------------------------------------------
# R-034 / R-028: deserialization safety + no-network import scans
# ---------------------------------------------------------------------------

DENYLIST_IMPORTS = {
    "pickle",
    "marshal",
    "dill",
    "shelve",
    "yaml",
    "torch",
    "requests",
    "httpx",
    "huggingface_hub",
    "transformers",
}


def _imported_module_roots() -> set[str]:
    roots: set[str] = set()
    for path in namespace_py_files():
        tree = ast.parse(path.read_text("utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    roots.add(alias.name)
                    roots.add(alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.module:
                roots.add(node.module)
                roots.add(node.module.split(".")[0])
    return roots


class TestImportScans:
    def test_namespace_exists_and_nonempty(self):
        files = namespace_py_files()
        assert files, "reproducibility/colm_aims_2026/ has no python modules"

    def test_no_unsafe_deserialization_or_network_imports(self):
        roots = _imported_module_roots()
        hits = sorted(roots & DENYLIST_IMPORTS)
        assert not hits, f"denied imports in namespace: {hits}"

    def test_urllib_request_specifically_denied(self):
        roots = _imported_module_roots()
        assert "urllib.request" not in roots

    def test_numpy_required_and_imported(self):
        # D5/R-051: NumPy 2.4.6 is REQUIRED (not on the deny-list) — the
        # inference implementation must import it somewhere.
        assert "numpy" in _imported_module_roots()


# ---------------------------------------------------------------------------
# R-016: create-once publish primitives (consumed, not forked)
# ---------------------------------------------------------------------------


class TestCreateOncePublish:
    def _stage(self, tmp_path):
        staged = tmp_path / "staged"
        # TEST_BUG fix (orchestrator-adjudicated): callers pass fresh subdirs
        # (tmp_path / "again"), so the parent must be created too.
        staged.mkdir(parents=True)
        (staged / "profile.json").write_text(
            json.dumps({"schema_version": 2}), encoding="utf-8"
        )
        return staged

    def test_second_publish_to_existing_run_id_fails(self, tmp_path):
        staged = self._stage(tmp_path)
        runs_root = tmp_path / "runs"
        schema.publish_evidence_package(staged, runs_root, "run-0001")
        staged2 = self._stage(tmp_path / "again")
        with pytest.raises(schema.ColmAimsError):
            schema.publish_evidence_package(staged2, runs_root, "run-0001")

    def test_publish_traversal_run_id_rejected(self, tmp_path):
        staged = self._stage(tmp_path)
        with pytest.raises(schema.ColmAimsError):
            schema.publish_evidence_package(
                staged, tmp_path / "runs", "../escape"
            )

    def test_interrupted_publish_leaves_no_partial_then_retry_succeeds(
        self, tmp_path, monkeypatch
    ):
        import os as _os

        staged = self._stage(tmp_path)
        runs_root = tmp_path / "runs"
        real_rename = _os.rename
        calls = {"n": 0}

        def failing_rename(src, dst, *a, **k):
            calls["n"] += 1
            raise OSError("simulated crash mid-publish (kill point)")

        monkeypatch.setattr(_os, "rename", failing_rename)
        with pytest.raises((schema.ColmAimsError, OSError)):
            schema.publish_evidence_package(staged, runs_root, "run-0002")
        monkeypatch.setattr(_os, "rename", real_rename)
        # No parseable partial artifact at the final path:
        final = runs_root / "run-0002"
        assert not (final / "profile.json").exists()
        # Retry (via the explicit recovery path if a claimed-empty relic
        # remains) produces exactly one resulting artifact.
        staged3 = self._stage(tmp_path / "retry")
        try:
            schema.publish_evidence_package(staged3, runs_root, "run-0002")
        except schema.ColmAimsError:
            schema.publish_evidence_package(
                staged3, runs_root, "run-0002", reclaim_crashed_relic=True
            )
        assert (final / "profile.json").is_file()
        published = list(runs_root.glob("run-0002*/profile.json"))
        assert len(published) == 1
