"""Contract tests for the production real-input driver (phase4_driver_d7b).

The driver is the FIRST production reader that constructs the caller-supplied
identity blocks (arms/provenance/grid/llm_involvement/estimands + the
parameterized D6 baseline + the QA-012 manifest) from the REAL frozen inputs
and a records root, then calls ``phase4_assemble_d7b.build_evidence_package``
to emit the real ``profile.json`` + evidence package + closure inventory.

FP/FN independence: the driver's reconstructed identity blocks are asserted
against the independent pure-oracle builders in ``tests/_colm_aims_v2_helpers``
(``make_arms``/``make_grid_block``/``make_estimand``/``canonical_horizon_identity``),
never against the production shapers the driver calls. The strongest leg is the
end-to-end ``verify --mode source`` recompute over a package the driver builds
from a synthetic records root + the REAL in-repo frozen inputs.

The driver is fully testable WITHOUT a certified ceremony and WITHOUT any
model inference: the records root is the same canonical synthetic data the
assembler tests use, and the D6 baseline is a fixture FINAL_CHECKSUMS file
pinned to the currently designated ``main.tex``/``main.pdf`` hashes.
"""
from __future__ import annotations

import json

import pytest

from reproducibility.colm_aims_2026 import closure, schema

from reproducibility.colm_aims_2026 import phase4_driver_d7b as driver

from tests._colm_aims_v2_helpers import (
    CANONICAL_KEYSET_SHA256,
    CELL_IDS,
    D6_MAIN_PDF_SHA256,
    D6_MAIN_TEX_SHA256,
    EXIT_INGRESS_ERROR,
    EXIT_PASS,
    EXIT_USAGE_ERROR,
    NAMESPACE_DIR,
    VERDICT_SOURCE_PASS,
    canonical_data,
    canonical_horizon_identity,
    colm_no_network,  # noqa: F401 - autouse fixture
    make_arms,
    make_estimand,
    make_grid_block,
    make_llm_involvement,
    make_profile_v2,
    run_cli,
    sha256_bytes,
)

# The REAL in-repo frozen inputs (never regenerated here).
FROZEN_DIR = NAMESPACE_DIR / "frozen"

# Frozen model manifest primary-scorer facts (frozen/model_snapshot_manifests.json).
PRIMARY_SCORER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PRIMARY_SCORER_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
PRIMARY_SCORER_WEIGHTS_SHA256 = (
    "53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db"
)
PRIMARY_SCORER_TOKENIZER_CONFIG_SHA256 = (
    "acb92769e8195aabd29b7b2137a9e6d6e25c476a4f15aa4355c233426c61576b"
)

# frozen/pairing_eligibility_v2.json declared counts (2,249 eligible + 9 excluded).
FROZEN_ELIGIBLE_COUNT = 2249
FROZEN_EXCLUDED_COUNT = 9


# ---------------------------------------------------------------------------
# Fixtures — records root / frozen D6 checksums / QA-012 corpus
# ---------------------------------------------------------------------------


def _complete_by_cell() -> dict[str, dict[str, dict]]:
    data = canonical_data()
    complete: dict[str, dict[str, dict]] = {}
    for cell_id in CELL_IDS:
        records: dict[str, dict] = {}
        blob = data["cells"][cell_id].records_bytes.decode("utf-8")
        for line in blob.splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            records[rec["item_key"]] = rec
        complete[cell_id] = records
    return complete


def _write_records_root(root):
    data = canonical_data()
    root.mkdir(parents=True, exist_ok=True)
    for cell_id in CELL_IDS:
        (root / f"{cell_id}.jsonl").write_bytes(
            data["cells"][cell_id].records_bytes
        )
    return root


def _final_checksums_entries() -> dict[str, str]:
    return {
        "main.tex": D6_MAIN_TEX_SHA256,
        "main.pdf": D6_MAIN_PDF_SHA256,
        "figures/fig1.pdf": "b" * 64,
        "references.bib": "c" * 64,
    }


def _write_final_checksums_json(path) -> dict[str, str]:
    entries = _final_checksums_entries()
    path.write_bytes(
        (json.dumps(entries, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    return entries


def _write_final_checksums_text(path) -> dict[str, str]:
    """sha256sum output form: ``<64-hex>  <relpath>`` lines."""
    entries = _final_checksums_entries()
    lines = [f"{sha}  {rel}" for rel, sha in sorted(entries.items())]
    path.write_bytes(("\n".join(lines) + "\n").encode("utf-8"))
    return entries


def _write_qa012_corpus(root):
    """A tiny zero-hit QA-012 corpus (no ``format:"QA"`` occurrences)."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "note.json").write_bytes(
        (json.dumps({"kind": "note", "value": 1}) + "\n").encode("utf-8")
    )
    return root


# ---------------------------------------------------------------------------
# record_horizon_identity — recomputed from records, fail-closed cross-cell
# ---------------------------------------------------------------------------


class TestRecordHorizonIdentity:
    def test_matches_canonical_horizon_identity(self):
        digest = driver.record_horizon_identity(_complete_by_cell())
        assert digest == canonical_horizon_identity()

    def test_fail_closed_when_cells_disagree(self):
        complete = _complete_by_cell()
        victim = CELL_IDS[0]
        # Shift every horizon in ONE cell so its per-item digest diverges.
        for rec in complete[victim].values():
            rec["trajectory_horizon"] = rec["trajectory_horizon"] + 1
        with pytest.raises(schema.ColmAimsError):
            driver.record_horizon_identity(complete)


# ---------------------------------------------------------------------------
# Identity blocks — reconstruct the canonical study design (no test imports)
# ---------------------------------------------------------------------------


class TestIdentityBlocks:
    def test_arms_match_canonical(self):
        assert driver.build_arms() == make_arms()

    def test_llm_involvement_all_none(self):
        assert driver.build_llm_involvement() == make_llm_involvement()

    def test_grid_matches_canonical(self):
        grid = driver.build_grid_block(
            keyset_digest=CANONICAL_KEYSET_SHA256,
            horizon_identity=canonical_horizon_identity(),
        )
        assert grid == make_grid_block()

    def test_estimands_match_canonical(self):
        estimands = driver.build_estimands(
            horizon_identity=canonical_horizon_identity()
        )
        assert set(estimands) == set(CELL_IDS)
        for cell_id in CELL_IDS:
            assert estimands[cell_id] == make_estimand(cell_id)


# ---------------------------------------------------------------------------
# provenance — model from the frozen manifest, retention from eligibility
# ---------------------------------------------------------------------------


class TestProvenanceFromFrozen:
    def _prov(self):
        return driver.build_provenance_from_frozen(
            FROZEN_DIR, keyset_digest=CANONICAL_KEYSET_SHA256
        )

    def test_model_maps_primary_scorer(self):
        model = self._prov()["model"]
        assert model["repository_namespace"] == PRIMARY_SCORER_MODEL
        assert model["revision"] == PRIMARY_SCORER_REVISION
        assert model["weights_sha256"] == PRIMARY_SCORER_WEIGHTS_SHA256
        assert (
            model["tokenizer_config_sha256"]
            == PRIMARY_SCORER_TOKENIZER_CONFIG_SHA256
        )

    def test_pre_package_retention_from_eligibility(self):
        retention = self._prov()["pre_package_retention"]
        assert retention == {
            "retained_count": FROZEN_ELIGIBLE_COUNT + FROZEN_EXCLUDED_COUNT,
            "paired_count": FROZEN_ELIGIBLE_COUNT,
            "upstream_unpaired_count": FROZEN_EXCLUDED_COUNT,
        }

    def test_eval_split_carries_record_keyset(self):
        splits = self._prov()["splits"]
        assert splits["eval"]["count"] == FROZEN_ELIGIBLE_COUNT
        assert splits["eval"]["keyset_sha256"] == CANONICAL_KEYSET_SHA256

    def test_missing_frozen_dir_is_typed_ingress(self, tmp_path):
        with pytest.raises(schema.TypedIngressError):
            driver.build_provenance_from_frozen(
                tmp_path / "nope", keyset_digest=CANONICAL_KEYSET_SHA256
            )


# ---------------------------------------------------------------------------
# D6 baseline — parameterized (file path OR explicit hashes), never hardcoded
# ---------------------------------------------------------------------------


class TestD6Baseline:
    def test_from_checksums_json_file(self, tmp_path):
        path = tmp_path / "FINAL_CHECKSUMS.json"
        entries = _write_final_checksums_json(path)
        baseline = driver.build_d6_baseline(checksums_path=path)
        assert baseline["main_tex_sha256"] == D6_MAIN_TEX_SHA256
        assert baseline["main_pdf_sha256"] == D6_MAIN_PDF_SHA256
        assert baseline["final_checksums_sha256"] == sha256_bytes(
            path.read_bytes()
        )
        assert baseline["final_checksums_entries"] == entries

    def test_from_checksums_text_file(self, tmp_path):
        path = tmp_path / "FINAL_CHECKSUMS.txt"
        entries = _write_final_checksums_text(path)
        baseline = driver.build_d6_baseline(checksums_path=path)
        assert baseline["final_checksums_entries"] == entries
        assert baseline["final_checksums_sha256"] == sha256_bytes(
            path.read_bytes()
        )

    def test_satisfies_closure_with_qa012(self, tmp_path):
        path = tmp_path / "FINAL_CHECKSUMS.json"
        _write_final_checksums_json(path)
        baseline = driver.build_d6_baseline(checksums_path=path)
        qa = {"status": "VERIFIED_VACUOUS", "inventory_sha256": "b" * 64}
        from reproducibility.colm_aims_2026 import (
            phase4_assemble_d7b as assembler,
        )

        inv = assembler.assemble_closure_inventory(
            d6_baseline=baseline, qa012=qa
        )
        assert closure.evaluate_closure(inv)["satisfied"] is True

    def test_default_without_checksums_is_unsatisfied(self):
        # The FINAL D6 checksums are post-de-anonymization and unknown now:
        # without them, closure MUST fail (never a hardcoded false pass).
        baseline = driver.build_d6_baseline()
        assert baseline["main_tex_sha256"] == D6_MAIN_TEX_SHA256
        assert baseline["main_pdf_sha256"] == D6_MAIN_PDF_SHA256
        assert not schema.is_sha256_hex(
            baseline.get("final_checksums_sha256")
        )
        from reproducibility.colm_aims_2026 import (
            phase4_assemble_d7b as assembler,
        )

        inv = assembler.assemble_closure_inventory(
            d6_baseline=baseline,
            qa012={"status": "VERIFIED_VACUOUS", "inventory_sha256": "b" * 64},
        )
        assert closure.evaluate_closure(inv)["satisfied"] is False

    def test_explicit_hash_overrides(self, tmp_path):
        path = tmp_path / "FINAL_CHECKSUMS.json"
        _write_final_checksums_json(path)
        baseline = driver.build_d6_baseline(
            checksums_path=path,
            main_tex_sha256="d" * 64,
            main_pdf_sha256="e" * 64,
        )
        assert baseline["main_tex_sha256"] == "d" * 64
        assert baseline["main_pdf_sha256"] == "e" * 64


# ---------------------------------------------------------------------------
# QA-012 block — detector manifest OR explicit status/hash
# ---------------------------------------------------------------------------


class TestQa012Block:
    def test_from_zero_hit_corpus(self, tmp_path):
        corpus = _write_qa012_corpus(tmp_path / "qa012")
        block = driver.build_qa012_block(roots=[corpus])
        assert block["status"] == "VERIFIED_VACUOUS"
        assert schema.is_sha256_hex(block["inventory_sha256"])

    def test_explicit_status_and_hash(self):
        block = driver.build_qa012_block(
            status="VERIFIED_VACUOUS", inventory_sha256="b" * 64
        )
        assert block == {
            "status": "VERIFIED_VACUOUS",
            "inventory_sha256": "b" * 64,
        }

    def test_requires_some_input(self):
        with pytest.raises(schema.ConfigSurfaceError):
            driver.build_qa012_block()


# ---------------------------------------------------------------------------
# run_driver — end-to-end build + create-once publish + source verify
# ---------------------------------------------------------------------------


def _run_driver(tmp_path, *, run_id="run-0001", reclaim_crashed_relic=False):
    records_root = _write_records_root(tmp_path / "records")
    out_dir = tmp_path / "out"
    checksums = tmp_path / "FINAL_CHECKSUMS.json"
    _write_final_checksums_json(checksums)
    qa_corpus = _write_qa012_corpus(tmp_path / "qa012")
    return driver.run_driver(
        records_root,
        out_dir,
        FROZEN_DIR,
        source_commit="d" * 40,
        d6_checksums=checksums,
        qa012_roots=[qa_corpus],
        run_id=run_id,
        reclaim_crashed_relic=reclaim_crashed_relic,
    )


class TestRunDriver:
    def test_published_profile_validates(self, tmp_path):
        built = _run_driver(tmp_path)
        assert built.published_tree.is_dir()
        assert (built.published_tree / "profile.json").is_file()
        schema.validate_profile(built.profile)
        for cell_id in CELL_IDS:
            rel = f"records/{cell_id}.jsonl"
            assert rel in built.profile["provenance"]["input_sha256"]
        # The frozen model manifest is genuinely load-bearing in provenance.
        assert (
            built.profile["provenance"]["model"]["repository_namespace"]
            == PRIMARY_SCORER_MODEL
        )

    def test_non_provenance_profile_matches_canonical_oracle(self, tmp_path):
        built = _run_driver(tmp_path)
        oracle = make_profile_v2(source_commit="d" * 40)
        for key in schema.PROFILE_TOP_LEVEL_KEYS:
            if key == "provenance":
                continue
            assert json.dumps(built.profile[key], sort_keys=True) == json.dumps(
                oracle[key], sort_keys=True
            ), f"profile[{key!r}] diverged from the canonical study oracle"

    def test_closure_inventory_satisfied(self, tmp_path):
        built = _run_driver(tmp_path)
        result = closure.evaluate_closure(built.closure_inventory)
        assert result["satisfied"] is True, result["failing_rows"]

    def test_source_verify_passes(self, tmp_path):
        built = _run_driver(tmp_path)
        receipts = tmp_path / "receipts"
        proc = run_cli(
            "--mode",
            "source",
            "--tree",
            str(built.published_tree),
            "--receipts-dir",
            str(receipts),
        )
        assert proc.returncode == EXIT_PASS, proc.stderr
        assert VERDICT_SOURCE_PASS in proc.stdout

    def test_second_publish_same_run_id_fails_closed(self, tmp_path):
        _run_driver(tmp_path)
        with pytest.raises(schema.ColmAimsError):
            _run_driver(tmp_path, run_id="run-0001")


# ---------------------------------------------------------------------------
# CLI — exit codes mirror verify.py / the assembler (0 / 2 / 3 / 4)
# ---------------------------------------------------------------------------


class TestCli:
    def test_exit_code_constants_pinned(self):
        assert driver.EXIT_PASS == 0
        assert driver.EXIT_USAGE_ERROR == 2
        assert driver.EXIT_INGRESS_ERROR == 3
        assert driver.EXIT_INTERNAL_ERROR == 4

    def test_main_end_to_end_source_pass(self, tmp_path):
        records_root = _write_records_root(tmp_path / "records")
        out_dir = tmp_path / "out"
        checksums = tmp_path / "FINAL_CHECKSUMS.json"
        _write_final_checksums_json(checksums)
        qa_corpus = _write_qa012_corpus(tmp_path / "qa012")
        rc = driver.main(
            [
                "--records-root",
                str(records_root),
                "--out-dir",
                str(out_dir),
                "--frozen-dir",
                str(FROZEN_DIR),
                "--source-commit",
                "d" * 40,
                "--d6-checksums",
                str(checksums),
                "--qa012-root",
                str(qa_corpus),
            ]
        )
        assert rc == EXIT_PASS
        published_tree = out_dir / "runs" / "run-0001"
        receipts = tmp_path / "receipts"
        proc = run_cli(
            "--mode",
            "source",
            "--tree",
            str(published_tree),
            "--receipts-dir",
            str(receipts),
        )
        assert proc.returncode == EXIT_PASS, proc.stderr
        assert VERDICT_SOURCE_PASS in proc.stdout

    def test_unknown_flag_is_usage_error(self):
        assert driver.main(["--frobnicate"]) == EXIT_USAGE_ERROR

    def test_missing_records_is_ingress_error(self, tmp_path):
        checksums = tmp_path / "FINAL_CHECKSUMS.json"
        _write_final_checksums_json(checksums)
        qa_corpus = _write_qa012_corpus(tmp_path / "qa012")
        rc = driver.main(
            [
                "--records-root",
                str(tmp_path / "nope"),
                "--out-dir",
                str(tmp_path / "out"),
                "--frozen-dir",
                str(FROZEN_DIR),
                "--source-commit",
                "d" * 40,
                "--d6-checksums",
                str(checksums),
                "--qa012-root",
                str(qa_corpus),
            ]
        )
        assert rc == EXIT_INGRESS_ERROR

    def test_missing_frozen_is_ingress_error(self, tmp_path):
        records_root = _write_records_root(tmp_path / "records")
        checksums = tmp_path / "FINAL_CHECKSUMS.json"
        _write_final_checksums_json(checksums)
        qa_corpus = _write_qa012_corpus(tmp_path / "qa012")
        rc = driver.main(
            [
                "--records-root",
                str(records_root),
                "--out-dir",
                str(tmp_path / "out"),
                "--frozen-dir",
                str(tmp_path / "no-frozen"),
                "--source-commit",
                "d" * 40,
                "--d6-checksums",
                str(checksums),
                "--qa012-root",
                str(qa_corpus),
            ]
        )
        assert rc == EXIT_INGRESS_ERROR

    def test_missing_qa012_input_is_usage_error(self, tmp_path):
        records_root = _write_records_root(tmp_path / "records")
        checksums = tmp_path / "FINAL_CHECKSUMS.json"
        _write_final_checksums_json(checksums)
        rc = driver.main(
            [
                "--records-root",
                str(records_root),
                "--out-dir",
                str(tmp_path / "out"),
                "--frozen-dir",
                str(FROZEN_DIR),
                "--source-commit",
                "d" * 40,
                "--d6-checksums",
                str(checksums),
            ]
        )
        assert rc == EXIT_USAGE_ERROR
