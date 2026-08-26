"""Contract tests for the production D7(b) assembler (phase4_assemble_d7b).

This module exercises the FIRST production code that assembles the D7(b)
evidence package: the pure inference/shaper builders (``compute_inference``,
``assemble_inference_block``, ``assemble_cell``, ``assemble_grid_block``,
``assemble_profile``, ``assemble_closure_inventory``) plus the create-once
orchestrator (``build_evidence_package``).

FP/FN independence: correctness of ``compute_inference`` is asserted against
the PURE d7b_* reference wrappers (``tests/_colm_aims_v2_helpers``), never
against the production ``pairing`` internals it calls. The strongest leg is
the end-to-end ``verify --mode source`` recompute over a package this module
builds.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from reproducibility.colm_aims_2026 import closure, schema

from reproducibility.colm_aims_2026 import phase4_assemble_d7b as assembler

from tests._colm_aims_v2_helpers import (
    CANONICAL_KEYSET_SHA256,
    CANONICAL_MATRIX_SHA256,
    CANONICAL_SEED,
    CELL_IDS,
    D6_MAIN_PDF_SHA256,
    D6_MAIN_TEX_SHA256,
    EXIT_INGRESS_ERROR,
    EXIT_PASS,
    EXIT_USAGE_ERROR,
    FAKE_SHA_A,
    FAKE_SHA_B,
    FAKE_SHA_C,
    N_ITEMS,
    REPO_ROOT,
    VERDICT_SOURCE_PASS,
    canonical_data,
    colm_no_network,  # noqa: F401 - autouse fixture
    d7b_holm,
    d7b_interval,
    d7b_p_value,
    d7b_resample_matrix,
    d7b_seed,
    make_arms,
    make_estimand,
    make_grid_block,
    make_llm_involvement,
    make_profile_v2,
    make_provenance,
    run_cli,
    sha256_bytes,
)


# ---------------------------------------------------------------------------
# Fixtures — synthetic complete_by_cell / records root from the pure oracle
# ---------------------------------------------------------------------------


def _complete_by_cell() -> dict[str, dict[str, dict]]:
    """Parse the canonical (pure-oracle) record bytes into the in-memory
    ``{cell_id: {item_key: record}}`` map the verifier builds at run time."""
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


def _write_records_root(root) -> dict[str, str]:
    """Write canonical per-cell record bytes to ``root/<cell>.jsonl`` and
    return the on-disk ``records/<cell>.jsonl`` -> sha256 hashes."""
    data = canonical_data()
    root.mkdir(parents=True, exist_ok=True)
    hashes: dict[str, str] = {}
    for cell_id in CELL_IDS:
        blob = data["cells"][cell_id].records_bytes
        (root / f"{cell_id}.jsonl").write_bytes(blob)
        hashes[f"records/{cell_id}.jsonl"] = sha256_bytes(blob)
    return hashes


def _signed_shift(record: dict) -> int:
    """Independent (test-side) sentinel-coded signed shift MC - REF."""
    horizon = record["trajectory_horizon"]
    mc = record["mc_stop_step"]
    ref = record["ref_stop_step"]
    s_mc = horizon if mc is None else mc
    s_ref = horizon if ref is None else ref
    return s_mc - s_ref


def _d6_baseline() -> dict:
    return {
        "main_tex_sha256": D6_MAIN_TEX_SHA256,
        "main_pdf_sha256": D6_MAIN_PDF_SHA256,
        "final_checksums_sha256": FAKE_SHA_A,
        "final_checksums_entries": {
            "main.tex": D6_MAIN_TEX_SHA256,
            "main.pdf": D6_MAIN_PDF_SHA256,
            "figures/fig1.pdf": FAKE_SHA_B,
            "references.bib": FAKE_SHA_C,
        },
    }


def _qa012() -> dict:
    return {"status": "VERIFIED_VACUOUS", "inventory_sha256": FAKE_SHA_B}


def _estimands() -> dict[str, dict]:
    return {cell_id: make_estimand(cell_id) for cell_id in CELL_IDS}


# ---------------------------------------------------------------------------
# compute_inference — reproduces the canonical pure-oracle literals
# ---------------------------------------------------------------------------


class TestComputeInference:
    def test_reproduces_canonical_golden_literals(self):
        result = assembler.compute_inference(_complete_by_cell())
        assert result.keyset_digest == CANONICAL_KEYSET_SHA256
        assert result.seed == CANONICAL_SEED
        assert result.matrix_digest["sha256"] == CANONICAL_MATRIX_SHA256

        data = canonical_data()
        for cell_id in CELL_IDS:
            cd = data["cells"][cell_id]
            cell = result.per_cell[cell_id]
            assert cell.ci[0] == cd.ci[0]
            assert cell.ci[1] == cd.ci[1]
            assert cell.raw_p_value == cd.raw_p_value
        assert (
            result.holm["rejected_cell_ids"]
            == data["holm"]["rejected_cell_ids"]
        )

    def test_matches_pure_reference_exactly(self):
        # FP/FN independence: recompute every D7(b) leg with the PURE wrappers
        # over the SAME records and assert bit-exact equality with production.
        complete = _complete_by_cell()
        result = assembler.compute_inference(complete)

        seed = d7b_seed(result.keyset_digest)
        matrix = d7b_resample_matrix(seed)
        assert result.seed == seed

        raw_p: dict[str, float] = {}
        for cell_id in CELL_IDS:
            records = complete[cell_id]
            ordered = sorted(records)
            d = np.array(
                [_signed_shift(records[key]) for key in ordered],
                dtype=np.float64,
            )
            lo, hi = d7b_interval(d, matrix)
            p = d7b_p_value(d, matrix)
            raw_p[cell_id] = p
            cell = result.per_cell[cell_id]
            assert cell.ci == (lo, hi)
            assert cell.raw_p_value == p
        holm = d7b_holm(raw_p)
        assert result.holm["rejected_cell_ids"] == holm["rejected_cell_ids"]
        assert result.holm["ordered_family"] == holm["ordered_family"]

    def test_rejects_wrong_cell_count(self):
        complete = _complete_by_cell()
        complete.pop(CELL_IDS[0])
        with pytest.raises(schema.ColmAimsError):
            assembler.compute_inference(complete)

    def test_rejects_mismatched_keyset_across_cells(self):
        complete = _complete_by_cell()
        victim = CELL_IDS[0]
        records = dict(complete[victim])
        stolen_key = next(iter(records))
        template = dict(records[stolen_key])
        records.pop(stolen_key)
        template["item_key"] = "itm-ffffffffffffffff"
        records["itm-ffffffffffffffff"] = template
        complete[victim] = records
        with pytest.raises(schema.ColmAimsError):
            assembler.compute_inference(complete)


# ---------------------------------------------------------------------------
# load_complete_by_cell — reads records/<cell>.jsonl from disk
# ---------------------------------------------------------------------------


def test_load_complete_by_cell_roundtrips(tmp_path):
    root = tmp_path / "records"
    _write_records_root(root)
    complete = assembler.load_complete_by_cell(root)
    assert sorted(complete) == sorted(CELL_IDS)
    for cell_id in CELL_IDS:
        assert len(complete[cell_id]) == N_ITEMS
    result = assembler.compute_inference(complete)
    assert result.keyset_digest == CANONICAL_KEYSET_SHA256
    assert result.seed == CANONICAL_SEED


def test_load_complete_by_cell_rejects_duplicate_physical_item_key(tmp_path):
    root = tmp_path / "records"
    _write_records_root(root)
    victim = root / f"{CELL_IDS[0]}.jsonl"
    lines = victim.read_bytes().splitlines(keepends=True)
    victim.write_bytes(b"".join(lines + [lines[0]]))

    with pytest.raises(schema.TypedIngressError, match="duplicate item_key"):
        assembler.load_complete_by_cell(root)


# ---------------------------------------------------------------------------
# assemble_profile — validates + byte-identical to the oracle profile
# ---------------------------------------------------------------------------


class TestAssembleProfile:
    def _assemble(self, *, source_commit, input_sha256):
        result = assembler.compute_inference(_complete_by_cell())
        return assembler.assemble_profile(
            result,
            arms=make_arms(),
            provenance=make_provenance(
                source_commit=source_commit, input_sha256=input_sha256
            ),
            grid=make_grid_block(),
            llm_involvement=make_llm_involvement(),
            estimands=_estimands(),
        )

    def test_validates_and_carries_d7b_provenance(self):
        profile = self._assemble(source_commit="a" * 40, input_sha256={})
        schema.validate_profile(profile)
        assert (
            profile["inference"]["analysis_provenance"]
            == schema.ANALYSIS_PROVENANCE_D7B
        )
        assert set(profile) == set(schema.PROFILE_TOP_LEVEL_KEYS)
        assert len(profile["cells"]) == 10

    def test_bytes_match_canonical_oracle_profile(self):
        # Production compute + assemble must be byte-identical to the pure
        # oracle profile (make_profile_v2), proving compute_inference matches
        # canonical_data at the full-profile level.
        commit = "b" * 40
        input_sha256 = {f"records/{c}.jsonl": FAKE_SHA_A for c in CELL_IDS}
        prod = self._assemble(source_commit=commit, input_sha256=input_sha256)
        oracle = make_profile_v2(
            source_commit=commit, input_sha256=input_sha256
        )
        assert schema.encode_profile(prod) == schema.encode_profile(oracle)


# ---------------------------------------------------------------------------
# assemble_closure_inventory — satisfied / unsatisfied via evaluate_closure
# ---------------------------------------------------------------------------


class TestAssembleClosureInventory:
    def test_satisfied(self):
        inv = assembler.assemble_closure_inventory(
            d6_baseline=_d6_baseline(), qa012=_qa012()
        )
        result = closure.evaluate_closure(inv)
        assert result["satisfied"] is False
        result = closure.evaluate_closure(
            inv,
            expected_final_checksums_entries_sha256=inv["d6_baseline"][
                "final_checksums_entries_sha256"
            ],
        )
        assert result["satisfied"] is True, result["failing_rows"]

    def test_unsatisfied_when_holm_row_broken(self):
        inv = assembler.assemble_closure_inventory(
            d6_baseline=_d6_baseline(),
            qa012=_qa012(),
            holm_satisfied_by=None,
        )
        assert closure.evaluate_closure(inv)["satisfied"] is False

    def test_unsatisfied_when_qa012_unverified(self):
        inv = assembler.assemble_closure_inventory(
            d6_baseline=_d6_baseline(),
            qa012={"status": "UNVERIFIED", "inventory_sha256": FAKE_SHA_B},
        )
        assert closure.evaluate_closure(inv)["satisfied"] is False


class TestQa012StatusBlock:
    def test_zero_hit_maps_to_vacuous(self):
        block = assembler.qa012_status_block(
            {"result": "zero_hit", "inventory_sha256": FAKE_SHA_B}
        )
        assert block == {
            "status": "VERIFIED_VACUOUS",
            "inventory_sha256": FAKE_SHA_B,
        }

    def test_unbound_hits_remain_blocking(self):
        block = assembler.qa012_status_block(
            {"result": "hits", "inventory_sha256": FAKE_SHA_A, "files": []}
        )
        assert block["status"] == "HITS_PRESENT"

    def test_only_exact_bound_hits_map_to_verified_fixtures(self):
        fixture_root = REPO_ROOT / "tests" / "fixtures" / "qa012_item10"
        bindings = json.loads((fixture_root / "bindings.json").read_text())
        files = []
        for basename, binding in bindings["files"].items():
            files.append(
                {
                    "path": f"item10/{basename}",
                    "size": binding["full_file_size"],
                    "sha256": binding["full_file_sha256"],
                    "hits": ["/0/format"],
                    "first_two_records_sha256": binding["excerpt_sha256"],
                }
            )
        block = assembler.qa012_status_block(
            {
                "result": "hits",
                "inventory_sha256": FAKE_SHA_A,
                "files": files,
            }
        )
        assert block["status"] == "VERIFIED_WITH_FIXTURES"

    def test_unknown_result_fails_closed(self):
        with pytest.raises(schema.ColmAimsError):
            assembler.qa012_status_block(
                {"result": "maybe", "inventory_sha256": FAKE_SHA_A}
            )


# ---------------------------------------------------------------------------
# build_evidence_package — create-once publish + end-to-end source verify
# ---------------------------------------------------------------------------


def _build(tmp_path, *, run_id="run-0001", reclaim_crashed_relic=False):
    records_root = tmp_path / "records"
    _write_records_root(records_root)
    out_dir = tmp_path / "out"
    return assembler.build_evidence_package(
        records_root,
        out_dir,
        source_commit="d" * 40,
        arms=make_arms(),
        provenance=make_provenance(source_commit="d" * 40),
        grid=make_grid_block(),
        llm_involvement=make_llm_involvement(),
        estimands=_estimands(),
        d6_baseline=_d6_baseline(),
        qa012=_qa012(),
        run_id=run_id,
        reclaim_crashed_relic=reclaim_crashed_relic,
    )


class TestBuildEvidencePackage:
    def test_caller_snapshot_tampering_is_rejected_before_publish(
        self, tmp_path
    ):
        records_root = tmp_path / "records"
        _write_records_root(records_root)
        snapshot = assembler.load_record_snapshot(records_root)
        victim_cell = CELL_IDS[0]
        victim_record = next(
            iter(snapshot.complete_by_cell[victim_cell].values())
        )
        victim_record["trajectory_horizon"] += 1

        with pytest.raises(
            schema.TypedIngressError, match="do not match its parsed records"
        ):
            assembler.build_evidence_package(
                records_root,
                tmp_path / "out",
                source_commit="d" * 40,
                arms=make_arms(),
                provenance=make_provenance(source_commit="d" * 40),
                grid=make_grid_block(),
                llm_involvement=make_llm_involvement(),
                estimands=_estimands(),
                d6_baseline=_d6_baseline(),
                qa012=_qa012(),
                record_snapshot=snapshot,
            )

        assert not (tmp_path / "out" / "runs" / "run-0001").exists()

    def test_published_profile_validates(self, tmp_path):
        built = _build(tmp_path)
        assert built.published_tree.is_dir()
        assert (built.published_tree / "profile.json").is_file()
        assert built.published_tree.name == "tree"
        assert built.closure_inventory_path == (
            built.published_tree.parent / "closure" / "closure_inventory.json"
        )
        assert built.closure_inventory_path.is_file()
        assert not (tmp_path / "out" / "closure_inventory.json").exists()
        schema.validate_profile(built.profile)
        for cell_id in CELL_IDS:
            rel = f"records/{cell_id}.jsonl"
            assert rel in built.profile["provenance"]["input_sha256"]

    def test_closure_inventory_satisfied(self, tmp_path):
        built = _build(tmp_path)
        assert not closure.evaluate_closure(built.closure_inventory)["satisfied"]
        assert closure.evaluate_closure(
            built.closure_inventory,
            expected_final_checksums_entries_sha256=built.closure_inventory[
                "d6_baseline"
            ]["final_checksums_entries_sha256"],
        )["satisfied"]

    def test_publishes_the_same_record_bytes_that_were_validated(
        self, tmp_path, monkeypatch
    ):
        records_root = tmp_path / "records"
        hashes = _write_records_root(records_root)
        victim_rel = f"records/{CELL_IDS[0]}.jsonl"
        victim = records_root / f"{CELL_IDS[0]}.jsonl"
        original = victim.read_bytes()
        real_reader = schema.read_regular_file_bytes
        mutated = False

        def mutate_after_capture(path, **kwargs):
            nonlocal mutated
            data = real_reader(path, **kwargs)
            if path == victim and not mutated:
                victim.write_bytes(b'{"tampered":true}\n')
                mutated = True
            return data

        monkeypatch.setattr(schema, "read_regular_file_bytes", mutate_after_capture)
        built = assembler.build_evidence_package(
            records_root,
            tmp_path / "out",
            source_commit="d" * 40,
            arms=make_arms(),
            provenance=make_provenance(source_commit="d" * 40),
            grid=make_grid_block(),
            llm_involvement=make_llm_involvement(),
            estimands=_estimands(),
            d6_baseline=_d6_baseline(),
            qa012=_qa012(),
        )

        assert mutated
        assert (built.published_tree / victim_rel).read_bytes() == original
        assert built.input_sha256[victim_rel] == hashes[victim_rel]

    def test_source_verify_passes(self, tmp_path):
        built = _build(tmp_path)
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
        _build(tmp_path)
        with pytest.raises(schema.ColmAimsError):
            _build(tmp_path, run_id="run-0001")

    def test_reclaim_recovers_empty_relic(self, tmp_path):
        _build(tmp_path)
        runs_root = tmp_path / "out" / "runs"
        (runs_root / "run-0002").mkdir()
        with pytest.raises(schema.ColmAimsError):
            _build(tmp_path, run_id="run-0002")
        rebuilt = _build(
            tmp_path, run_id="run-0002", reclaim_crashed_relic=True
        )
        assert (rebuilt.published_tree / "profile.json").is_file()
        assert rebuilt.closure_inventory_path.is_file()


# ---------------------------------------------------------------------------
# CLI contract — exit codes mirror verify.py (0 / 2 / 3 / 4)
# ---------------------------------------------------------------------------


class TestCli:
    def test_exit_code_constants_pinned(self):
        assert assembler.EXIT_PASS == 0
        assert assembler.EXIT_USAGE_ERROR == 2
        assert assembler.EXIT_INGRESS_ERROR == 3
        assert assembler.EXIT_INTERNAL_ERROR == 4

    def test_main_computes_inference_summary(self, tmp_path):
        records_root = tmp_path / "records"
        _write_records_root(records_root)
        out_dir = tmp_path / "out"
        rc = assembler.main(
            [
                "--records-root",
                str(records_root),
                "--out-dir",
                str(out_dir),
                "--source-commit",
                "d" * 40,
            ]
        )
        assert rc == EXIT_PASS
        summary = json.loads(
            (out_dir / "inference_summary.json").read_text("utf-8")
        )
        assert summary["seed"] == CANONICAL_SEED
        assert summary["pairing_population_keyset_sha256"] == (
            CANONICAL_KEYSET_SHA256
        )

    def test_unknown_flag_is_usage_error(self):
        assert assembler.main(["--frobnicate"]) == EXIT_USAGE_ERROR

    def test_missing_records_is_ingress_error(self, tmp_path):
        rc = assembler.main(
            [
                "--records-root",
                str(tmp_path / "nope"),
                "--out-dir",
                str(tmp_path / "out"),
                "--source-commit",
                "d" * 40,
            ]
        )
        assert rc == EXIT_INGRESS_ERROR
