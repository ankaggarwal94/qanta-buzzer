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
from pathlib import Path

import pytest

from reproducibility.colm_aims_2026 import closure, qa012, schema

from reproducibility.colm_aims_2026 import phase4_driver_d7b as driver
from reproducibility.colm_aims_2026 import phase4_assemble_d7b as assembler

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
    make_closure_profile_bytes,
    make_estimand,
    make_grid_block,
    make_llm_involvement,
    make_profile_v2,
    run_cli,
    sha256_bytes,
)

# The REAL in-repo frozen inputs (never regenerated here).
FROZEN_DIR = NAMESPACE_DIR / "frozen"
TEST_SOURCE_COMMIT = "d" * 40
TEST_SOURCE_TREE = "e" * 40
PLACEHOLDER_ACTIVATION_DIGEST = "9" * 64


@pytest.fixture(autouse=True)
def _bind_live_driver_checkout(monkeypatch):
    """Synthetic transaction fixtures bind a synthetic clean Git identity."""
    monkeypatch.setattr(
        driver,
        "_live_repo_identity",
        lambda: {
            "commit": TEST_SOURCE_COMMIT,
            "tree_sha256": TEST_SOURCE_TREE,
            "dirty": False,
        },
    )

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
FROZEN_KEYSET_SHA256 = (
    "d0ebac8f300f936f10298e2186532dfc1efd0fee6f400c1a1d8696cf86dd00f1"
)
FROZEN_HORIZON_SHA256 = (
    "b0514b6cbe6dfffad0ce225869d20b306377d5baff1e1aca4b9cc9904a95486d"
)
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
    # Reuse the independently exercised full certificate-component builder;
    # this transaction fixture must carry a genuinely reproducible ready
    # certificate rather than a hand-written ready:true wrapper.
    from tests.test_colm_aims_v2_phase4_pre import _good_components

    components = _good_components()
    components["repo"]["commit"] = TEST_SOURCE_COMMIT
    components["repo"]["tree_sha256"] = TEST_SOURCE_TREE
    for suite_receipt in components["suite_receipts"].values():
        suite_receipt["commit"] = TEST_SOURCE_COMMIT
        suite_receipt["tree_sha256"] = TEST_SOURCE_TREE
    certificate = driver.phase4.assemble_certificate(components)
    assert certificate["ready"] is True, certificate["failing_checks"]
    certificate_path = root.parent / "certificate.json"
    certificate_bytes = schema.encode_json(certificate)
    certificate_path.write_bytes(certificate_bytes)
    activation_digest = sha256_bytes(certificate_bytes)
    ledger = {
        "activation_digest": activation_digest,
        "certificate_path": str(certificate_path.absolute()),
        "certificate_commit": TEST_SOURCE_COMMIT,
        "certificate_tree": TEST_SOURCE_TREE,
        "argv": ["python", "producer.py"],
        "consumed_at": "2026-08-26T00:00:00+00:00",
    }
    ledger_path = root.parent / "launch-ledger.json"
    ledger_path.write_bytes(schema.encode_json(ledger))
    export_basename = "stopdff_fair_qa_regenerated.json"
    export_path = root.parent / export_basename
    export_path.write_bytes(b'{"verdict":"PASS"}\n')
    receipt = {
        "schema_version": schema.SCHEMA_VERSION,
        "receipt_type": "phase4_launch",
        "activation_digest": activation_digest,
        "ledger_sha256": sha256_bytes(ledger_path.read_bytes()),
        "producer_exit_code": 0,
        "comparator_verdict": "PASS",
        "comparator_checked": 194,
        "export_basename": export_basename,
        "export_sha256": sha256_bytes(export_path.read_bytes()),
        "records_sha256": {
            cell_id: sha256_bytes((root / f"{cell_id}.jsonl").read_bytes())
            for cell_id in CELL_IDS
        },
    }
    (root.parent / driver.LAUNCH_RECEIPT_NAME).write_bytes(
        schema.encode_json(receipt)
    )
    return root


def _launch_receipt(records_root: Path) -> Path:
    return Path(records_root).parent / driver.LAUNCH_RECEIPT_NAME


def _launch_ledger(records_root: Path) -> Path:
    return Path(records_root).parent / "launch-ledger.json"


def _launch_certificate(records_root: Path) -> Path:
    return Path(records_root).parent / "certificate.json"


def _activation_digest(records_root: Path) -> str:
    return sha256_bytes(_launch_certificate(records_root).read_bytes())


def _resign_transaction_chain(
    records_root: Path,
    certificate: dict,
    *,
    mutate_ledger=None,
) -> str:
    """Rewrite a synthetic certificate and all enclosing transaction hashes."""
    certificate_bytes = schema.encode_json(certificate)
    _launch_certificate(records_root).write_bytes(certificate_bytes)
    activation_digest = sha256_bytes(certificate_bytes)
    ledger_path = _launch_ledger(records_root)
    ledger = json.loads(ledger_path.read_text("utf-8"))
    ledger["activation_digest"] = activation_digest
    if mutate_ledger is not None:
        mutate_ledger(ledger)
    ledger_path.write_bytes(schema.encode_json(ledger))
    receipt_path = _launch_receipt(records_root)
    receipt = json.loads(receipt_path.read_text("utf-8"))
    receipt["activation_digest"] = activation_digest
    receipt["ledger_sha256"] = sha256_bytes(ledger_path.read_bytes())
    receipt_path.write_bytes(schema.encode_json(receipt))
    return activation_digest


def _final_checksums_entries() -> dict[str, str]:
    return dict(closure.D6_FINAL_CHECKSUM_ENTRIES)


def _write_final_checksums_json(path) -> dict[str, str]:
    entries = _final_checksums_entries()
    path.write_bytes(closure.D6_FINAL_CHECKSUMS_BYTES)
    return entries


def _write_final_checksums_text(path) -> dict[str, str]:
    """sha256sum output form: ``<64-hex>  <relpath>`` lines."""
    entries = _final_checksums_entries()
    path.write_bytes(closure.D6_FINAL_CHECKSUMS_BYTES)
    return entries


def _bind_synthetic_records_to_frozen(monkeypatch) -> None:
    """Keep synthetic driver tests explicit about their noncanonical records."""
    eligibility = driver.phase4.load_pairing_eligibility(
        FROZEN_DIR / "pairing_eligibility_v2.json"
    )
    synthetic = dict(eligibility)
    synthetic["pairing_population_keyset_sha256"] = CANONICAL_KEYSET_SHA256
    synthetic["horizon_map_sha256"] = canonical_horizon_identity()
    monkeypatch.setattr(
        driver.phase4,
        "load_pairing_eligibility",
        lambda _path: synthetic,
    )
    # The shared canonical test fixture uses the generic opaque-hash scheme,
    # not the production Phase-4 dataset-QID scheme.
    monkeypatch.setattr(
        schema, "PHASE4_ITEM_KEY_DERIVATION", schema.ITEM_KEY_DERIVATION
    )


def _write_qa012_corpus(root):
    """A tiny zero-hit QA-012 corpus (no ``format:"QA"`` occurrences)."""
    roots = {}
    for prong in qa012.REQUIRED_SCOPE_PRONGS:
        prong_root = root / prong
        prong_root.mkdir(parents=True, exist_ok=True)
        (prong_root / "note.json").write_bytes(
            (json.dumps({"kind": "note", "value": 1}) + "\n").encode(
                "utf-8"
            )
        )
        roots[prong] = prong_root
    return roots


def _qa012_cli_args():
    return ["--qa012-authority", str(qa012.CANONICAL_AUTHORITY_PATH)]


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
            FROZEN_DIR,
            keyset_digest=FROZEN_KEYSET_SHA256,
            horizon_identity=FROZEN_HORIZON_SHA256,
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
        assert splits["eval"]["keyset_sha256"] == FROZEN_KEYSET_SHA256

    def test_missing_frozen_dir_is_typed_ingress(self, tmp_path):
        with pytest.raises(schema.TypedIngressError):
            driver.build_provenance_from_frozen(
                tmp_path / "nope",
                keyset_digest=FROZEN_KEYSET_SHA256,
                horizon_identity=FROZEN_HORIZON_SHA256,
            )

    def test_rejects_record_keyset_that_differs_from_frozen_pin(self):
        with pytest.raises(schema.TypedIngressError, match="keyset digest"):
            driver.build_provenance_from_frozen(
                FROZEN_DIR,
                keyset_digest=CANONICAL_KEYSET_SHA256,
                horizon_identity=FROZEN_HORIZON_SHA256,
            )

    def test_rejects_record_horizon_that_differs_from_frozen_pin(self):
        with pytest.raises(schema.TypedIngressError, match="horizon identity"):
            driver.build_provenance_from_frozen(
                FROZEN_DIR,
                keyset_digest=FROZEN_KEYSET_SHA256,
                horizon_identity=canonical_horizon_identity(),
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
        qa_roots = _write_qa012_corpus(tmp_path / "qa012")
        qa = assembler.qa012_authority_status_block(
            qa012.CANONICAL_AUTHORITY_PATH
        )
        profile_bytes = make_closure_profile_bytes()
        inv = assembler.assemble_closure_inventory(
            d6_baseline=baseline,
            qa012=qa,
            profile_sha256=sha256_bytes(profile_bytes),
            analysis_provenance=schema.ANALYSIS_PROVENANCE_D7B,
        )
        assert closure.evaluate_closure(inv, profile_bytes=profile_bytes)[
            "satisfied"
        ] is True

    def test_default_without_checksums_is_unsatisfied(self, tmp_path):
        # The FINAL D6 checksums are post-de-anonymization and unknown now:
        # without them, closure MUST fail (never a hardcoded false pass).
        baseline = driver.build_d6_baseline()
        assert baseline["main_tex_sha256"] == D6_MAIN_TEX_SHA256
        assert baseline["main_pdf_sha256"] == D6_MAIN_PDF_SHA256
        assert not schema.is_sha256_hex(
            baseline.get("final_checksums_sha256")
        )
        baseline.update(
            {
                "final_checksums_sha256": "f" * 64,
                "final_checksums_entries": {
                    "main.tex": D6_MAIN_TEX_SHA256
                },
                "final_checksums_entries_sha256": closure.checksum_entries_sha256(
                    {"main.tex": D6_MAIN_TEX_SHA256}
                ),
            }
        )
        profile_bytes = make_closure_profile_bytes()
        qa_roots = _write_qa012_corpus(tmp_path / "qa012")
        inv = assembler.assemble_closure_inventory(
            d6_baseline=baseline,
            qa012=assembler.qa012_status_block(
                qa012.build_inventory_manifest(qa_roots)
            ),
            profile_sha256=sha256_bytes(profile_bytes),
            analysis_provenance=schema.ANALYSIS_PROVENANCE_D7B,
        )
        out = closure.evaluate_closure(inv, profile_bytes=profile_bytes)
        assert out["satisfied"] is False
        assert any("FINAL_CHECKSUMS" in row for row in out["failing_rows"])

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
        manifest = driver.build_qa012_block(roots=corpus)
        assert manifest["result"] == "zero_hit"
        qa012.validate_inventory_manifest(manifest)
        assert schema.is_sha256_hex(manifest["inventory_sha256"])

    def test_explicit_status_and_hash(self):
        block = driver.build_qa012_block(
            status="HITS_PRESENT", inventory_sha256="b" * 64
        )
        assert block == {
            "status": "HITS_PRESENT",
            "inventory_sha256": "b" * 64,
        }

    def test_exact_authority_is_the_only_satisfying_path(self):
        block = driver.build_qa012_block(
            authority_path=qa012.CANONICAL_AUTHORITY_PATH
        )
        assert block["status"] == "VERIFIED_WITH_FIXTURES"
        assert block["authority_sha256"] == qa012.CANONICAL_AUTHORITY_SHA256
        assert all("C:/" not in value for value in block.values())

    @pytest.mark.parametrize(
        "status", ["VERIFIED_VACUOUS", "VERIFIED_WITH_FIXTURES"]
    )
    def test_explicit_satisfied_status_cannot_bypass_scan(self, status):
        with pytest.raises(schema.ConfigSurfaceError, match="pinned rev3"):
            driver.build_qa012_block(
                status=status, inventory_sha256="b" * 64
            )

    def test_requires_some_input(self):
        with pytest.raises(schema.ConfigSurfaceError):
            driver.build_qa012_block()


# ---------------------------------------------------------------------------
# run_driver — end-to-end build + create-once publish + source verify
# ---------------------------------------------------------------------------


def _run_driver(
    tmp_path, monkeypatch, *, run_id="run-0001", reclaim_crashed_relic=False
):
    _bind_synthetic_records_to_frozen(monkeypatch)
    records_root = _write_records_root(tmp_path / "records")
    out_dir = tmp_path / "out"
    checksums = tmp_path / "FINAL_CHECKSUMS.json"
    _write_final_checksums_json(checksums)
    return driver.run_driver(
        records_root,
        out_dir,
        FROZEN_DIR,
        source_commit=TEST_SOURCE_COMMIT,
        launch_receipt=_launch_receipt(records_root),
        launch_ledger=_launch_ledger(records_root),
        activation_digest=_activation_digest(records_root),
        d6_checksums=checksums,
        qa012_authority=qa012.CANONICAL_AUTHORITY_PATH,
        run_id=run_id,
        reclaim_crashed_relic=reclaim_crashed_relic,
    )


class TestRunDriver:
    @pytest.mark.parametrize("artifact", ["ledger", "export"])
    def test_launch_receipt_authenticates_transaction_artifacts(
        self, tmp_path, artifact
    ):
        records_root = _write_records_root(tmp_path / "records")
        target = (
            _launch_ledger(records_root)
            if artifact == "ledger"
            else records_root.parent / "stopdff_fair_qa_regenerated.json"
        )
        target.write_bytes(b"tampered\n")
        with pytest.raises(schema.TypedIngressError, match="authenticate"):
            driver.validate_launch_receipt(
                records_root,
                _launch_receipt(records_root),
                _launch_ledger(records_root),
                _activation_digest(records_root),
                TEST_SOURCE_COMMIT,
            )

    def test_missing_referenced_certificate_refuses(self, tmp_path):
        records_root = _write_records_root(tmp_path / "records")
        activation_digest = _activation_digest(records_root)
        _launch_certificate(records_root).unlink()
        with pytest.raises(schema.ColmAimsError, match="missing|unreadable"):
            driver.validate_launch_receipt(
                records_root,
                _launch_receipt(records_root),
                _launch_ledger(records_root),
                activation_digest,
                TEST_SOURCE_COMMIT,
            )

    def test_fabricated_self_consistent_ready_certificate_refuses(
        self, tmp_path
    ):
        records_root = _write_records_root(tmp_path / "records")
        fabricated_components = {
            key: {} for key in driver.phase4.CERT_COMPONENT_KEYS
        }
        fabricated_components["repo"] = {
            "commit": TEST_SOURCE_COMMIT,
            "tree_sha256": TEST_SOURCE_TREE,
        }
        fabricated = {
            "schema_version": driver.phase4.CERT_SCHEMA_VERSION,
            "ready": True,
            "failing_checks": [],
            "components": fabricated_components,
        }
        activation_digest = _resign_transaction_chain(
            records_root, fabricated
        )
        with pytest.raises(schema.TypedIngressError, match="fabricated"):
            driver.validate_launch_receipt(
                records_root,
                _launch_receipt(records_root),
                _launch_ledger(records_root),
                activation_digest,
                TEST_SOURCE_COMMIT,
            )

    @pytest.mark.parametrize(
        ("identity_field", "message"),
        [("commit", "commit disagrees"), ("tree_sha256", "tree disagrees")],
    )
    def test_certificate_identity_must_match_authenticated_ledger(
        self, tmp_path, identity_field, message
    ):
        records_root = _write_records_root(tmp_path / "records")
        certificate = json.loads(
            _launch_certificate(records_root).read_text("utf-8")
        )
        certificate["components"]["repo"][identity_field] = "f" * 40
        activation_digest = _resign_transaction_chain(
            records_root, certificate
        )
        with pytest.raises(schema.TypedIngressError, match=message):
            driver.validate_launch_receipt(
                records_root,
                _launch_receipt(records_root),
                _launch_ledger(records_root),
                activation_digest,
                TEST_SOURCE_COMMIT,
            )

    def test_ledger_commit_must_match_driver_source_commit(self, tmp_path):
        records_root = _write_records_root(tmp_path / "records")
        with pytest.raises(schema.TypedIngressError, match="source_commit"):
            driver.validate_launch_receipt(
                records_root,
                _launch_receipt(records_root),
                _launch_ledger(records_root),
                _activation_digest(records_root),
                "f" * 40,
            )

    def test_certificate_bytes_must_match_activation_digest(self, tmp_path):
        records_root = _write_records_root(tmp_path / "records")
        activation_digest = _activation_digest(records_root)
        _launch_certificate(records_root).write_bytes(b'{}\n')
        with pytest.raises(schema.TypedIngressError, match="activation digest"):
            driver.validate_launch_receipt(
                records_root,
                _launch_receipt(records_root),
                _launch_ledger(records_root),
                activation_digest,
                TEST_SOURCE_COMMIT,
            )

    @pytest.mark.parametrize("drift", ["commit", "tree_sha256", "dirty"])
    def test_live_checkout_must_match_authenticated_certificate(
        self, tmp_path, monkeypatch, drift
    ):
        records_root = _write_records_root(tmp_path / "records")
        identity = {
            "commit": TEST_SOURCE_COMMIT,
            "tree_sha256": TEST_SOURCE_TREE,
            "dirty": False,
        }
        identity[drift] = True if drift == "dirty" else "f" * 40
        monkeypatch.setattr(driver, "_live_repo_identity", lambda: identity)
        with pytest.raises(schema.TypedIngressError, match="live publication"):
            driver.validate_launch_receipt(
                records_root,
                _launch_receipt(records_root),
                _launch_ledger(records_root),
                _activation_digest(records_root),
                TEST_SOURCE_COMMIT,
            )

    def test_each_record_file_is_read_once(self, tmp_path, monkeypatch):
        _bind_synthetic_records_to_frozen(monkeypatch)
        records_root = _write_records_root(tmp_path / "records").absolute()
        out_dir = tmp_path / "out"
        checksums = tmp_path / "FINAL_CHECKSUMS.json"
        _write_final_checksums_json(checksums)

        original_read = driver.assembler.schema.read_regular_file_bytes
        record_reads: list[str] = []

        def _counting_read(path, *, tree_root=None, max_bytes=schema.MAX_ARTIFACT_BYTES):
            if tree_root is not None and Path(tree_root).absolute() == records_root:
                record_reads.append(Path(path).name)
            return original_read(path, tree_root=tree_root, max_bytes=max_bytes)

        monkeypatch.setattr(
            driver.assembler.schema,
            "read_regular_file_bytes",
            _counting_read,
        )
        driver.run_driver(
            records_root,
            out_dir,
            FROZEN_DIR,
            source_commit=TEST_SOURCE_COMMIT,
            launch_receipt=_launch_receipt(records_root),
            launch_ledger=_launch_ledger(records_root),
            activation_digest=_activation_digest(records_root),
            d6_checksums=checksums,
            qa012_authority=qa012.CANONICAL_AUTHORITY_PATH,
        )

        assert sorted(record_reads) == sorted(
            f"{cell_id}.jsonl" for cell_id in CELL_IDS
        )

    def test_published_profile_validates(self, tmp_path, monkeypatch):
        built = _run_driver(tmp_path, monkeypatch)
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

    def test_non_provenance_profile_matches_canonical_oracle(
        self, tmp_path, monkeypatch
    ):
        built = _run_driver(tmp_path, monkeypatch)
        oracle = make_profile_v2(source_commit=TEST_SOURCE_COMMIT)
        for key in schema.PROFILE_TOP_LEVEL_KEYS:
            if key == "provenance":
                continue
            assert json.dumps(built.profile[key], sort_keys=True) == json.dumps(
                oracle[key], sort_keys=True
            ), f"profile[{key!r}] diverged from the canonical study oracle"

    def test_closure_inventory_satisfied(self, tmp_path, monkeypatch):
        built = _run_driver(tmp_path, monkeypatch)
        result = closure.evaluate_closure(
            built.closure_inventory,
            profile_bytes=schema.encode_profile(built.profile),
        )
        assert result["satisfied"] is True, result["failing_rows"]

    def test_source_verify_passes(self, tmp_path, monkeypatch):
        built = _run_driver(tmp_path, monkeypatch)
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

    def test_second_publish_same_run_id_fails_closed(self, tmp_path, monkeypatch):
        _run_driver(tmp_path, monkeypatch)
        with pytest.raises(schema.ColmAimsError):
            _run_driver(tmp_path, monkeypatch, run_id="run-0001")

    def test_invalid_closure_does_not_consume_run_slot(
        self, tmp_path, monkeypatch
    ):
        # No D6 manifest means the closure cannot satisfy its final checksum row.
        # Bind only the synthetic record identities so execution reaches closure.
        _bind_synthetic_records_to_frozen(monkeypatch)
        records_root = _write_records_root(tmp_path / "records")
        out_dir = tmp_path / "out"

        with pytest.raises(schema.SchemaValidationError):
            driver.run_driver(
                records_root,
                out_dir,
                FROZEN_DIR,
                source_commit=TEST_SOURCE_COMMIT,
                launch_receipt=_launch_receipt(records_root),
                launch_ledger=_launch_ledger(records_root),
                activation_digest=_activation_digest(records_root),
                qa012_authority=qa012.CANONICAL_AUTHORITY_PATH,
            )

        assert not (out_dir / "runs" / "run-0001").exists()

    def test_non_json_closure_scalar_does_not_consume_run_slot(
        self, tmp_path, monkeypatch
    ):
        _bind_synthetic_records_to_frozen(monkeypatch)
        records_root = _write_records_root(tmp_path / "records")
        checksums = tmp_path / "FINAL_CHECKSUMS.json"
        _write_final_checksums_json(checksums)
        out_dir = tmp_path / "out"

        with pytest.raises(schema.SchemaValidationError):
            driver.run_driver(
                records_root,
                out_dir,
                FROZEN_DIR,
                source_commit=TEST_SOURCE_COMMIT,
                launch_receipt=_launch_receipt(records_root),
                launch_ledger=_launch_ledger(records_root),
                activation_digest=_activation_digest(records_root),
                d6_checksums=checksums,
                d6_main_tex_sha256=float("nan"),
                qa012_authority=qa012.CANONICAL_AUTHORITY_PATH,
            )

        assert not (out_dir / "runs" / "run-0001").exists()


# ---------------------------------------------------------------------------
# CLI — exit codes mirror verify.py / the assembler (0 / 2 / 3 / 4)
# ---------------------------------------------------------------------------


class TestCli:
    def test_exit_code_constants_pinned(self):
        assert driver.EXIT_PASS == 0
        assert driver.EXIT_USAGE_ERROR == 2
        assert driver.EXIT_INGRESS_ERROR == 3
        assert driver.EXIT_INTERNAL_ERROR == 4

    def test_main_end_to_end_source_pass(self, tmp_path, monkeypatch):
        _bind_synthetic_records_to_frozen(monkeypatch)
        records_root = _write_records_root(tmp_path / "records")
        out_dir = tmp_path / "out"
        checksums = tmp_path / "FINAL_CHECKSUMS.json"
        _write_final_checksums_json(checksums)
        rc = driver.main(
            [
                "--records-root",
                str(records_root),
                "--out-dir",
                str(out_dir),
                "--frozen-dir",
                str(FROZEN_DIR),
                "--source-commit",
                TEST_SOURCE_COMMIT,
                "--launch-receipt",
                str(_launch_receipt(records_root)),
                "--launch-ledger",
                str(_launch_ledger(records_root)),
                "--activation-digest",
                _activation_digest(records_root),
                "--d6-checksums",
                str(checksums),
                *_qa012_cli_args(),
            ]
        )
        assert rc == EXIT_PASS
        published_tree = out_dir / "runs" / "run-0001" / "tree"
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
        rc = driver.main(
            [
                "--records-root",
                str(tmp_path / "nope"),
                "--out-dir",
                str(tmp_path / "out"),
                "--frozen-dir",
                str(FROZEN_DIR),
                "--source-commit",
                TEST_SOURCE_COMMIT,
                "--launch-receipt",
                str(tmp_path / driver.LAUNCH_RECEIPT_NAME),
                "--launch-ledger",
                str(tmp_path / "missing-ledger.json"),
                "--activation-digest",
                PLACEHOLDER_ACTIVATION_DIGEST,
                "--d6-checksums",
                str(checksums),
                *_qa012_cli_args(),
            ]
        )
        assert rc == EXIT_INGRESS_ERROR

    def test_missing_frozen_is_ingress_error(self, tmp_path):
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
                str(tmp_path / "no-frozen"),
                "--source-commit",
                TEST_SOURCE_COMMIT,
                "--launch-receipt",
                str(_launch_receipt(records_root)),
                "--launch-ledger",
                str(_launch_ledger(records_root)),
                "--activation-digest",
                _activation_digest(records_root),
                "--d6-checksums",
                str(checksums),
                *_qa012_cli_args(),
            ]
        )
        assert rc == EXIT_INGRESS_ERROR

    def test_missing_qa012_input_is_usage_error(self, tmp_path, monkeypatch):
        _bind_synthetic_records_to_frozen(monkeypatch)
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
                TEST_SOURCE_COMMIT,
                "--launch-receipt",
                str(_launch_receipt(records_root)),
                "--launch-ledger",
                str(_launch_ledger(records_root)),
                "--activation-digest",
                _activation_digest(records_root),
                "--d6-checksums",
                str(checksums),
            ]
        )
        assert rc == EXIT_USAGE_ERROR
