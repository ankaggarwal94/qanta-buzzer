"""RED suite — create-once publication, receipts, run-scoped supersede.

Covers: R-016, R-036, R-039.
Spec: .correctless/specs/camera-ready-aims-evidence.md
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from reproducibility.colm_aims_2026 import receipt as receipt_mod
from reproducibility.colm_aims_2026 import schema, verifier
from tests._colm_aims_helpers import (  # noqa: F401  (colm_no_network is an autouse fixture)
    VERDICT_SOURCE_PASS,
    build_package,
    colm_no_network,
    expected_code_sha256,
    expected_tree_sha256,
    make_ledger,
    make_profile,
    repo_head_commit,
    sha256_file,
    tree_hashes,
)


# ---------------------------------------------------------------------------
# R-016: create-once publication via the no-replace primitives
# ---------------------------------------------------------------------------


def test_write_profile_publishes_parseable_artifact(tmp_path: Path):
    # Tests R-016 [integration]: a publish produces exactly one parseable
    # artifact at the final path.
    target = tmp_path / "out" / "profile.json"
    profile = make_profile()
    schema.write_profile(target, profile)
    assert target.is_file()
    assert schema.decode_profile(target.read_bytes()) == profile


def test_second_publish_to_existing_path_fails_not_clobbers(tmp_path: Path):
    # Tests R-016 [integration]: a second publish to an existing path fails
    # rather than clobbering (create-once, no-replace).
    target = tmp_path / "profile.json"
    profile = make_profile()
    schema.write_profile(target, profile)
    original = target.read_bytes()
    other = make_profile()
    other["cells"][0]["cell_id"] = "cell-0002"
    with pytest.raises(FileExistsError):
        schema.write_profile(target, other)
    assert target.read_bytes() == original


def test_interrupted_publish_leaves_no_parseable_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Tests R-016 [integration]: an interrupted publish (kill before the
    # publication step) leaves NO parseable partial artifact at the final
    # path — regardless of which durable-publish primitive the writer uses,
    # the final-name step is what we sever.
    target = tmp_path / "profile.json"

    def crash(*args, **kwargs):
        raise OSError("simulated kill mid-publish")

    monkeypatch.setattr(os, "link", crash)
    monkeypatch.setattr(os, "replace", crash)
    monkeypatch.setattr(os, "rename", crash)
    with pytest.raises(OSError):
        schema.write_profile(target, make_profile())
    assert not target.exists(), "no artifact may exist at the final path"


def test_retry_after_interrupt_succeeds_with_exactly_one_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Tests R-016 [integration]: a retried publish of identical content
    # succeeds with exactly one resulting artifact, and the staging-debris
    # policy is pinned. DECISION: auto-reclaim — after interrupt + retry the
    # directory holds exactly the published artifact, no staging debris.
    target = tmp_path / "profile.json"
    profile = make_profile()

    def crash(*args, **kwargs):
        raise OSError("simulated kill mid-publish")

    monkeypatch.setattr(os, "link", crash)
    monkeypatch.setattr(os, "replace", crash)
    monkeypatch.setattr(os, "rename", crash)
    with pytest.raises(OSError):
        schema.write_profile(target, profile)
    monkeypatch.undo()

    schema.write_profile(target, profile)  # retry, identical content
    assert schema.decode_profile(target.read_bytes()) == profile
    entries = sorted(p.name for p in tmp_path.iterdir())
    assert entries == ["profile.json"], (
        f"staging debris or duplicate artifacts left behind: {entries}"
    )


def test_writer_rounds_through_stopdff_v5_grade_primitives(tmp_path: Path):
    # Tests R-016 [integration]: the guarantees match the
    # scripts/stopdff_v5/fileio.py no-replace contract — an existing
    # DIRECTORY at the destination also fails closed instead of being
    # replaced.
    target = tmp_path / "profile.json"
    target.mkdir()
    with pytest.raises((FileExistsError, IsADirectoryError, ValueError, OSError)):
        schema.write_profile(target, make_profile())
    assert target.is_dir()  # untouched


def test_write_profile_routes_through_fileio_create_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Tests R-016 [integration] (audit BL-4a): the evidence writer must route
    # its final-name publication through scripts/stopdff_v5/fileio's
    # create-once primitive. DECISION: the namespace calls the primitive via
    # module attribute (`fileio.create_once_bytes(...)`) so routing stays
    # interceptable/auditable — a from-import alias fails this test by design.
    import scripts.stopdff_v5.fileio as fileio

    calls: list[Path] = []
    real = fileio.create_once_bytes

    def recorder(path, data, **kwargs):
        calls.append(Path(path))
        return real(path, data, **kwargs)

    monkeypatch.setattr(fileio, "create_once_bytes", recorder)
    target = tmp_path / "profile.json"
    schema.write_profile(target, make_profile())
    assert calls and calls[-1] == target, (
        "write_profile must publish via fileio.create_once_bytes"
    )
    assert target.is_file()


def test_publish_evidence_package_routes_through_dir_create_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Tests R-016/R-039 [integration] (audit BL-4a): directory publication
    # must route through fileio.publish_dir_create_once (the mkdir-claim +
    # rename discipline that fails closed on ANY pre-existing destination).
    import scripts.stopdff_v5.fileio as fileio

    dests: list[Path] = []
    real = fileio.publish_dir_create_once

    def recorder(staged, dest, **kwargs):
        dests.append(Path(dest))
        return real(staged, dest, **kwargs)

    monkeypatch.setattr(fileio, "publish_dir_create_once", recorder)
    runs_root = tmp_path / "runs"
    published = schema.publish_evidence_package(
        _stage(tmp_path, "a", '{"v": 1}\n'), runs_root, "run-0001"
    )
    assert dests and dests[-1] == published, (
        "publish_evidence_package must publish via fileio.publish_dir_create_once"
    )


# ---------------------------------------------------------------------------
# R-036: verification receipts
# ---------------------------------------------------------------------------

RECEIPT_REQUIRED_FIELDS = [
    "schema_version",
    "mode",
    "verdict",
    "legs",
    "input_tree_sha256",
    "expectations_anchor_sha256",
    "verifier_code_sha256",
    "timestamp_utc",
]


def _receipts(pkg) -> list[Path]:
    return sorted(p for p in pkg.receipts_dir.glob("**/*") if p.is_file())


def test_every_verifier_run_emits_schema_versioned_receipt(tmp_path: Path):
    # Tests R-036 [integration]: every verifier run emits a schema-versioned
    # JSON receipt with mode, verdict, per-leg outcomes, input-tree hash,
    # expectations-anchor hash, verifier code hash, timestamp.
    pkg = build_package(tmp_path)
    report = verifier.run_verifier(
        pkg.tree, mode="source", receipts_dir=pkg.receipts_dir
    )
    receipts = _receipts(pkg)
    assert len(receipts) == 1
    payload = json.loads(receipts[0].read_text("utf-8"))
    for field in RECEIPT_REQUIRED_FIELDS:
        assert field in payload, f"receipt missing {field!r}"
    assert payload["mode"] == "source"
    assert payload["verdict"] == VERDICT_SOURCE_PASS
    assert re.fullmatch(r"[0-9a-f]{64}", payload["input_tree_sha256"])
    assert re.fullmatch(r"[0-9a-f]{64}", payload["verifier_code_sha256"])
    assert report.receipt_path is not None
    assert Path(report.receipt_path) == receipts[0]


def test_receipt_written_outside_verified_tree(tmp_path: Path):
    # Tests R-036 [integration]: the receipt path is outside the verified
    # artifact tree (and the tree is byte-identical after the run — R-014).
    pkg = build_package(tmp_path)
    before = tree_hashes(pkg.tree)
    report = verifier.run_verifier(
        pkg.tree, mode="source", receipts_dir=pkg.receipts_dir
    )
    receipt = Path(report.receipt_path).resolve()
    assert pkg.tree.resolve() not in receipt.parents
    assert tree_hashes(pkg.tree) == before


def test_receipts_are_create_once_under_run_scoped_unique_names(tmp_path: Path):
    # Tests R-036 [integration]: receipts publish create-once under
    # run-scoped unique names — a second run adds a NEW receipt and leaves
    # the first byte-identical.
    pkg = build_package(tmp_path)
    verifier.run_verifier(pkg.tree, mode="source", receipts_dir=pkg.receipts_dir)
    first = _receipts(pkg)
    assert len(first) == 1
    first_bytes = first[0].read_bytes()

    verifier.run_verifier(pkg.tree, mode="source", receipts_dir=pkg.receipts_dir)
    second = _receipts(pkg)
    assert len(second) == 2, "second run must add a run-scoped receipt"
    assert first[0] in second
    assert first[0].read_bytes() == first_bytes
    names = {p.name for p in second}
    assert len(names) == 2, "receipt names must be unique per run"


def test_release_receipt_binds_expectations_anchor(tmp_path: Path):
    # Tests R-036 [integration]: release-mode receipts carry the
    # expectations-anchor hash binding the receipt to the anchored inputs.
    pkg = build_package(tmp_path)
    verifier.run_verifier(
        pkg.tree,
        mode="release",
        receipts_dir=pkg.receipts_dir,
        expectations=pkg.expectations_path,
    )
    payload = json.loads(_receipts(pkg)[0].read_text("utf-8"))
    assert payload["mode"] == "release"
    # Audit item 1: the anchor hash must equal a TEST-SIDE digest of the
    # actual expectations file bytes — a constant hex64 cannot pass.
    assert payload["expectations_anchor_sha256"] == sha256_file(
        pkg.expectations_path
    )
    assert payload["legs"], "per-leg outcomes must be recorded"


def test_receipt_input_tree_hash_is_deterministic(tmp_path: Path):
    # Tests R-036 [integration]: the input-tree hash is a function of the
    # tree bytes — two runs over the identical tree record the same hash.
    pkg = build_package(tmp_path)
    verifier.run_verifier(pkg.tree, mode="source", receipts_dir=pkg.receipts_dir)
    verifier.run_verifier(pkg.tree, mode="source", receipts_dir=pkg.receipts_dir)
    payloads = [json.loads(p.read_text("utf-8")) for p in _receipts(pkg)]
    assert payloads[0]["input_tree_sha256"] == payloads[1]["input_tree_sha256"]


def test_receipt_tree_hash_matches_independent_digest_and_tracks_bytes(
    tmp_path: Path,
):
    # Tests R-036 [integration] (audit BL-3a): input_tree_sha256 equals an
    # INDEPENDENTLY computed digest over the tree bytes, and changes when a
    # single tree file's bytes change — a hardcoded constant cannot pass.
    pkg = build_package(tmp_path)
    verifier.run_verifier(pkg.tree, mode="source", receipts_dir=pkg.receipts_dir)
    first = json.loads(_receipts(pkg)[0].read_text("utf-8"))
    assert first["input_tree_sha256"] == expected_tree_sha256(pkg.tree)

    # Mutate one byte-bearing file that no ingress parser consumes (the
    # sealed payload blob), so the run stays valid but the tree differs.
    (pkg.tree / "sealed-notes.bin").write_bytes(b"MUTATED-CANARY\n")
    verifier.run_verifier(pkg.tree, mode="source", receipts_dir=pkg.receipts_dir)
    second = json.loads(_receipts(pkg)[1].read_text("utf-8"))
    assert second["input_tree_sha256"] == expected_tree_sha256(pkg.tree)
    assert second["input_tree_sha256"] != first["input_tree_sha256"]


def test_receipt_verifier_code_hash_matches_namespace_bytes(tmp_path: Path):
    # Tests R-036 [integration] (audit BL-3b): verifier_code_sha256 equals a
    # test-side digest of the namespace's .py bytes, so the receipt actually
    # binds the verifier code that ran.
    pkg = build_package(tmp_path)
    verifier.run_verifier(pkg.tree, mode="source", receipts_dir=pkg.receipts_dir)
    payload = json.loads(_receipts(pkg)[0].read_text("utf-8"))
    assert payload["verifier_code_sha256"] == expected_code_sha256()


def test_receipt_emission_is_create_once_at_existing_path(tmp_path: Path):
    # Tests R-036 [integration] (audit BL-3c): writing a second receipt at an
    # existing receipt path fails rather than clobbering.
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "profile.json").write_text("{}", encoding="utf-8")
    receipts_dir = tmp_path / "receipts"
    payload = {
        "schema_version": 1,
        "mode": "source",
        "verdict": "FAIL",
        "legs": [],
    }
    path = receipt_mod.emit_receipt(
        payload, receipts_dir=receipts_dir, verified_tree=tree, run_id="run-x"
    )
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        receipt_mod.emit_receipt(
            {**payload, "verdict": "PASS_SOURCE_ONLY"},
            receipts_dir=receipts_dir,
            verified_tree=tree,
            run_id="run-x",
        )
    assert path.read_bytes() == original


def test_emit_receipt_routes_through_fileio_create_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # Tests R-036/R-016 [integration] (audit item 2): receipt emission must
    # route through the fileio create-once primitive — an `if exists: raise`
    # pre-check plus publish_bytes (a TOCTOU create-once) must not pass.
    # Same module-attribute routing DECISION as the write_profile test.
    import scripts.stopdff_v5.fileio as fileio

    calls: list[Path] = []
    real = fileio.create_once_bytes

    def recorder(path, data, **kwargs):
        calls.append(Path(path))
        return real(path, data, **kwargs)

    monkeypatch.setattr(fileio, "create_once_bytes", recorder)
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "profile.json").write_text("{}", encoding="utf-8")
    receipts_dir = tmp_path / "receipts"
    path = receipt_mod.emit_receipt(
        {"schema_version": 1, "mode": "source", "verdict": "FAIL", "legs": []},
        receipts_dir=receipts_dir,
        verified_tree=tree,
        run_id="run-routed",
    )
    assert calls and calls[-1] == Path(path), (
        "emit_receipt must publish via fileio.create_once_bytes"
    )
    assert Path(path).is_file()


# ---------------------------------------------------------------------------
# R-039: run-scoped publish dirs + pointer-only canonical selection
# ---------------------------------------------------------------------------


def _stage(tmp_path: Path, name: str, content: str) -> Path:
    staged = tmp_path / f"staged-{name}"
    staged.mkdir(parents=True)
    (staged / "profile.json").write_text(content, encoding="utf-8")
    return staged


def test_evidence_packages_publish_into_run_scoped_directories(tmp_path: Path):
    # Tests R-039 [unit]: evidence packages publish into run-scoped
    # directories under the runs root.
    runs_root = tmp_path / "runs"
    staged = _stage(tmp_path, "a", '{"v": 1}\n')
    published = schema.publish_evidence_package(staged, runs_root, "run-0001")
    assert published.is_dir()
    assert runs_root.resolve() in published.resolve().parents
    assert (published / "profile.json").read_text("utf-8") == '{"v": 1}\n'


def test_republishing_same_run_id_fails_closed(tmp_path: Path):
    # Tests R-039 [unit]: historical bytes are retained, never republished
    # over — the same run slot cannot be claimed twice.
    runs_root = tmp_path / "runs"
    published = schema.publish_evidence_package(
        _stage(tmp_path, "a", '{"v": 1}\n'), runs_root, "run-0001"
    )
    original = sha256_file(published / "profile.json")
    with pytest.raises(FileExistsError):
        schema.publish_evidence_package(
            _stage(tmp_path, "b", '{"v": 2}\n'), runs_root, "run-0001"
        )
    assert sha256_file(published / "profile.json") == original


def test_publish_into_preexisting_empty_dir_fails_closed(tmp_path: Path):
    # Tests R-039/R-016 [unit] (audit BL-4c): a destination that already
    # exists as an EMPTY directory fails closed — matching the
    # publish_dir_create_once mkdir-claim semantics (a bare os.rename would
    # silently replace an empty dir; the mkdir gate must refuse it).
    runs_root = tmp_path / "runs"
    slot = runs_root / "run-0001"
    slot.mkdir(parents=True)  # pre-claimed empty slot
    staged = _stage(tmp_path, "a", '{"v": 1}\n')
    with pytest.raises(FileExistsError):
        schema.publish_evidence_package(staged, runs_root, "run-0001")
    assert slot.is_dir()
    assert list(slot.iterdir()) == [], "the empty slot must not be filled"
    assert staged.is_dir(), "staged dir stays with the caller on refusal"
    assert (staged / "profile.json").is_file()


def test_retire_is_new_run_dir_plus_pointer_change(tmp_path: Path):
    # Tests R-039 [unit]: retiring a defective artifact is a ledger status
    # change plus a NEW run directory — the old run's bytes are retained.
    runs_root = tmp_path / "runs"
    first = schema.publish_evidence_package(
        _stage(tmp_path, "a", '{"v": 1}\n'), runs_root, "run-0001"
    )
    first_hash = sha256_file(first / "profile.json")
    second = schema.publish_evidence_package(
        _stage(tmp_path, "b", '{"v": 2}\n'), runs_root, "run-0002"
    )
    assert first != second
    assert sha256_file(first / "profile.json") == first_hash  # retained

    commit = repo_head_commit()
    superseded = make_ledger(source_commit=commit, canonical_run_id="run-0002")
    resolved = verifier.resolve_canonical_package(runs_root, superseded)
    assert resolved == second


def test_canonical_selection_only_via_ledger_pointer(tmp_path: Path):
    # Tests R-039 [unit]: canonical selection happens ONLY via the
    # ledger/expectations pointer — never "newest wins".
    runs_root = tmp_path / "runs"
    schema.publish_evidence_package(
        _stage(tmp_path, "a", '{"v": 1}\n'), runs_root, "run-0001"
    )
    schema.publish_evidence_package(
        _stage(tmp_path, "b", '{"v": 2}\n'), runs_root, "run-0002"
    )
    commit = repo_head_commit()
    pointed = make_ledger(source_commit=commit, canonical_run_id="run-0001")
    resolved = verifier.resolve_canonical_package(runs_root, pointed)
    assert resolved.name == "run-0001"  # pointer wins over recency

    unpointed = make_ledger(source_commit=commit)
    del unpointed["canonical_run_id"]
    with pytest.raises(schema.ColmAimsError):
        verifier.resolve_canonical_package(runs_root, unpointed)


def test_pointer_to_missing_run_is_typed_error_not_fallback(tmp_path: Path):
    # Tests R-039 [unit]: a dangling pointer errors — it never silently falls
    # back to another run directory.
    runs_root = tmp_path / "runs"
    schema.publish_evidence_package(
        _stage(tmp_path, "a", '{"v": 1}\n'), runs_root, "run-0001"
    )
    ghost = make_ledger(source_commit=repo_head_commit(), canonical_run_id="run-9999")
    with pytest.raises(schema.ColmAimsError):
        verifier.resolve_canonical_package(runs_root, ghost)
