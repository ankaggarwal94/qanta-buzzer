"""Negative mutation suite for the standalone checker (see ACCEPTANCE_CONTRACT.md).

The synthetic valid-package factory lives in ``selftest_fixtures``
(``build_valid_package`` is re-exported here for its historical import path).
run_self_test() applies a battery of mutations to that package and asserts the
checker REJECTS every one, exercising BOTH rejection layers per mutation:

- stale-checksum adversary: the mutation is validated as applied, so the SHA256SUMS
  sweep (or a deeper layer) must reject;
- checksum-consistent adversary: SHA256SUMS is regenerated after the mutation (except
  for the mutations that tamper SHA256SUMS itself), so the semantic recompute /
  binding layer alone must reject.

A final-profile phase then builds a receipt-bearing final package (production
``writers.build_evidenced_prerequisite_receipt`` path), proves it validates
end-to-end under ``require_final_profile=True``, and asserts receipt/evidence-ledger
forgeries are rejected with regenerated checksums.

Synthetic fixtures only; the Modal mutation gate re-runs this same synthetic-fixture
suite inside the image (it does not mutate the real package).
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable

from . import checker
from .identity import compute_id, sha256_bytes, sha256_file
from .writers import write_sha256sums

# The fixture factory lives in ``selftest_fixtures``; re-export the historical
# names so ``selftest.build_valid_package`` (and friends) keep working for the
# gates below and for every external consumer.
from .selftest_fixtures import (  # noqa: F401  (fixture re-exports)
    CATEGORIES,
    _hex,
    _mint_final_prerequisite_receipts,
    _synth_rows,
    build_valid_package,
)
from . import selftest_fixtures as _selftest_fixtures


def __getattr__(name: str) -> Any:
    # ``_SYNTH_FVI_STUDY`` is a mutable module-level cache owned by
    # ``selftest_fixtures``; delegate so callers observe its rebinding
    # instead of a stale import-time snapshot.
    if name == "_SYNTH_FVI_STUDY":
        return _selftest_fixtures._SYNTH_FVI_STUDY
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# --- mutations --------------------------------------------------------------------


def _load(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def _save(p: Path, obj: Any) -> None:
    p.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def _first_cell(run_root: Path) -> Path:
    return sorted((run_root / "cells").glob("*.json"))[0]


def _mut_stale_cache(rr, b):
    p = _first_cell(rr); rec = _load(p)
    k = next(iter(rec["index_shift_by_item"]))
    rec["index_shift_by_item"][k] = int(rec["index_shift_by_item"][k]) + 3
    _save(p, rec)


def _mut_flip_verdict(rr, b):
    p = _first_cell(rr); rec = _load(p)
    flip = {"PASS": "FAIL", "WARN": "PASS", "FAIL": "PASS"}
    rec["verdict"] = flip.get(rec.get("verdict"), "PASS")
    _save(p, rec)


def _mut_hide_coverage(rr, b):
    p = _first_cell(rr); rec = _load(p)
    rec["coverage"]["clean"] = not rec["coverage"]["clean"]
    _save(p, rec)


def _mut_tamper_ceiling(rr, b):
    p = _first_cell(rr); rec = _load(p)
    key = next(iter(rec["ceiling_flags"]))
    rec["ceiling_flags"][key] = not rec["ceiling_flags"][key]
    _save(p, rec)


def _mut_wrong_family_ci(rr, b):
    agg = _load(rr / "aggregate.json"); agg["family"]["ci"] = [99.0, 100.0]; _save(rr / "aggregate.json", agg)


def _mut_wrong_family_verdict(rr, b):
    agg = _load(rr / "aggregate.json")
    agg["family"]["verdict"] = "FAIL" if agg["family"]["verdict"] != "FAIL" else "PASS"
    _save(rr / "aggregate.json", agg)


def _mut_wrong_release(rr, b):
    agg = _load(rr / "aggregate.json")
    agg["release_status"] = "INVALID" if agg["release_status"] == "VALID" else "VALID"
    _save(rr / "aggregate.json", agg)


def _mut_dual_backend(rr, b):
    shutil.copy(rr / "run_manifest.json", rr / "command_manifest.json")


def _mut_missing_backend_manifest(rr, b):
    (rr / "run_manifest.json").unlink()


def _mut_wrong_seed(rr, b):
    pm = _load(rr / "bootstrap_plan.json"); pm["identity"]["seed"] = 2
    pm["id"] = compute_id(pm["identity"]); _save(rr / "bootstrap_plan.json", pm)


def _mut_wrong_replicates(rr, b):
    pm = _load(rr / "bootstrap_plan.json"); pm["identity"]["replicate_count"] = 999
    pm["id"] = compute_id(pm["identity"]); _save(rr / "bootstrap_plan.json", pm)


def _mut_tamper_run_spec_id(rr, b):
    sm = _load(rr / "run_spec.json"); sm["id"] = _hex("0"); _save(rr / "run_spec.json", sm)


def _mut_tamper_plan_hash(rr, b):
    pm = _load(rr / "bootstrap_plan.json")
    pm["identity"]["resample_index_sha256"] = _hex("0")
    pm["id"] = compute_id(pm["identity"]); _save(rr / "bootstrap_plan.json", pm)


def _mut_fresh_with_resume(rr, b):
    (rr / "attempts.jsonl").write_text(
        json.dumps({"attempt": 1, "mode": "fresh", "command": ["dp_sweep", "--resume"]}) + "\n",
        encoding="utf-8")


def _mut_resume_without_bare(rr, b):
    with open(rr / "attempts.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"attempt": 2, "mode": "resume", "command": ["dp_sweep"]}) + "\n")


def _mut_overwrite(rr, b):
    with open(rr / "attempts.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"attempt": 2, "mode": "resume", "command": ["dp_sweep", "--resume", "--overwrite"]}) + "\n")


def _mut_unsafe_checksum(rr, b):
    with open(rr / "SHA256SUMS", "a", encoding="utf-8") as f:
        f.write(f"{_hex('a')}  ../evil.txt\n")


def _mut_duplicate_checksum(rr, b):
    lines = (rr / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    lines.append(lines[0])
    (rr / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mut_symlink(rr, b):
    target = rr / "aggregate.json"
    link = rr / "evil_link.json"
    os.symlink(target, link)
    with open(rr / "SHA256SUMS", "a", encoding="utf-8") as f:
        f.write(f"{sha256_file(target)}  evil_link.json\n")


def _mut_checksum_value(rr, b):
    lines = (rr / "SHA256SUMS").read_text(encoding="utf-8").splitlines()
    parts = lines[0].split()
    lines[0] = f"{_hex('0')}  {parts[1]}"
    (rr / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _mut_invalid_png(rr, b):
    for png in (rr / "figures").glob("*.png"):
        png.write_bytes(b"not a png")
        break


def _mut_truncated_png_after_ihdr(rr, b):
    for png in (rr / "figures").glob("*.png"):
        png.write_bytes(png.read_bytes()[:24])
        break


def _mut_missing_external_artifacts(rr, b):
    (rr / "external_artifacts.json").unlink()


def _mut_unconverged_completed(rr, b):
    agg = _load(rr / "aggregate.json"); agg["fvi_selected"]["max_iterations"] = 1
    _save(rr / "aggregate.json", agg)


def _mut_fingerprint(rr, b):
    p = _first_cell(rr); rec = _load(p); rec["fingerprint_id"] = _hex("0"); _save(p, rec)


def _mut_adapter_row_hash(rr, b):
    # corrupt fit rows after manifest was written -> validate-adapter must reject
    with open(b / "fit_rows.jsonl.gz", "ab") as f:
        f.write(b"\x00corrupt")


def _mut_adapter_calibration_hash(rr, b):
    with open(b / "calibration.json", "ab") as handle:
        handle.write(b"\n")


def _mut_backend_adapter_binding(rr, b):
    path = rr / "run_manifest.json"
    record = _load(path)
    record["identity"]["adapter_bundle_id"] = _hex("9")
    record["id"] = compute_id(record["identity"])
    _save(path, record)


def _mut_attempt_adapter_binding(rr, b):
    path = rr / "attempts.jsonl"
    attempt = json.loads(path.read_text(encoding="utf-8").strip())
    attempt["adapter_id"] = _hex("9")
    path.write_text(json.dumps(attempt) + "\n", encoding="utf-8")


def _mut_cell_adapter_binding(rr, b):
    path = _first_cell(rr)
    record = _load(path)
    record["adapter_bundle_id"] = _hex("9")
    _save(path, record)


def _mut_fingerprint_adapter_hash(rr, b):
    path = _first_cell(rr)
    record = _load(path)
    record["fingerprint_identity"]["adapter_fit_rows_sha256"] = _hex("9")
    record["fingerprint_id"] = compute_id(record["fingerprint_identity"])
    _save(path, record)


def _mut_aggregate_adapter_binding(rr, b):
    path = rr / "aggregate.json"
    aggregate = _load(path)
    aggregate["adapter_bundle_id"] = _hex("9")
    _save(path, aggregate)


def _mut_aggregate_fvi_binding(rr, b):
    # Offset from the recorded value: a fixed constant can collide with the
    # study-selected setting, turning this into a checksum-only byte change.
    path = rr / "aggregate.json"
    aggregate = _load(path)
    aggregate["fvi_selected"]["max_iterations"] = (
        int(aggregate["fvi_selected"]["max_iterations"]) + 100
    )
    _save(path, aggregate)


def _mut_unknown_attempt_mode(rr, b):
    path = rr / "attempts.jsonl"
    attempt = json.loads(path.read_text(encoding="utf-8").strip())
    attempt["mode"] = "replay"
    path.write_text(json.dumps(attempt) + "\n", encoding="utf-8")


def _mut_fingerprint_kind(rr, b):
    path = _first_cell(rr)
    record = _load(path)
    record["fingerprint_identity"]["kind"] = "forged"
    record["fingerprint_id"] = compute_id(record["fingerprint_identity"])
    _save(path, record)


def _mut_fingerprint_producers(rr, b):
    path = _first_cell(rr)
    record = _load(path)
    record["fingerprint_identity"]["producer_hashes"] = {
        "checker.py": _hex("f"),
        "sweep.py": _hex("f"),
    }
    record["fingerprint_id"] = compute_id(record["fingerprint_identity"])
    _save(path, record)


def _mut_cell_gate_override(rr, b):
    path = _first_cell(rr)
    record = _load(path)
    record["mc_gate_overridden"] = not record["mc_gate_overridden"]
    _save(path, record)


def _mut_backend_environment(rr, b):
    path = rr / "run_manifest.json"
    record = _load(path)
    record["environment"] = {
        "python_version": "0.0.0",
        "package_versions": {"numpy": "0"},
    }
    _save(path, record)


def _mut_missing_fvi_evidence(rr, b):
    path = rr / "external_artifacts.json"
    record = _load(path)
    record["artifacts"] = [
        artifact
        for artifact in record["artifacts"]
        if artifact.get("role") != "fvi_study"
    ]
    _save(path, record)


def _mut_missing_attempt_result(rr, b):
    next((rr / "attempt_results").glob("*.json")).unlink()


def _mut_attempt_result_counts(rr, b):
    path = next((rr / "attempt_results").glob("*.json"))
    record = _load(path)
    record["completed"] = 999
    _save(path, record)


# Mutations that tamper the SHA256SUMS file itself: regenerating the checksum
# inventory would erase the mutation, so these three exercise ONLY the
# stale-checksum layer (that layer is exactly what they test).
_CHECKSUM_CONTENT_MUTATIONS = frozenset({
    "unsafe_checksum_traversal",
    "duplicate_checksum_entry",
    "checksum_value_mismatch",
})


def _rewrite_receipt_ledger_entry(rr: Path, role: str, path: Path) -> None:
    """Rebind one external-artifact ledger entry to a forged receipt's bytes."""
    record = _load(rr / "external_artifacts.json")
    manifest = _load(path)
    data = path.read_bytes()
    for artifact in record["artifacts"]:
        if artifact.get("role") == role:
            artifact["content_id"] = manifest["id"]
            artifact["sha256"] = sha256_bytes(data)
            artifact["byte_size"] = len(data)
    _save(rr / "external_artifacts.json", record)


def _mut_final_receipt_evidence_bytes_tampered(rr, b):
    # Digest binding: packaged evidence bytes no longer hash to the receipt's
    # evidence_sha256, even though every checksum in SHA256SUMS is consistent.
    path = rr / "evidence" / "prerequisite_receipts" / "smoke.evidence.json"
    path.write_bytes(path.read_bytes() + b"\n")


def _mut_final_receipt_id_forged(rr, b):
    path = rr / "evidence" / "prerequisite_receipts" / "mutation.json"
    manifest = _load(path)
    manifest["id"] = _hex("0")
    _save(path, manifest)
    _rewrite_receipt_ledger_entry(rr, "prerequisite_receipt_mutation", path)


def _mut_final_receipt_binding_mismatch(rr, b):
    # Self-consistent forgery: the receipt hashes to its own identity but was
    # minted for a different adapter; the run-spec receipt pin must reject it.
    path = rr / "evidence" / "prerequisite_receipts" / "determinism.json"
    manifest = _load(path)
    manifest["identity"]["bindings"]["adapter_bundle_id"] = _hex("9")
    manifest["id"] = compute_id(manifest["identity"])
    _save(path, manifest)
    _rewrite_receipt_ledger_entry(rr, "prerequisite_receipt_determinism", path)


def _mut_final_missing_receipt_role(rr, b):
    record = _load(rr / "external_artifacts.json")
    record["artifacts"] = [
        artifact
        for artifact in record["artifacts"]
        if artifact.get("role") != "prerequisite_receipt_smoke"
    ]
    _save(rr / "external_artifacts.json", record)


def _mut_final_spec_drops_receipts(rr, b):
    # A self-consistent final spec that claims no receipts: the profile-variant
    # receipt requirement (and every run_spec_id binding) must reject it.
    path = rr / "run_spec.json"
    manifest = _load(path)
    manifest["identity"]["evidence_roots"]["prerequisite_receipts"] = {}
    manifest["id"] = compute_id(manifest["identity"])
    _save(path, manifest)


def _mut_extra_evidence_file(rr, b):
    # Attested-but-unaudited payload: an extra file under evidence/ outside
    # the bound content roots and receipt names stays checksum-consistent
    # once SHA256SUMS is regenerated, so only the evidence-namespace
    # membership of the package path policy can reject it.
    (rr / "evidence" / "backdoor.bin").write_bytes(b"attested but unaudited\n")


def _mut_extra_evidence_dir(rr, b):
    # An empty directory holds no bytes for SHA256SUMS or the bound-content
    # inventories to see; only the entry-free-directory rule of the package
    # path policy can reject the smuggled name.
    (rr / "evidence" / "source_snapshot" / "source" / "rogue").mkdir()


_FINAL_RECEIPT_MUTATIONS: dict[str, Callable] = {
    "final_receipt_evidence_bytes_tampered": (
        _mut_final_receipt_evidence_bytes_tampered
    ),
    "final_receipt_id_forged": _mut_final_receipt_id_forged,
    "final_receipt_binding_mismatch": _mut_final_receipt_binding_mismatch,
    "final_missing_prerequisite_receipt_role": _mut_final_missing_receipt_role,
    "final_spec_drops_prerequisite_receipts": _mut_final_spec_drops_receipts,
    # Evidence-namespace tampers ride the final phase so the pinned
    # receipt_evidence.MUTATION_ROSTER stays untouched (same pattern as the
    # receipt mutations above); the final variant also carries the fullest
    # evidence/ namespace (prerequisite_receipts/ included).
    "final_extra_evidence_file": _mut_extra_evidence_file,
    "final_extra_evidence_dir": _mut_extra_evidence_dir,
}


_RUN_MUTATIONS: dict[str, Callable] = {
    "stale_cache": _mut_stale_cache,
    "cell_verdict_serialized_not_trusted": _mut_flip_verdict,
    "coverage_clean_serialized_not_trusted": _mut_hide_coverage,
    "ceiling_flags_tampered": _mut_tamper_ceiling,
    "wrong_family_maximum_statistic": _mut_wrong_family_ci,
    "family_verdict_hides_cell_warn": _mut_wrong_family_verdict,
    "wrong_release_status": _mut_wrong_release,
    "dual_backend_manifests": _mut_dual_backend,
    "missing_backend_manifest": _mut_missing_backend_manifest,
    "wrong_bootstrap_seed": _mut_wrong_seed,
    "wrong_bootstrap_count": _mut_wrong_replicates,
    "tampered_run_spec_id": _mut_tamper_run_spec_id,
    "wrong_bootstrap_plan_hash": _mut_tamper_plan_hash,
    "fresh_attempt_with_resume": _mut_fresh_with_resume,
    "resume_without_bare_resume": _mut_resume_without_bare,
    "overwrite_in_evidence_run": _mut_overwrite,
    "unsafe_checksum_traversal": _mut_unsafe_checksum,
    "duplicate_checksum_entry": _mut_duplicate_checksum,
    "symlink_in_package": _mut_symlink,
    "checksum_value_mismatch": _mut_checksum_value,
    "invalid_png": _mut_invalid_png,
    "truncated_png_after_ihdr": _mut_truncated_png_after_ihdr,
    "missing_external_artifacts": _mut_missing_external_artifacts,
    "unconverged_fvi_marked_completed": _mut_unconverged_completed,
    "cell_fingerprint_tampered": _mut_fingerprint,
    "adapter_calibration_bytes_tampered": _mut_adapter_calibration_hash,
    "backend_adapter_binding": _mut_backend_adapter_binding,
    "attempt_adapter_binding": _mut_attempt_adapter_binding,
    "cell_adapter_binding": _mut_cell_adapter_binding,
    "fingerprint_adapter_hash_binding": _mut_fingerprint_adapter_hash,
    "aggregate_adapter_binding": _mut_aggregate_adapter_binding,
    "aggregate_fvi_binding": _mut_aggregate_fvi_binding,
    "unknown_attempt_mode": _mut_unknown_attempt_mode,
    "fingerprint_kind": _mut_fingerprint_kind,
    "fingerprint_producer_binding": _mut_fingerprint_producers,
    "cell_gate_override": _mut_cell_gate_override,
    "backend_environment_binding": _mut_backend_environment,
    "missing_fvi_evidence": _mut_missing_fvi_evidence,
    "missing_attempt_result": _mut_missing_attempt_result,
    "attempt_result_counts": _mut_attempt_result_counts,
}


def run_final_receipt_self_test(
    base_dir: Path,
) -> tuple[bool, list[dict[str, Any]]]:
    """Final-profile receipt/evidence-ledger negative phase.

    Builds a receipt-bearing final package, proves the checker accepts it
    end-to-end under ``require_final_profile=True``, then asserts every
    receipt/evidence-ledger forgery and evidence-namespace tamper is rejected
    by the semantic layer alone (SHA256SUMS is regenerated after each
    mutation).
    """
    base_dir = Path(base_dir)
    valid_dir = base_dir / "valid"
    built = build_valid_package(valid_dir, final_variant=True)

    results: list[dict[str, Any]] = []
    baseline = checker.validate_run(
        built["run_root"], backend="modal",
        adapter_bundle=built["adapter_bundle"],
        require_final_profile=True, require_package=True,
    )
    results.append({"mutation": "<final baseline valid>", "expected": "PASS",
                    "passed_check": baseline.passed, "ok": baseline.passed,
                    "errors": baseline.errors[:3]})
    all_ok = baseline.passed

    for name, fn in _FINAL_RECEIPT_MUTATIONS.items():
        mdir = base_dir / f"mut_{name}"
        if mdir.exists():
            shutil.rmtree(mdir)
        shutil.copytree(valid_dir, mdir, symlinks=True)
        rr, bundle = mdir / "runs" / "run", mdir / "adapter_bundle"
        fn(rr, bundle)
        write_sha256sums(rr)
        res = checker.validate_run(
            rr, backend="modal", adapter_bundle=bundle,
            require_final_profile=True, require_package=True,
        )
        rejected = not res.passed
        results.append({"mutation": name, "expected": "REJECT",
                        "passed_check": res.passed, "ok": rejected,
                        "errors": res.errors[:2]})
        all_ok = all_ok and rejected

    return all_ok, results


def run_self_test(base_dir: Path) -> tuple[bool, list[dict[str, Any]]]:
    """Run the normative negative-mutation gate on synthetic fixtures.

    Every run-package mutation is validated twice: once as applied (the
    stale-checksum sweep or a deeper layer must reject), and once after
    ``SHA256SUMS`` is regenerated (the semantic recompute/binding layer alone
    must reject a checksum-consistent adversary). The three SHA256SUMS-content
    mutations run only the first pass, since regeneration would erase them.
    A final-profile receipt phase then runs via
    :func:`run_final_receipt_self_test`.

    Returns
    -------
    tuple[bool, list[dict[str, Any]]]
        ``all_ok`` plus one result entry per mutation. On success the entries
        match ``receipt_evidence.MUTATION_ROSTER`` exactly (the mutation-gate
        receipt evidence pins that roster, so passing final-phase entries are
        not appended); on failure the failing final-phase entries are appended
        for diagnostics.
    """
    base_dir = Path(base_dir)
    valid_dir = base_dir / "valid"
    built = build_valid_package(valid_dir)

    results: list[dict[str, Any]] = []
    baseline = checker.validate_run(
        built["run_root"], backend="modal", adapter_bundle=built["adapter_bundle"],
        require_final_profile=False, require_package=True,
    )
    results.append({"mutation": "<baseline valid>", "expected": "PASS",
                    "passed_check": baseline.passed, "ok": baseline.passed,
                    "errors": baseline.errors[:3]})

    all_ok = baseline.passed

    for name, fn in _RUN_MUTATIONS.items():
        mdir = base_dir / f"mut_{name}"
        if mdir.exists():
            shutil.rmtree(mdir)
        shutil.copytree(valid_dir, mdir, symlinks=True)
        rr, bundle = mdir / "run", mdir / "adapter_bundle"
        fn(rr, bundle)
        stale = checker.validate_run(
            rr, backend="modal", adapter_bundle=bundle,
            require_final_profile=False, require_package=True,
        )
        if name in _CHECKSUM_CONTENT_MUTATIONS:
            accepted_by_any_layer = stale.passed
            errors = stale.errors[:2]
        else:
            write_sha256sums(rr)
            semantic = checker.validate_run(
                rr, backend="modal", adapter_bundle=bundle,
                require_final_profile=False, require_package=True,
            )
            accepted_by_any_layer = stale.passed or semantic.passed
            errors = semantic.errors[:2]
        rejected = not accepted_by_any_layer
        results.append({"mutation": name, "expected": "REJECT",
                        "passed_check": accepted_by_any_layer,
                        "ok": rejected, "errors": errors})
        all_ok = all_ok and rejected

    # adapter-level mutation
    adir = base_dir / "mut_adapter_row_hash"
    if adir.exists():
        shutil.rmtree(adir)
    shutil.copytree(valid_dir, adir, symlinks=True)
    _mut_adapter_row_hash(adir / "run", adir / "adapter_bundle")
    ares = checker.validate_adapter(adir / "adapter_bundle")
    results.append({"mutation": "invalid_adapter_row_hash", "expected": "REJECT",
                    "passed_check": ares.passed, "ok": not ares.passed, "errors": ares.errors[:2]})
    all_ok = all_ok and not ares.passed

    final_ok, final_results = run_final_receipt_self_test(base_dir / "final")
    all_ok = all_ok and final_ok
    if not final_ok:
        # Diagnostics only: the mutation-gate receipt evidence pins the result
        # roster to receipt_evidence.MUTATION_ROSTER, and a failing gate never
        # mints a receipt, so the extra entries are safe to surface here.
        results.extend(final_results)

    return all_ok, results
