"""Release-mode verification: R-012 (fail-closed legs), R-013 (independent
anchoring/containment), R-044 (anchored grid pins), R-021 (end-to-end
mutation wiring via the real CLI), R-065 (ledger<->anchor commit equality),
R-066 (git-object fail-closed), R-035 (manifest reconciliation), R-069/R-039
(canonical selection wired into the release path), R-033 (vacuous inputs).

GREENFIELD RED: fails at collection until the namespace exists.
"""
from __future__ import annotations

import json

import pytest

from reproducibility.colm_aims_2026 import schema, verifier

from tests._colm_aims_v2_helpers import (
    ANCHORED_GRID_PREFIX,
    ANCHORED_INFERENCE_PREFIX,
    EXIT_GATE_FAIL,
    EXIT_INGRESS_ERROR,
    EXIT_PASS,
    EXIT_USAGE_ERROR,
    LEG_ANCHOR_COMMIT,
    LEG_CANONICAL_SELECTION,
    LEG_GIT_OBJECT,
    LEG_LEDGER_ANCHOR_EQ,
    REMEDIATION_CLASSES,
    VERDICT_FAIL,
    VERDICT_RELEASE_PASS,
    assert_failing_leg,
    assert_failing_leg_prefix,
    build_package_v2,
    build_runs_site,
    cli_args_for,
    cli_args_for_runs_site,
    colm_no_network,  # noqa: F401 - autouse fixture
    expected_tree_sha256,
    gitless_path_dir,
    latest_receipt,
    release_report,
    rewrite_json,
    run_cli,
    sha256_file,
)


# ---------------------------------------------------------------------------
# Baseline: the canonical package reaches PASS_RELEASE end-to-end
# ---------------------------------------------------------------------------


def test_canonical_release_passes_cli_end_to_end(tmp_path):
    pkg = build_package_v2(tmp_path)
    proc = run_cli(*cli_args_for(pkg, "release"))
    assert proc.returncode == EXIT_PASS, proc.stderr
    assert VERDICT_RELEASE_PASS in proc.stdout
    receipt = latest_receipt(pkg.receipts_dir)
    assert receipt["verdict"] == VERDICT_RELEASE_PASS


# ---------------------------------------------------------------------------
# R-013: no self-attestation; containment on resolved symlink-free paths
# ---------------------------------------------------------------------------


class TestIndependentAnchoring:
    def test_release_without_expectations_is_usage_error(self, tmp_path):
        pkg = build_package_v2(tmp_path)
        proc = run_cli(
            "--mode",
            "release",
            "--tree",
            str(pkg.tree),
            "--receipts-dir",
            str(pkg.receipts_dir),
        )
        assert proc.returncode == EXIT_USAGE_ERROR

    def test_expectations_inside_tree_refused(self, tmp_path):
        pkg = build_package_v2(tmp_path)
        inside = pkg.tree / "expectations.json"
        inside.write_bytes(pkg.expectations_path.read_bytes())
        with pytest.raises(verifier.ContainmentError):
            verifier.run_verifier(
                pkg.tree,
                mode="release",
                receipts_dir=pkg.receipts_dir,
                expectations=inside,
            )

    def test_symlinked_expectations_inside_tree_refused(self, tmp_path):
        # Containment uses fully resolved, SYMLINK-FREE paths: a symlink
        # inside the tree pointing at the real out-of-tree expectations is
        # still self-attestation surface and is refused.
        pkg = build_package_v2(tmp_path)
        link = pkg.tree / "exp-link.json"
        link.symlink_to(pkg.expectations_path)
        with pytest.raises(verifier.ContainmentError):
            verifier.run_verifier(
                pkg.tree,
                mode="release",
                receipts_dir=pkg.receipts_dir,
                expectations=link,
            )

    def test_anchor_commit_leg_is_string_exact_without_checkout(
        self, tmp_path
    ):
        # R-013/R-066 split: with git unavailable the STRING-EXACT anchor
        # leg still PASSes while the object-existence leg FAILs.
        pkg = build_package_v2(tmp_path)
        proc = run_cli(
            *cli_args_for(pkg, "release"),
            env_overrides={"PATH": gitless_path_dir()},
        )
        assert proc.returncode == EXIT_GATE_FAIL
        receipt = latest_receipt(pkg.receipts_dir)
        by_id = {leg["leg_id"]: leg for leg in receipt["legs"]}
        assert by_id[LEG_ANCHOR_COMMIT]["status"] == "PASS"
        assert by_id[LEG_GIT_OBJECT]["status"] == "FAIL"

    def test_missing_expectations_grid_block_fails_closed(self, tmp_path):
        # The in-package grid block is never its own release oracle: absent
        # grid pins in the expectations, release FAILs (MISSING_EXPECTATION).
        def mutate(exp):
            del exp["bindings"]["grid"]

        pkg = build_package_v2(tmp_path, expectations_mutator=mutate)
        report = release_report(pkg)
        legs = assert_failing_leg_prefix(report, ANCHORED_GRID_PREFIX)
        assert any(
            leg.get("remediation") == "MISSING_EXPECTATION" for leg in legs
        )

    def test_unknown_anchor_key_is_typed_config_error(self, tmp_path):
        # R-063 (Track A' R1): the anchor block is CLOSED — a typo'd key is
        # a typed error, never a silent default.
        def mutate(exp):
            exp["anchor"]["ledger_pathh"] = "ledger.json"

        pkg = build_package_v2(tmp_path, expectations_mutator=mutate)
        with pytest.raises(schema.ConfigSurfaceError):
            release_report(pkg)

    def test_missing_anchor_required_key_is_typed_config_error(
        self, tmp_path
    ):
        def mutate(exp):
            del exp["anchor"]["ledger_path"]

        pkg = build_package_v2(tmp_path, expectations_mutator=mutate)
        with pytest.raises(schema.ConfigSurfaceError):
            release_report(pkg)


# ---------------------------------------------------------------------------
# R-065: ledger anchored_source_commit == expectations anchor.source_commit
# ---------------------------------------------------------------------------


class TestLedgerAnchorEquality:
    def test_reanchored_ledger_with_rebuilt_digest_still_fails(
        self, tmp_path
    ):
        # Track A F2 shape: re-anchor the ledger to a different commit and
        # REBUILD the expectations ledger hash over the new bytes so the
        # ledger-hash leg passes — the commit-EQUALITY leg must still fire
        # and name both commits.
        other_commit = "9" * 40

        def ledger_mutator(ledger):
            ledger["anchored_source_commit"] = other_commit

        pkg = build_package_v2(tmp_path, ledger_mutator=ledger_mutator)
        # build_package_v2 hashes the mutated ledger into the anchor, so the
        # digest leg is green and ONLY the equality leg separates them.
        assert pkg.expectations["anchor"]["ledger_sha256"] == sha256_file(
            pkg.ledger_path
        )
        report = release_report(pkg)
        leg = assert_failing_leg(report, LEG_LEDGER_ANCHOR_EQ)
        blob = json.dumps(leg)
        assert other_commit in blob
        assert pkg.source_commit in blob


# ---------------------------------------------------------------------------
# R-066: git-object existence is fail-closed at release
# ---------------------------------------------------------------------------


class TestGitObjectLeg:
    def test_git_disappeared_release_fails(self, tmp_path):
        pkg = build_package_v2(tmp_path)
        proc = run_cli(
            *cli_args_for(pkg, "release"),
            env_overrides={"PATH": gitless_path_dir()},
        )
        assert proc.returncode == EXIT_GATE_FAIL
        assert VERDICT_RELEASE_PASS not in proc.stdout

    def test_git_disappeared_source_mode_unaffected(self, tmp_path):
        pkg = build_package_v2(tmp_path)
        proc = run_cli(
            *cli_args_for(pkg, "source"),
            env_overrides={"PATH": gitless_path_dir()},
        )
        assert proc.returncode == EXIT_PASS

    def test_repo_present_object_missing_fails(self, tmp_path):
        # A commit that provably does not exist in this repository.
        pkg = build_package_v2(tmp_path, source_commit="f" * 40)
        report = release_report(pkg)
        assert_failing_leg(report, LEG_GIT_OBJECT)


# ---------------------------------------------------------------------------
# R-021: bindings demonstrably reach the verdict (real CLI, one mutation at
# a time, verdict flips each time, sentinel-free output throughout)
# ---------------------------------------------------------------------------


def _binding_mutations():
    def model_revision(exp):
        exp["bindings"]["model"]["revision"] = "0" * 40

    def split_hash(exp):
        exp["bindings"]["splits"]["eval"]["keyset_sha256"] = "0" * 64

    def producer_hash(exp):
        exp["bindings"]["producer"]["sha256"] = "0" * 64

    def calibration_identity(exp):
        exp["bindings"]["calibration_identity"]["shared"] = "cal-OTHER"

    def ledger_anchor_hash(exp):
        exp["anchor"]["ledger_sha256"] = "0" * 64

    def grid_reference_ids(exp):
        exp["bindings"]["grid"]["reference_ids"] = sorted(
            ["idealized", "kdisjoint", "khard", "klex", "kother"]
        )

    def grid_calibration_ids(exp):
        exp["bindings"]["grid"]["calibration_ids"] = ["pooled", "shared"]

    def grid_cell_ids(exp):
        cells = list(exp["bindings"]["grid"]["cell_ids"])
        cells[0] = "idealized__pooled"
        exp["bindings"]["grid"]["cell_ids"] = cells

    def grid_record_files(exp):
        exp["bindings"]["grid"]["record_files"]["idealized__shared"] = (
            "records/other.jsonl"
        )

    def grid_item_keys(exp):
        exp["bindings"]["grid"]["item_keys_sha256"] = "0" * 64

    def inference_seed(exp):
        exp["bindings"]["inference"]["seed"] += 1

    def inference_matrix_digest(exp):
        exp["bindings"]["inference"]["resample_matrix_digest"]["sha256"] = (
            "0" * 64
        )

    return {
        "model_revision": model_revision,
        "split_hash": split_hash,
        "producer_hash": producer_hash,
        "calibration_identity": calibration_identity,
        "ledger_anchor_hash": ledger_anchor_hash,
        "grid_reference_ids": grid_reference_ids,
        "grid_calibration_ids": grid_calibration_ids,
        "grid_cell_ids": grid_cell_ids,
        "grid_record_files": grid_record_files,
        "grid_item_keys": grid_item_keys,
        "inference_seed": inference_seed,
        "inference_matrix_digest": inference_matrix_digest,
    }


class TestMutationWiring:
    @pytest.mark.parametrize("name", sorted(_binding_mutations()))
    def test_each_expectation_mutation_flips_the_release_verdict(
        self, tmp_path, name
    ):
        pkg = build_package_v2(
            tmp_path, expectations_mutator=_binding_mutations()[name]
        )
        proc = run_cli(*cli_args_for(pkg, "release"))
        assert proc.returncode == EXIT_GATE_FAIL, (
            f"mutation {name} did not flip the verdict; stdout:"
            f" {proc.stdout[-400:]}"
        )
        assert VERDICT_FAIL in proc.stdout
        assert VERDICT_RELEASE_PASS not in proc.stdout

    def test_anchored_inference_seed_leg_fires(self, tmp_path):
        pkg = build_package_v2(
            tmp_path,
            expectations_mutator=_binding_mutations()["inference_seed"],
        )
        report = release_report(pkg)
        assert_failing_leg_prefix(report, ANCHORED_INFERENCE_PREFIX)

    def test_anchored_grid_item_keys_leg_fires(self, tmp_path):
        pkg = build_package_v2(
            tmp_path,
            expectations_mutator=_binding_mutations()["grid_item_keys"],
        )
        report = release_report(pkg)
        assert_failing_leg_prefix(report, ANCHORED_GRID_PREFIX)

    def test_collect_dont_halt_reports_both_broken_legs(self, tmp_path):
        muts = _binding_mutations()

        def both(exp):
            muts["model_revision"](exp)
            muts["split_hash"](exp)

        pkg = build_package_v2(tmp_path, expectations_mutator=both)
        report = release_report(pkg)
        failing = json.dumps(
            [leg for leg in report.legs if leg.get("status") == "FAIL"]
        )
        assert "model" in failing and "split" in failing

    def test_failing_legs_carry_remediation_class_and_both_values(
        self, tmp_path
    ):
        pkg = build_package_v2(
            tmp_path, expectations_mutator=_binding_mutations()["split_hash"]
        )
        report = release_report(pkg)
        failing = [leg for leg in report.legs if leg.get("status") == "FAIL"]
        assert failing
        for leg in failing:
            assert leg.get("remediation") in REMEDIATION_CLASSES
            assert "expected" in leg and "observed" in leg


# ---------------------------------------------------------------------------
# R-012: value-admissibility predicates on artifact-side bindings
# ---------------------------------------------------------------------------


class TestBindingAdmissibility:
    @pytest.mark.parametrize(
        "bad_revision",
        ["abc123", "v1.0", "main", ""],
        ids=["short-sha", "tag", "branch", "empty"],
    )
    def test_mutable_model_revision_forms_rejected(
        self, tmp_path, bad_revision
    ):
        # Artifact-side mutation: expectations mirror the mutated value, so
        # only a VALUE-ADMISSIBILITY predicate can catch it (full-length
        # commit SHA required; short hash/tag/branch are reassignable).
        def mutate(prov):
            prov["model"]["revision"] = bad_revision

        pkg = build_package_v2(tmp_path, binding_mutator=mutate)
        report = release_report(pkg)
        assert report.verdict != VERDICT_RELEASE_PASS

    def test_unresolved_binding_fails(self, tmp_path):
        def mutate(prov):
            prov["continuation_identity"] = "UNRESOLVED"

        pkg = build_package_v2(tmp_path, binding_mutator=mutate)
        report = release_report(pkg)
        assert report.verdict != VERDICT_RELEASE_PASS

    def test_dirty_state_true_fails_release(self, tmp_path):
        def mutate(prov):
            prov["dirty_state"]["git_dirty"] = True

        pkg = build_package_v2(tmp_path, binding_mutator=mutate)
        report = release_report(pkg)
        assert report.verdict != VERDICT_RELEASE_PASS


# ---------------------------------------------------------------------------
# R-035: manifest reconciliation in both directions
# ---------------------------------------------------------------------------


class TestManifestReconciliation:
    def test_declared_but_absent_fails(self, tmp_path):
        def mutate(manifest):
            manifest["artifacts"].append(
                {"path": "ghost.json", "role": "other"}
            )

        pkg = build_package_v2(tmp_path, manifest_mutator=mutate)
        report = release_report(pkg)
        assert report.verdict != VERDICT_RELEASE_PASS

    def test_present_but_undeclared_fails(self, tmp_path):
        pkg = build_package_v2(
            tmp_path,
            extra_tree_files={"stray.json": b'{"unknown_kind": 1}\n'},
            extra_rights_paths=("stray.json",),
            # deliberately NOT in manifest or allowlist
        )
        report = release_report(pkg)
        assert report.verdict != VERDICT_RELEASE_PASS

    def test_allowlisted_undeclared_passes(self, tmp_path):
        pkg = build_package_v2(
            tmp_path,
            extra_tree_files={"stray.json": b'{"unknown_kind": 1}\n'},
            extra_manifest_allowlist=("stray.json",),
            extra_rights_paths=("stray.json",),
        )
        report = release_report(pkg)
        assert report.verdict == VERDICT_RELEASE_PASS

    def test_empty_manifest_artifacts_fails(self, tmp_path):
        # R-033: release requires >= 1 manifest-declared artifact; an empty
        # presentation manifest can never certify.
        def mutate(manifest):
            manifest["artifacts"] = []

        pkg = build_package_v2(tmp_path, manifest_mutator=mutate)
        try:
            report = release_report(pkg)
            assert report.verdict != VERDICT_RELEASE_PASS
        except schema.ColmAimsError:
            pass  # a typed refusal is equally fail-closed


# ---------------------------------------------------------------------------
# R-033: vacuous inputs are typed errors / failing runs
# ---------------------------------------------------------------------------


class TestVacuousInputs:
    def test_empty_tree_is_typed_error_naming_path_and_layout(
        self, tmp_path
    ):
        empty = tmp_path / "empty-tree"
        empty.mkdir()
        receipts = tmp_path / "receipts"
        receipts.mkdir()
        proc = run_cli(
            "--mode",
            "source",
            "--tree",
            str(empty),
            "--receipts-dir",
            str(receipts),
        )
        assert proc.returncode == EXIT_INGRESS_ERROR
        assert "empty-tree" in proc.stderr
        assert "layout" in proc.stderr.lower()

    def test_zero_retained_claim_rows_fails_release(self, tmp_path):
        def mutate(ledger):
            ledger["rows"] = []

        pkg = build_package_v2(tmp_path, ledger_mutator=mutate)
        try:
            report = release_report(pkg)
            assert report.verdict != VERDICT_RELEASE_PASS
        except schema.ColmAimsError:
            pass  # a typed refusal is equally fail-closed


# ---------------------------------------------------------------------------
# R-069 / R-039: canonical selection wired into the ACTUAL release path
# ---------------------------------------------------------------------------


class TestCanonicalSelection:
    def test_parse_error_cannot_write_receipt_beneath_runs_root(self, tmp_path):
        runs_root = tmp_path / "runs"
        receipts = runs_root / "receipts"
        runs_root.mkdir()

        proc = run_cli(
            "--mode",
            "release",
            "--runs-root",
            str(runs_root),
            "--receipts-dir",
            str(receipts),
            "--unknown-release-option",
        )

        assert proc.returncode == EXIT_USAGE_ERROR
        assert not list(receipts.glob("receipt-*.json"))

    def test_parse_error_still_receipts_outside_runs_root(self, tmp_path):
        runs_root = tmp_path / "runs"
        receipts = tmp_path / "receipts"
        runs_root.mkdir()

        proc = run_cli(
            "--mode",
            "release",
            "--runs-root",
            str(runs_root),
            "--receipts-dir",
            str(receipts),
            "--unknown-release-option",
        )

        assert proc.returncode == EXIT_USAGE_ERROR
        receipt = latest_receipt(receipts)
        assert receipt["mode"] == "release"
        assert receipt["verdict"] == VERDICT_FAIL

    def test_runs_site_baseline_passes_through_pointer(self, tmp_path):
        site = build_runs_site(tmp_path)
        proc = run_cli(*cli_args_for_runs_site(site))
        assert proc.returncode == EXIT_PASS, proc.stderr
        # The receipt binds the POINTED run's tree (not any other).
        receipt = latest_receipt(site.receipts_dir)
        assert receipt["input_tree_sha256"] == expected_tree_sha256(
            site.run_tree
        )

    @pytest.mark.parametrize(
        "receipt_location",
        ["runs_root", "selected_closure", "other_tree", "future_slot"],
    )
    def test_success_cannot_write_receipts_anywhere_beneath_runs_root(
        self, tmp_path, receipt_location
    ):
        site = build_runs_site(tmp_path, extra_runs=("run-other",))
        locations = {
            "runs_root": site.runs_root,
            "selected_closure": site.run_tree / "closure",
            "other_tree": site.runs_root / "run-other" / "tree",
            "future_slot": site.runs_root / "run-future" / "receipts",
        }
        receipts = locations[receipt_location]
        before = {
            path.relative_to(site.runs_root).as_posix(): path.read_bytes()
            for path in site.runs_root.rglob("*")
            if path.is_file()
        }
        args = cli_args_for_runs_site(site)
        args[-1] = str(receipts)

        proc = run_cli(*args)

        assert proc.returncode == EXIT_USAGE_ERROR
        assert "outside the entire canonical runs root" in (
            proc.stdout + proc.stderr
        )
        after = {
            path.relative_to(site.runs_root).as_posix(): path.read_bytes()
            for path in site.runs_root.rglob("*")
            if path.is_file()
        }
        assert after == before

    def test_symlinked_run_dir_refused_even_in_root(self, tmp_path):
        site = build_runs_site(tmp_path, run_id="run-real")
        link = site.runs_root / "run-0001"
        link.symlink_to(site.runs_root / "run-real")

        def repoint(ledger):
            ledger["canonical_run_id"] = "run-0001"

        rewrite_json(site.ledger_path, repoint)
        rewrite_json(
            site.expectations_path,
            lambda exp: exp["anchor"].update(
                ledger_sha256=sha256_file(site.ledger_path)
            ),
        )
        proc = run_cli(*cli_args_for_runs_site(site))
        assert proc.returncode == EXIT_GATE_FAIL
        assert "run-0001" in proc.stdout + proc.stderr

    def test_escaping_pointer_refused(self, tmp_path):
        outside = tmp_path / "outside-run" / "tree"
        outside.mkdir(parents=True)
        site = build_runs_site(tmp_path, canonical_pointer="../outside-run")
        proc = run_cli(*cli_args_for_runs_site(site))
        assert proc.returncode == EXIT_GATE_FAIL

    def test_empty_crash_relic_refused(self, tmp_path):
        site = build_runs_site(
            tmp_path, canonical_pointer="run-empty", extra_runs=()
        )
        (site.runs_root / "run-empty").mkdir()
        proc = run_cli(*cli_args_for_runs_site(site))
        assert proc.returncode == EXIT_GATE_FAIL

    def test_dangling_pointer_refused(self, tmp_path):
        site = build_runs_site(tmp_path, canonical_pointer="run-9999")
        proc = run_cli(*cli_args_for_runs_site(site))
        assert proc.returncode == EXIT_GATE_FAIL

    def test_never_newest_wins_pointer_selects_older_run(self, tmp_path):
        # A NEWER decoy run exists; the pointer names the older valid run.
        # The release must verify the POINTED run (and pass), proving
        # newest-wins is not the selection rule.
        site = build_runs_site(tmp_path, extra_runs=("run-9999-newer",))
        (site.runs_root / "run-9999-newer" / "tree" / "profile.json").write_bytes(
            b'{"schema_version": 2}\n'
        )
        proc = run_cli(*cli_args_for_runs_site(site))
        assert proc.returncode == EXIT_PASS, proc.stderr
        receipt = latest_receipt(site.receipts_dir)
        assert receipt["input_tree_sha256"] == expected_tree_sha256(
            site.run_tree
        )

    def test_never_falls_back_after_invalid_pointer(self, tmp_path):
        # The pointer is dangling while a perfectly valid package sits in
        # another run dir: the release must FAIL, never fall back.
        site = build_runs_site(tmp_path, canonical_pointer="run-missing")
        proc = run_cli(*cli_args_for_runs_site(site))
        assert proc.returncode == EXIT_GATE_FAIL

    def test_missing_pointer_refused(self, tmp_path):
        site = build_runs_site(tmp_path)

        def drop_pointer(ledger):
            del ledger["canonical_run_id"]

        rewrite_json(site.ledger_path, drop_pointer)
        rewrite_json(
            site.expectations_path,
            lambda exp: exp["anchor"].update(
                ledger_sha256=sha256_file(site.ledger_path)
            ),
        )
        proc = run_cli(*cli_args_for_runs_site(site))
        assert proc.returncode == EXIT_GATE_FAIL

    def test_refusals_surface_as_canonical_selection_leg(self, tmp_path):
        site = build_runs_site(tmp_path, canonical_pointer="run-9999")
        proc = run_cli(*cli_args_for_runs_site(site))
        assert proc.returncode == EXIT_GATE_FAIL
        receipt = latest_receipt(site.receipts_dir)
        failing = [
            leg["leg_id"]
            for leg in receipt["legs"]
            if leg.get("status") == "FAIL"
        ]
        assert LEG_CANONICAL_SELECTION in failing

    def test_tree_and_runs_root_flags_are_mutually_exclusive(self, tmp_path):
        pkg = build_package_v2(tmp_path)
        proc = run_cli(
            "--mode",
            "release",
            "--tree",
            str(pkg.tree),
            "--runs-root",
            str(tmp_path / "runs"),
            "--expectations",
            str(pkg.expectations_path),
            "--receipts-dir",
            str(pkg.receipts_dir),
        )
        assert proc.returncode == EXIT_USAGE_ERROR

    def test_resolve_canonical_package_unit_contract(self, tmp_path):
        # R-039 unit surface (kept from v1): pointer must be a single path
        # component naming a real, non-empty, in-root, non-symlink dir.
        runs_root = tmp_path / "runs"
        (runs_root / "run-0001").mkdir(parents=True)
        (runs_root / "run-0001" / "tree").mkdir()
        ledger = {"canonical_run_id": "run-0001"}
        resolved = verifier.resolve_canonical_package(runs_root, ledger)
        assert resolved == runs_root / "run-0001"
        with pytest.raises(schema.ColmAimsError):
            verifier.resolve_canonical_package(
                runs_root, {"canonical_run_id": "../run-0001"}
            )
        with pytest.raises(schema.ColmAimsError):
            verifier.resolve_canonical_package(runs_root, {})

    def test_retired_run_bytes_are_retained_unchanged(self, tmp_path):
        # R-039: retiring == ledger pointer change + new run dir; the old
        # run's bytes stay byte-identical after a release over the new one.
        site = build_runs_site(tmp_path, extra_runs=("run-0000-old",))
        old_file = site.runs_root / "run-0000-old" / "tree" / "old.json"
        old_file.write_bytes(b'{"schema_version": 2, "historical": true}\n')
        before = old_file.read_bytes()
        run_cli(*cli_args_for_runs_site(site))
        assert old_file.read_bytes() == before


# ---------------------------------------------------------------------------
# R-014: HISTORICAL_NONCERTIFYING classification is surfaced
# ---------------------------------------------------------------------------


def test_release_verdict_tokens_are_the_closed_release_enum():
    assert verifier.RELEASE_MODE_VERDICTS == frozenset(
        {VERDICT_RELEASE_PASS, VERDICT_FAIL}
    )
