"""RED suite — verifier CLI wiring, exit codes, fail-closed flags, namespace.

Covers: R-018, R-021, R-022, R-037.
Spec: .correctless/specs/camera-ready-aims-evidence.md

R-021 runs the DOCUMENTED command line via subprocess (never an imported
main()): `python -m reproducibility.colm_aims_2026.verify` from the repo root.
# No documented entrypoint in ARCHITECTURE.md — the CLI itself is the
# inferred entry point (spec OQ-003).
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests._colm_aims_helpers import (  # noqa: F401  (colm_no_network is an autouse fixture)
    EXIT_GATE_FAIL,
    EXIT_INGRESS_ERROR,
    EXIT_PASS,
    EXIT_USAGE_ERROR,
    REPO_ROOT,
    SENTINEL,
    VERDICT_RELEASE_PASS,
    VERDICT_SOURCE_PASS,
    build_package,
    cli_args_for,
    colm_no_network,
    run_cli,
)

NAMESPACE_DIR = REPO_ROOT / "reproducibility" / "colm_aims_2026"


def _assert_sentinel_free(proc: subprocess.CompletedProcess, pkg=None) -> None:
    # R-026 leak surface asserted on every CLI run (R-021).
    assert SENTINEL not in proc.stdout
    assert SENTINEL not in proc.stderr
    if pkg is not None:
        for receipt in pkg.receipts_dir.glob("**/*"):
            if receipt.is_file():
                assert SENTINEL not in receipt.read_text("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# R-021 + R-037: end-to-end wiring over a complete tiny fixture package
# ---------------------------------------------------------------------------


def test_cli_source_mode_passes_on_pristine_package(tmp_path: Path):
    # Tests R-021 [integration]: the real CLI over the complete tiny package,
    # source mode: exit code EXIT_PASS and the ceiling verdict on stdout.
    pkg = build_package(tmp_path)
    proc = run_cli(*cli_args_for(pkg, "source"))
    assert proc.returncode == EXIT_PASS, proc.stderr
    assert VERDICT_SOURCE_PASS in proc.stdout
    _assert_sentinel_free(proc, pkg)


def test_cli_release_mode_passes_on_pristine_package(tmp_path: Path):
    # Tests R-021 [integration]: release mode over the pristine package
    # reaches the release PASS verdict with exit code EXIT_PASS.
    pkg = build_package(tmp_path)
    proc = run_cli(*cli_args_for(pkg, "release"))
    assert proc.returncode == EXIT_PASS, proc.stderr
    assert VERDICT_RELEASE_PASS in proc.stdout
    _assert_sentinel_free(proc, pkg)


SINGLE_BINDING_MUTATIONS = [
    (
        "model_revision",
        lambda exp: exp["bindings"]["model"].update(revision="9" * 40),
    ),
    (
        "split_hash",
        lambda exp: exp["bindings"]["splits"]["eval"].update(
            keyset_sha256="1" * 64
        ),
    ),
    (
        "producer_hash",
        lambda exp: exp["bindings"]["producer"].update(sha256="2" * 64),
    ),
    (
        "calibration_identity",
        lambda exp: exp["bindings"].update(calibration_identity="cal-9999"),
    ),
    (
        "ledger_anchor",
        lambda exp: exp["anchor"].update(ledger_sha256="0" * 64),
    ),
]


@pytest.mark.parametrize(
    "name,mutator", SINGLE_BINDING_MUTATIONS, ids=[n for n, _ in SINGLE_BINDING_MUTATIONS]
)
def test_cli_single_binding_mutation_flips_verdict(tmp_path: Path, name, mutator):
    # Tests R-021 [integration]: bindings demonstrably reach the verdict —
    # mutating one expectation at a time flips the verdict, with the pinned
    # gate-FAIL exit code and sentinel-free output.
    pkg = build_package(tmp_path, expectations_mutator=mutator)
    proc = run_cli(*cli_args_for(pkg, "release"))
    assert proc.returncode == EXIT_GATE_FAIL, (proc.returncode, proc.stderr)
    assert "FAIL" in proc.stdout
    assert VERDICT_RELEASE_PASS not in proc.stdout
    _assert_sentinel_free(proc, pkg)


def test_cli_output_references_tree_files_relatively(tmp_path: Path):
    # Tests R-021/R-020 [integration]: per-file diagnostics identify files by
    # tree-relative path — the absolute tmp prefix of files INSIDE the tree
    # never appears in output (R-026 accepts only the relative form).
    pkg = build_package(
        tmp_path,
        expectations_mutator=lambda exp: exp["bindings"]["input_hashes"].update(
            **{"records.jsonl": "3" * 64}
        ),
    )
    proc = run_cli(*cli_args_for(pkg, "release"))
    assert proc.returncode == EXIT_GATE_FAIL
    assert "records.jsonl" in proc.stdout + proc.stderr
    assert str(pkg.tree / "records.jsonl") not in proc.stdout + proc.stderr


# ---------------------------------------------------------------------------
# R-037: exit-code contract + invocation forms
# ---------------------------------------------------------------------------


def test_exit_code_constants_pinned_and_distinct():
    # Tests R-037 [integration]: distinct exit codes are pinned for
    # mode-ceiling pass, gate FAIL, usage/config error, typed ingress error.
    # DECISION: 0 / 1 / 2 / 3 respectively.
    from reproducibility.colm_aims_2026 import verify

    codes = {
        verify.EXIT_PASS,
        verify.EXIT_GATE_FAIL,
        verify.EXIT_USAGE_ERROR,
        verify.EXIT_INGRESS_ERROR,
    }
    assert verify.EXIT_PASS == EXIT_PASS
    assert verify.EXIT_GATE_FAIL == EXIT_GATE_FAIL
    assert verify.EXIT_USAGE_ERROR == EXIT_USAGE_ERROR
    assert verify.EXIT_INGRESS_ERROR == EXIT_INGRESS_ERROR
    assert len(codes) == 4


def test_cli_typed_ingress_error_exit_code(tmp_path: Path):
    # Tests R-037/R-020 [integration]: malformed artifact bytes exit with the
    # typed-ingress code, naming the file relatively.
    pkg = build_package(tmp_path)
    pkg.profile_path.write_bytes(b'{"schema_version": 1, "trunc')
    proc = run_cli(*cli_args_for(pkg, "source"))
    assert proc.returncode == EXIT_INGRESS_ERROR, (proc.returncode, proc.stderr)
    out = proc.stdout + proc.stderr
    assert "profile.json" in out
    assert str(pkg.tree / "profile.json") not in out


def test_cli_empty_evaluation_exit_code_and_no_receipt(tmp_path: Path):
    # Tests R-037/R-012 [integration]: the empty-evaluation refusal errors
    # before any report is emitted — ingress-class exit code, no receipt.
    from tests.test_colm_aims_verifier_gates import _build_empty_eval_package

    pkg = _build_empty_eval_package(tmp_path)
    proc = run_cli(*cli_args_for(pkg, "release"))
    assert proc.returncode == EXIT_INGRESS_ERROR, (proc.returncode, proc.stderr)
    assert list(pkg.receipts_dir.iterdir()) == []


def test_cli_vacuous_tree_exit_code(tmp_path: Path):
    # Tests R-037/R-033 [integration]: a zero-artifact tree is a typed error
    # with the ingress exit code, naming the resolved path and expected layout.
    pkg = build_package(tmp_path)
    for p in list(pkg.tree.iterdir()):
        p.unlink()
    proc = run_cli(*cli_args_for(pkg, "source"))
    assert proc.returncode == EXIT_INGRESS_ERROR, (proc.returncode, proc.stderr)
    out = proc.stdout + proc.stderr
    assert "profile.json" in out  # expected layout named


def test_direct_path_invocation_bootstraps_or_names_module_form(tmp_path: Path):
    # Tests R-037 [integration]: direct-path invocation either bootstraps
    # sys.path (repo convention) and behaves identically, or errors naming
    # the documented module-run form.
    pkg = build_package(tmp_path)
    proc = subprocess.run(
        [
            sys.executable,
            str(NAMESPACE_DIR / "verify.py"),
            *cli_args_for(pkg, "source"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    worked = proc.returncode == EXIT_PASS and VERDICT_SOURCE_PASS in proc.stdout
    named_module_form = "python -m reproducibility.colm_aims_2026.verify" in (
        proc.stdout + proc.stderr
    )
    assert worked or named_module_form, (proc.returncode, proc.stderr[-500:])


# ---------------------------------------------------------------------------
# R-022: fail closed on unknown flags/keys; no bypass doors
# ---------------------------------------------------------------------------


def test_unknown_flag_is_usage_error_not_noop(tmp_path: Path):
    # Tests R-022 [unit]: unknown flags error with the usage exit code.
    pkg = build_package(tmp_path)
    proc = run_cli(*cli_args_for(pkg, "source"), "--frobnicate")
    assert proc.returncode == EXIT_USAGE_ERROR, (proc.returncode, proc.stderr)
    assert "frobnicate" in (proc.stdout + proc.stderr)


@pytest.mark.parametrize(
    "door",
    [
        "--skip-rights",
        "--allow-missing-rights",
        "--force",
        "--skip-gates",
        "--no-fail-closed",
        "--allow-dirty",
    ],
)
def test_no_flag_door_disables_a_release_gate(tmp_path: Path, door):
    # Tests R-022 [unit]: no flag door disables a release gate — plausible
    # bypass flags must not exist (usage error, not a weakened run).
    pkg = build_package(
        tmp_path,
        rights_mutator=lambda r: r["paths"][0].update(status="UNVERIFIED"),
    )
    proc = run_cli(*cli_args_for(pkg, "release"), door)
    assert proc.returncode == EXIT_USAGE_ERROR, (door, proc.returncode)


def test_abbreviated_flags_rejected_allow_abbrev_false(tmp_path: Path):
    # Tests R-022 [unit]: allow_abbrev=False — abbreviated flag forms cannot
    # smuggle past the unknown-flag check.
    pkg = build_package(tmp_path)
    proc = run_cli(
        "--mod",
        "source",
        "--tree",
        str(pkg.tree),
        "--receipts-dir",
        str(pkg.receipts_dir),
    )
    assert proc.returncode == EXIT_USAGE_ERROR, (proc.returncode, proc.stderr)


def test_no_environment_door_disables_a_release_gate(tmp_path: Path):
    # Tests R-022 [unit]: no environment variable disables a release gate.
    pkg = build_package(
        tmp_path,
        rights_mutator=lambda r: r["paths"][0].update(status="UNVERIFIED"),
    )
    env = dict(os.environ)
    env.update(
        COLM_AIMS_SKIP_GATES="1",
        COLM_AIMS_ALLOW_RELEASE="1",
        COLM_AIMS_FORCE_PASS="1",
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "reproducibility.colm_aims_2026.verify",
            *cli_args_for(pkg, "release"),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == EXIT_GATE_FAIL, (proc.returncode, proc.stderr)
    assert "FAIL" in proc.stdout  # a real gate-FAIL verdict, not a crash
    assert VERDICT_RELEASE_PASS not in proc.stdout


def test_unknown_expectations_key_fails_closed(tmp_path: Path):
    # Tests R-022/R-020 [unit]: an unknown key in the config surface
    # (expectations file) errors — it is never a silent no-op.
    pkg = build_package(
        tmp_path,
        expectations_mutator=lambda exp: exp.update(bypass_gates=True),
    )
    proc = run_cli(*cli_args_for(pkg, "release"))
    assert proc.returncode in (EXIT_GATE_FAIL, EXIT_INGRESS_ERROR)
    assert proc.returncode != EXIT_PASS
    assert "bypass_gates" in (proc.stdout + proc.stderr)


# ---------------------------------------------------------------------------
# R-018: namespace isolation
# ---------------------------------------------------------------------------

# Source: shasum -a 256 scripts/verify_audit_release.py at the RED baseline
# (2026-08-19, branch feature/camera-ready-aims-spec). R-018 pins this file
# byte-identical for the whole feature.
VERIFY_AUDIT_RELEASE_SHA256 = (
    "8d4e76c5e183e6efb96844ac13b55dd3fbaa1eab64b9da74fe611466f456513a"
)


def test_legacy_verifier_script_stays_byte_identical():
    # Tests R-018 [integration]: scripts/verify_audit_release.py stays
    # byte-identical while this feature lands.
    data = (REPO_ROOT / "scripts" / "verify_audit_release.py").read_bytes()
    assert hashlib.sha256(data).hexdigest() == VERIFY_AUDIT_RELEASE_SHA256


def test_namespace_does_not_touch_legacy_verifier():
    # Tests R-018 [integration]: nothing under reproducibility/colm_aims_2026
    # imports the legacy verifier, and the legacy verifier does not import
    # the new namespace.
    for py in sorted(NAMESPACE_DIR.glob("*.py")):
        tree = ast.parse(py.read_text("utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for name in names:
                assert not name.startswith("scripts.verify_audit_release"), py

    legacy = ast.parse(
        (REPO_ROOT / "scripts" / "verify_audit_release.py").read_text("utf-8")
    )
    for node in ast.walk(legacy):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        for name in names:
            assert not name.startswith("reproducibility")


def test_new_code_lives_only_under_the_new_namespace():
    # Tests R-018 [integration]: the feature's production modules exist under
    # reproducibility/colm_aims_2026/ (structural floor; the containment
    # tests below are the enforcement — audit BL-1).
    expected = {
        "__init__.py",
        "schema.py",
        "pairing.py",
        "verifier.py",
        "ledger.py",
        "receipt.py",
        "render.py",
        "verify.py",
    }
    present = {p.name for p in NAMESPACE_DIR.glob("*.py")}
    assert expected <= present


# R-018 containment (audit BL-1). Everything this feature adds must live in
# these locations:
ALLOWED_FEATURE_PREFIXES = (
    "reproducibility/colm_aims_2026/",
    "tests/",
    "docs/",
    ".correctless/",
)

# Exact-path allowances (audit item 5): deliberately NOT a blanket
# reproducibility/ prefix — only the namespace package marker and the one
# tracked doc R-038 REQUIRES GREEN to edit (historical-scope header on
# reproducibility/source_to_claim.md).
ALLOWED_EXACT_FILES = frozenset(
    {
        "reproducibility/__init__.py",
        "reproducibility/source_to_claim.md",
    }
)


def _is_allowed_feature_path(path: str) -> bool:
    return path in ALLOWED_EXACT_FILES or path.startswith(ALLOWED_FEATURE_PREFIXES)

# Pre-existing repo state at the RED baseline, unrelated to this feature —
# frozen exemptions so the containment tests bite only on NEW out-of-scope
# paths. Source: `git status --porcelain` on feature/camera-ready-aims-spec
# at commit ae0e2487 (2026-08-19): untracked hazard-efficacy results,
# planning reviews, and the handoff prompt predate this feature.
PREEXISTING_WORKTREE_PREFIXES = (
    ".planning/",
    "results/",
    "handoff_prompt_camera_ready_2026-08-18.md",
)

# Source: `git diff --name-only $(git merge-base HEAD main) HEAD` at the RED
# baseline (41 paths) — the branch carries earlier hazard-efficacy commits.
PREEXISTING_COMMITTED_PATHS = frozenset(
    {
        ".correctless/AGENT_CONTEXT.md",
        ".correctless/hooks/workflow-advance.sh",
        ".correctless/specs/hazard-efficacy-eval.md",
        ".correctless/specs/hazard-pretrain-bridge.md",
        ".correctless/verification/hazard-efficacy-eval-verification.md",
        ".correctless/verification/hazard-pretrain-bridge-verification.md",
        ".planning/hazard-eval-harness-spec.md",
        ".planning/hazard-pretrain-handoff.md",
        "AGENTS.md",
        "configs/t5_policy_base_prelim.yaml",
        "docs/dev-journal.md",
        "docs/hazard-efficacy-report.md",
        "docs/workflow-history.md",
        "results/hazard_efficacy_base/hazard_efficacy_plot.png",
        "results/hazard_efficacy_base/hazard_efficacy_report.json",
        "results/hazard_efficacy_smoke/hazard_efficacy_plot.png",
        "results/hazard_efficacy_smoke/hazard_efficacy_report.json",
        "scripts/_common.py",
        "scripts/compare_policies.py",
        "scripts/run_hazard_efficacy.py",
        "scripts/train_t5_policy.py",
        "tests/_hazard_efficacy_fixtures.py",
        "tests/test_bootstrap_ci_validation.py",
        "tests/test_csli_model_list_guard.py",
        "tests/test_csli_thread_safe_caches.py",
        "tests/test_evaluate_t5_runs_e2.py",
        "tests/test_hazard_ablation_dynamics.py",
        "tests/test_hazard_efficacy_harness.py",
        "tests/test_hazard_efficacy_mini_audit.py",
        "tests/test_hazard_efficacy_orchestration.py",
        "tests/test_hazard_efficacy_probe.py",
        "tests/test_hazard_pretrain.py",
        "tests/test_ppo_t5.py",
        "tests/test_pr14_review_regressions.py",
        "tests/test_stopdff_v5_package.py",
        "tests/test_stopdff_value_model.py",
        "tests/test_train_seed_e1.py",
        "tests/test_train_t5_policy_script.py",
        "training/hazard_pretrain.py",
        "training/train_ppo_t5.py",
        "training/train_supervised_t5.py",
    }
)


def _git_lines(*args: str) -> list[str]:
    proc = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    return [ln for ln in proc.stdout.splitlines() if ln.strip()]


def _in_allowed_or(path: str, exemptions) -> bool:
    return _is_allowed_feature_path(path) or path.startswith(tuple(exemptions))


def test_feature_worktree_paths_contained_to_namespace():
    # Tests R-018 [integration]: every changed/untracked working-tree path
    # falls under the namespace, tests/, docs/, or .correctless/ (plus the
    # frozen pre-existing baseline). Passes at RED by design and bites the
    # moment GREEN (or any later phase) writes outside the allowed set.
    offenders = []
    for line in _git_lines("status", "--porcelain"):
        path = line[3:]
        if " -> " in path:  # rename records "old -> new"; the NEW path counts
            path = path.split(" -> ", 1)[1]
        if not _in_allowed_or(path, PREEXISTING_WORKTREE_PREFIXES):
            offenders.append(path)
    assert not offenders, (
        f"feature work leaked outside the allowed locations: {offenders}"
    )


def test_feature_committed_paths_contained_to_namespace():
    # Tests R-018 [integration]: any path committed on this branch beyond the
    # frozen RED-baseline set falls under the allowed locations. Passes at
    # RED by design (nothing committed yet); bites on out-of-scope commits.
    base = _git_lines("merge-base", "HEAD", "main")[0]
    changed = _git_lines("diff", "--name-only", base, "HEAD")
    offenders = [
        p
        for p in changed
        if p not in PREEXISTING_COMMITTED_PATHS and not _is_allowed_feature_path(p)
    ]
    assert not offenders, (
        f"out-of-namespace paths committed by this feature: {offenders}"
    )


# Audit item 4: pre-existing untracked .py files under the exempt trees at
# the RED baseline. Source: `git status --porcelain -uall` filtered to
# ^(.planning|results)/.*\.py$ on feature/camera-ready-aims-spec, 2026-08-19.
PREEXISTING_EXEMPT_TREE_PY = frozenset(
    {
        ".planning/reviews/_tools/assemble_primer.py",
        ".planning/reviews/_tools/synthesis_join.py",
    }
)


def test_no_new_python_files_under_exempt_trees():
    # Tests R-018 [integration] (audit item 4): the .planning/ and results/
    # worktree exemptions must never become a side door for CODE — any .py
    # file under them beyond the frozen 2-path baseline fails. Uses -uall so
    # files inside untracked directories are enumerated individually.
    # Passes at RED by design; bites the moment code lands there.
    offenders = []
    for line in _git_lines("status", "--porcelain", "-uall"):
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        if (
            path.endswith(".py")
            and path.startswith((".planning/", "results/"))
            and path not in PREEXISTING_EXEMPT_TREE_PY
        ):
            offenders.append(path)
    assert not offenders, (
        f"new python files under exempt worktree trees: {offenders}"
    )


def test_no_module_outside_namespace_imports_the_namespace():
    # Tests R-018 [integration]: no production module outside
    # reproducibility/colm_aims_2026/ imports it (tests/ legitimately do).
    # Passes at RED by design; bites if GREEN wires the namespace into the
    # legacy pipeline.
    scan_roots = [
        REPO_ROOT / d
        for d in (
            "agents",
            "evaluation",
            "models",
            "qb_data",
            "qb_env",
            "training",
            "scripts",
            "schemas",
        )
    ]
    candidates = [p for p in REPO_ROOT.glob("*.py")]
    for root in scan_roots:
        if root.is_dir():
            candidates.extend(root.rglob("*.py"))
    offenders = []
    for py in candidates:
        if "__pycache__" in py.parts:
            continue
        text = py.read_text("utf-8", errors="replace")
        if "colm_aims_2026" not in text:
            continue  # cheap pre-filter
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            if any(n.startswith("reproducibility.colm_aims_2026") for n in names):
                offenders.append(str(py.relative_to(REPO_ROOT)))
    assert not offenders, (
        f"modules outside the namespace import it: {offenders}"
    )


def test_cli_module_run_from_repo_root_is_the_documented_form(tmp_path: Path):
    # Tests R-037/R-021 [integration]: the documented invocation
    # `python -m reproducibility.colm_aims_2026.verify` from the repo root is
    # what this suite runs; a JSON-parseable verdict token must appear on
    # stdout (sentinel-free machine-readable surface).
    pkg = build_package(tmp_path)
    proc = run_cli(*cli_args_for(pkg, "source"))
    assert proc.returncode == EXIT_PASS
    assert VERDICT_SOURCE_PASS in proc.stdout
    # Receipt emitted for the run (R-036 wiring through the CLI).
    receipts = [p for p in pkg.receipts_dir.glob("**/*") if p.is_file()]
    assert receipts, "CLI run emitted no receipt"
    payload = json.loads(receipts[0].read_text("utf-8"))
    assert payload["mode"] == "source"
