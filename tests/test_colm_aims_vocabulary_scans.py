"""RED suite — vocabulary discipline, no-network guard, deserialization scan.

Covers: R-027, R-028, R-034.
Spec: .correctless/specs/camera-ready-aims-evidence.md

Note: the static scans (R-028 deny-list, R-034 deserialization) hold
vacuously over the RED stubs by design — they are prohibition invariants
that must keep holding when GREEN fills the namespace in.
"""
from __future__ import annotations

import ast
import re
import socket
from pathlib import Path

import pytest

from reproducibility.colm_aims_2026 import render, verifier
from tests._colm_aims_helpers import (  # noqa: F401  (colm_no_network is an autouse fixture)
    FIXTURES_DIR,
    REPO_ROOT,
    VERDICT_FAIL,
    VERDICT_RELEASE_PASS,
    VERDICT_SOURCE_PASS,
    build_package,
    colm_no_network,
    load_vocabulary,
)

NAMESPACE_DIR = REPO_ROOT / "reproducibility" / "colm_aims_2026"


# ---------------------------------------------------------------------------
# R-027: banned-phrase + required-qualifier vocabulary
# ---------------------------------------------------------------------------


def _assert_vocabulary_clean(text: str, *, require_qualifiers: bool) -> None:
    vocab = load_vocabulary()
    low = text.lower()
    scrubbed = low
    for exception in vocab["allowed_exceptions"]:
        scrubbed = scrubbed.replace(exception, "")
    for phrase in vocab["banned_phrases"]:
        assert phrase not in scrubbed, f"banned phrase {phrase!r} in output"
    for term in vocab["banned_phrases_case_sensitive"]:
        assert re.search(rf"\b{re.escape(term)}\b", text) is None, (
            f"banned ACM third-party term {term!r} in output"
        )
    if require_qualifiers:
        for qualifier in vocab["required_qualifiers"]:
            assert qualifier in low, f"required qualifier {qualifier!r} missing"


def test_vocabulary_fixture_is_enumerated_and_pins_core_entries():
    # Tests R-027 [unit]: the enforceable core is an enumerated banned-phrase
    # and required-qualifier list maintained as a fixture file.
    vocab = load_vocabulary()
    assert vocab["banned_phrases"], "banned phrase list must be non-empty"
    assert "qa effect" in vocab["banned_phrases"]  # unqualified "QA effect"
    assert "would hide real shifts" in vocab["banned_phrases"]
    assert "constructed qa reference" in vocab["required_qualifiers"]
    assert (
        vocab["sanctioned_observed_claim_output"]
        == "observed_paired_claim=OBSERVED_PAIRED_STUDY_REQUIRED"
    )


def test_vocabulary_scanner_actually_bites():
    # Tests R-027 [unit]: self-test of the scan tooling — an overclaiming
    # output is caught, so a passing scan is meaningful.
    with pytest.raises(AssertionError):
        _assert_vocabulary_clean(
            "The measured QA effect shows the observed stopping policy is preserved.",
            require_qualifiers=False,
        )
    with pytest.raises(AssertionError):
        _assert_vocabulary_clean(
            "Result Reproduced by the authors.", require_qualifiers=False
        )


def test_source_mode_summary_respects_vocabulary(tmp_path: Path):
    # Tests R-027 [unit]: the vocabulary is asserted over every renderer
    # output — source-mode summary carries the required "constructed QA
    # reference" qualification and no banned phrase.
    pkg = build_package(tmp_path)
    report = verifier.run_verifier(
        pkg.tree, mode="source", receipts_dir=pkg.receipts_dir
    )
    assert report.verdict == VERDICT_SOURCE_PASS
    _assert_vocabulary_clean(render.render_summary(report), require_qualifiers=True)


def test_release_mode_summary_respects_vocabulary(tmp_path: Path):
    # Tests R-027 [unit]: release-mode pass summary is vocabulary-clean.
    pkg = build_package(tmp_path)
    report = verifier.run_verifier(
        pkg.tree,
        mode="release",
        receipts_dir=pkg.receipts_dir,
        expectations=pkg.expectations_path,
    )
    assert report.verdict == VERDICT_RELEASE_PASS
    _assert_vocabulary_clean(render.render_summary(report), require_qualifiers=True)


def test_failing_summary_respects_vocabulary(tmp_path: Path):
    # Tests R-027 [unit]: failure summaries are renderer outputs too — the
    # vocabulary discipline applies to them.
    pkg = build_package(
        tmp_path,
        expectations_mutator=lambda exp: exp["bindings"]["model"].update(
            revision="9" * 40
        ),
    )
    report = verifier.run_verifier(
        pkg.tree,
        mode="release",
        receipts_dir=pkg.receipts_dir,
        expectations=pkg.expectations_path,
    )
    assert report.verdict == VERDICT_FAIL
    _assert_vocabulary_clean(render.render_summary(report), require_qualifiers=True)


def test_namespace_docs_respect_vocabulary():
    # Tests R-027 [unit]: no doc produced by this feature asserts observed
    # decision preservation or drops the qualification. Docs existence is
    # R-038's job; every doc that exists must scan clean and the README must
    # carry the qualifier.
    for md in sorted(NAMESPACE_DIR.glob("**/*.md")):
        _assert_vocabulary_clean(
            md.read_text("utf-8"),
            require_qualifiers=(md.name == "README.md"),
        )


# ---------------------------------------------------------------------------
# R-028: no network / no model downloads / no training
# ---------------------------------------------------------------------------


def test_no_network_guard_blocks_inet_connections():
    # Tests R-028 [unit]: the namespace's test conftest-equivalent installs a
    # no-network guard (primary gate) — INET connects raise inside colm tests.
    with pytest.raises(RuntimeError, match="no-network|network disabled"):
        socket.create_connection(("127.0.0.1", 9), timeout=0.1)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        with pytest.raises(RuntimeError, match="no-network|network disabled"):
            sock.connect(("127.0.0.1", 9))
    finally:
        sock.close()


IMPORT_DENY_LIST = [
    "requests",
    "httpx",
    "urllib.request",
    "huggingface_hub",
    "transformers",
    "torch",
]


def _imported_names(py_path: Path) -> set[str]:
    names: set[str] = set()
    tree = ast.parse(py_path.read_text("utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            names.add(module)
            for alias in node.names:
                if module:
                    names.add(f"{module}.{alias.name}")
    return names


def test_import_scan_rejects_network_and_model_deny_list():
    # Tests R-028 [unit]: an import scan over reproducibility/colm_aims_2026/
    # rejects the enumerated deny-list (requests, httpx, urllib.request,
    # huggingface_hub, transformers, torch).
    py_files = sorted(NAMESPACE_DIR.glob("**/*.py"))
    assert py_files, "namespace must exist"
    for py in py_files:
        imported = _imported_names(py)
        for name in imported:
            for denied in IMPORT_DENY_LIST:
                assert not (
                    name == denied or name.startswith(denied + ".")
                ), f"{py.name} imports denied module {name!r}"


def test_fixtures_are_tiny_and_synthetic():
    # Tests R-028 [unit]: fixtures are tiny and synthetic — bounded total
    # size, no absolute paths inside any fixture file.
    total = 0
    for path in FIXTURES_DIR.glob("**/*"):
        if path.is_file():
            total += path.stat().st_size
            body = path.read_text("utf-8", errors="replace")
            assert "/Users/" not in body and "/home/" not in body, path
    assert 0 < total < 200_000, f"fixture corpus unexpectedly large: {total} bytes"


# ---------------------------------------------------------------------------
# R-034: deserialization safety
# ---------------------------------------------------------------------------

FORBIDDEN_IMPORTS = {"pickle", "marshal"}
FORBIDDEN_CALLS = {
    ("pickle", "load"),
    ("pickle", "loads"),
    ("marshal", "load"),
    ("marshal", "loads"),
    ("torch", "load"),
    ("yaml", "load"),  # non-safe YAML loading; yaml.safe_load stays legal
}


def test_no_unsafe_deserialization_in_namespace():
    # Tests R-034 [unit]: pickle, marshal, torch.load, and non-safe
    # yaml.load never appear in reproducibility/colm_aims_2026/ (AST scan,
    # sibling of R-028's scan). Evidence ingestion is JSON/JSONL only.
    # Audit item 3: also catches the from-import forms (`from yaml import
    # load`, `from pickle import load/loads`, aliases included) and
    # bare-Name calls of those tracked imports.
    py_files = sorted(NAMESPACE_DIR.glob("**/*.py"))
    assert py_files, "namespace must exist"
    for py in py_files:
        tree = ast.parse(py.read_text("utf-8"))
        tracked_names: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    assert root not in FORBIDDEN_IMPORTS, (
                        f"{py.name} imports {alias.name!r}"
                    )
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                root = module.split(".")[0]
                assert root not in FORBIDDEN_IMPORTS, (
                    f"{py.name} imports from {node.module!r}"
                )
                for alias in node.names:
                    if (root, alias.name) in FORBIDDEN_CALLS:
                        raise AssertionError(
                            f"{py.name} does `from {module} import "
                            f"{alias.name}` — unsafe deserialization is "
                            "forbidden (R-034)"
                        )
                    tracked_names[alias.asname or alias.name] = (
                        f"{module}.{alias.name}"
                    )
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and (func.value.id, func.attr) in FORBIDDEN_CALLS
            ):
                raise AssertionError(
                    f"{py.name} calls {func.value.id}.{func.attr} — "
                    "unsafe deserialization is forbidden (R-034)"
                )
            if isinstance(func, ast.Name) and func.id in tracked_names:
                origin = tracked_names[func.id]
                module, _, attr = origin.rpartition(".")
                if (module.split(".")[0], attr) in FORBIDDEN_CALLS:
                    raise AssertionError(
                        f"{py.name} calls {func.id}(...) (from-imported "
                        f"{origin}) — unsafe deserialization is forbidden "
                        "(R-034)"
                    )


def test_no_bare_replace_or_rename_publish_in_namespace():
    # Tests R-016 [unit] (audit BL-4b, sibling of the R-034 scan): bare
    # os.replace / os.rename / shutil.move and the single-argument Path-style
    # .replace(target) / .rename(target) never appear in
    # reproducibility/colm_aims_2026/ — final-path publication must route
    # through the create-once primitives (scripts/stopdff_v5/fileio.py).
    # Two-argument .replace(old, new) (str.replace) stays legal.
    # Audit item 3: also catches `from os import replace/rename` and
    # `from shutil import move` (aliases included) plus bare-Name calls of
    # those tracked imports.
    # Passes vacuously over the RED stubs by design; bites at GREEN.
    dangerous_from_imports = {
        ("os", "replace"),
        ("os", "rename"),
        ("shutil", "move"),
    }
    py_files = sorted(NAMESPACE_DIR.glob("**/*.py"))
    assert py_files, "namespace must exist"
    offenders: list[str] = []
    for py in py_files:
        tree = ast.parse(py.read_text("utf-8"))
        tracked_names: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    if (module, alias.name) in dangerous_from_imports:
                        offenders.append(
                            f"{py.name}:{node.lineno} from {module} "
                            f"import {alias.name}"
                        )
                        tracked_names[alias.asname or alias.name] = (
                            f"{module}.{alias.name}"
                        )
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            argc = len(node.args) + len(node.keywords)
            if isinstance(func, ast.Attribute):
                if isinstance(func.value, ast.Name) and (
                    (func.value.id == "os" and func.attr in ("replace", "rename"))
                    or (func.value.id == "shutil" and func.attr == "move")
                ):
                    offenders.append(
                        f"{py.name}:{node.lineno} {func.value.id}.{func.attr}"
                    )
                elif func.attr in ("replace", "rename") and argc == 1:
                    # Path.replace(target) / Path.rename(target) publish forms.
                    offenders.append(
                        f"{py.name}:{node.lineno} .{func.attr}(<1 arg>)"
                    )
            elif isinstance(func, ast.Name) and func.id in tracked_names:
                offenders.append(
                    f"{py.name}:{node.lineno} {func.id}(...) "
                    f"(from-imported {tracked_names[func.id]})"
                )
    assert not offenders, (
        "bare replace/rename publication found in namespace "
        f"(route through fileio create-once primitives): {offenders}"
    )
