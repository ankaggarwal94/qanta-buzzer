"""Two-mode fail-closed verifier (source-contract / release).

Spec rules owned here: R-012..R-015, R-017, R-019 (as verified surface),
R-021 (as CLI backend), R-033, R-035, R-036 (emission call), R-039
(canonical selection).
Spec: .correctless/specs/camera-ready-aims-evidence.md
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import ledger as ledger_mod
from . import receipt as receipt_mod
from . import pairing, schema
from .schema import ColmAimsError


class VacuousInputError(ColmAimsError):
    """Zero candidate artifacts / empty ledger / empty manifest (R-033)."""


class ContainmentError(ColmAimsError):
    """Expectations file not outside the verified artifact tree (R-013)."""


# R-017: closed source-mode verdict enum; strongest member PASS_SOURCE_ONLY.
VERDICT_SOURCE_PASS = "PASS_SOURCE_ONLY"
VERDICT_RELEASE_PASS = "PASS_RELEASE"
VERDICT_FAIL = "FAIL"
SOURCE_MODE_VERDICTS = frozenset({VERDICT_SOURCE_PASS, VERDICT_FAIL})
RELEASE_MODE_VERDICTS = frozenset({VERDICT_RELEASE_PASS, VERDICT_FAIL})

CERTIFIABLE = "CERTIFIABLE"
HISTORICAL_NONCERTIFYING = "HISTORICAL_NONCERTIFYING"

# R-012: the release binding legs, one per independently anchored binding.
BINDING_KEYS = (
    "schema_profile",
    "producer",
    "semantic_command",
    "seeds",
    "dirty_state",
    "splits",
    "calibration_identity",
    "continuation_identity",
    "input_hashes",
    "split_metadata_sha256",
    "mc_build",
    "model",
    "runtime_packages",
)

_EXPECTATIONS_KEYS = frozenset(
    {"schema_version", "anchor", "rights_inventory", "tree_files", "bindings"}
)
_MANIFEST_KEYS = frozenset(
    {"schema_version", "artifacts", "allowlist_undeclared"}
)

# R-012/R-014: the estimand-defining dependency-closure identities.
_CLOSURE_IDENTITY_KEYS = ("calibration_identity", "continuation_identity")

_EXPECTED_LAYOUT = (
    "profile.json (strict constructed-reference profile), records.jsonl"
    " (retained per-item records), presentation_manifest.json"
    " (presentation manifest)"
)

# Row-status strength order for the R-012 stale-status recomputation gate.
_STATUS_STRENGTH = {"FAIL": 0, "UNVERIFIED": 1, "PASS": 2}


@dataclass
class VerificationReport:
    """Structured result of one verifier run (type definition for tests)."""

    mode: str
    verdict: str
    legs: list[dict[str, Any]] = field(default_factory=list)
    validated_artifacts: list[str] = field(default_factory=list)
    receipt_path: Path | None = None
    classifications: dict[str, str] = field(default_factory=dict)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _keyset_sha256(keys: list[str]) -> str:
    return hashlib.sha256("\n".join(sorted(keys)).encode("utf-8")).hexdigest()


def _tree_file_map(tree: Path) -> dict[str, Path]:
    return {
        p.relative_to(tree).as_posix(): p
        for p in Path(tree).rglob("*")
        if p.is_file()
    }


def _digest_over_lines(lines: list[str]) -> str:
    """Pinned digest shape (R-036): sha256 over newline-joined
    ``<posix relpath>:<sha256>`` lines with a trailing newline."""
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def _tree_digest(tree: Path) -> str:
    """Pinned input-tree digest (R-036) over every file in the tree."""
    entries = {rel: _sha256_file(p) for rel, p in _tree_file_map(tree).items()}
    return _digest_over_lines(
        [f"{rel}:{sha}" for rel, sha in sorted(entries.items())]
    )


def _code_digest() -> str:
    """Pinned verifier-code digest (R-036) over the namespace's .py files."""
    namespace = Path(__file__).resolve().parent
    return _digest_over_lines(
        [
            f"{p.relative_to(namespace).as_posix()}:{_sha256_file(p)}"
            for p in sorted(namespace.glob("**/*.py"))
        ]
    )


def _pass(leg_id: str) -> dict[str, Any]:
    return {"leg_id": leg_id, "outcome": "PASS"}


def _fail(
    leg_id: str,
    *,
    expected: Any,
    observed: Any,
    remediation: str = "ARTIFACT_DEFECT",
) -> dict[str, Any]:
    return {
        "leg_id": leg_id,
        "outcome": "FAIL",
        "expected": expected,
        "observed": observed,
        "remediation_class": remediation,
    }


def _record_leg(
    legs: list[dict[str, Any]],
    leg_id: str,
    passed: bool,
    *,
    expected: Any,
    observed: Any,
    remediation: str = "ARTIFACT_DEFECT",
) -> None:
    """Append one PASS leg, or one FAIL leg carrying expected/observed."""
    if passed:
        legs.append(_pass(leg_id))
    else:
        legs.append(
            _fail(
                leg_id,
                expected=expected,
                observed=observed,
                remediation=remediation,
            )
        )


def _is_resolved_identity(value: Any) -> bool:
    """A closure identity is resolved when it is a non-empty string other than
    the explicit ``UNRESOLVED`` marker (R-012/R-014)."""
    return isinstance(value, str) and bool(value) and value != "UNRESOLVED"


def classify_certifiability(profile: dict[str, Any]) -> str:
    """CERTIFIABLE vs HISTORICAL_NONCERTIFYING closure classification (R-014).

    Only a producer/closure change invalidates an artifact: superseded,
    dirty, or unresolved estimand-defining dependency closures classify
    HISTORICAL_NONCERTIFYING; non-closure metadata differences do not.
    """
    if not isinstance(profile, dict):
        raise ColmAimsError("profile must be an object (R-014)")
    prov = profile.get("provenance") or {}
    dirty_state = prov.get("dirty_state") or {}
    if dirty_state.get("git_dirty") is not False:
        return HISTORICAL_NONCERTIFYING
    if "superseded_by_producer_sha256" in prov:
        return HISTORICAL_NONCERTIFYING
    for name in _CLOSURE_IDENTITY_KEYS:
        if not _is_resolved_identity(prov.get(name)):
            return HISTORICAL_NONCERTIFYING
    return CERTIFIABLE


def parse_legacy_profile(data: bytes) -> dict[str, Any]:
    """Parse a known legacy profile family from captured bytes (R-014).

    Families: the ``paper_exports/csli.json``, ``paper_exports/
    calibration.json``, and ``paper_exports/audit_card.json`` aggregate
    formats. Legacy artifacts are refused only on a demonstrably missing
    named invariant — never merely for predating the strict schema.
    """
    try:
        obj = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise schema.TypedIngressError(
            f"legacy artifact bytes are not valid JSON: {exc}"
        ) from exc
    if not isinstance(obj, dict):
        raise schema.TypedIngressError("legacy artifact must be a JSON object")
    if "panel_csli" in obj:
        family = "csli"
    elif "per_bucket" in obj and "max_ece" in obj:
        family = "calibration"
    elif "metrics" in obj and "overall_verdict" in obj:
        family = "audit_card"
    else:
        raise schema.SchemaValidationError(
            "unknown legacy artifact family — not one of the enumerated"
            " csli/calibration/audit_card aggregate formats (R-014)"
        )
    if "metadata" not in obj:
        raise schema.SchemaValidationError(
            f"legacy {family} artifact is missing the named invariant"
            " 'metadata' (generation/provenance block) — refusal requires a"
            " demonstrably missing named invariant (R-014)"
        )
    return {"legacy_family": family, "aggregate_only": True, "payload": obj}


def legacy_certifies(legacy: dict[str, Any], claim_kind: str) -> bool:
    """Whether a legacy artifact can certify a claim kind (R-014).

    Aggregate-only files cannot certify per-item paired claims.
    """
    if legacy.get("aggregate_only", True):
        return claim_kind == "aggregate"
    return True


def resolve_canonical_package(runs_root: Path, ledger: dict[str, Any]) -> Path:
    """Canonical run selection strictly via the ledger pointer (R-039)."""
    run_id = ledger.get("canonical_run_id")
    if not isinstance(run_id, str) or not run_id:
        raise ColmAimsError(
            "ledger declares no canonical_run_id pointer — canonical"
            " selection happens only via the ledger/expectations pointer,"
            " never newest-wins (R-039)"
        )
    path = Path(runs_root) / run_id
    if not path.is_dir():
        raise ColmAimsError(
            f"canonical run pointer {run_id!r} does not resolve to a"
            " published run directory under the runs root — dangling"
            " pointers never fall back (R-039)"
        )
    return path


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _load_expectations(path: Path) -> tuple[dict[str, Any], bytes]:
    """Typed, fail-closed load of the anchored expectations file (R-022)."""
    name = Path(path).name
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise schema.TypedIngressError(
            f"{name}: unreadable expectations file"
            f" ({exc.__class__.__name__}) (R-020)"
        ) from exc
    try:
        obj = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise schema.TypedIngressError(
            f"{name}: malformed JSON: {exc} (R-020)"
        ) from exc
    if not isinstance(obj, dict):
        raise schema.TypedIngressError(f"{name}: expectations must be an object")
    if "schema_version" not in obj:
        raise schema.TypedIngressError(
            f"{name}: missing required field 'schema_version' (R-020)"
        )
    version = obj["schema_version"]
    if version != schema.SCHEMA_VERSION:
        raise schema.TypedIngressError(
            f"{name}: unsupported schema_version {version!r}; supported range"
            f" {schema.SUPPORTED_SCHEMA_VERSION_MIN}.."
            f"{schema.SUPPORTED_SCHEMA_VERSION_MAX}; verifier revision"
            f" {schema.VERIFIER_REVISION} (R-020)"
        )
    unknown = sorted(set(obj) - _EXPECTATIONS_KEYS)
    if unknown:
        raise schema.TypedIngressError(
            f"{name}: unknown expectations key(s) {unknown} — the config"
            " surface fails closed; no key disables a release gate (R-022)"
        )
    return obj, data


def _load_json_lenient(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    """Collect-don't-halt sidecar load: (object, error-description)."""
    name = Path(path).name
    if not Path(path).is_file():
        return None, f"{name}: absent"
    try:
        obj = json.loads(Path(path).read_bytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"{name}: malformed JSON: {exc}"
    if not isinstance(obj, dict):
        return None, f"{name}: must be a JSON object"
    return obj, None


def _git_object_exists(commit: str) -> bool | None:
    """Optional anchor object-existence check when a repository is available
    (R-013). Returns None when no repository/git is reachable from cwd."""
    if not (Path.cwd() / ".git").exists():
        return None
    try:
        proc = subprocess.run(
            ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    return proc.returncode == 0


# ---------------------------------------------------------------------------
# run_verifier
# ---------------------------------------------------------------------------


def _classify_legacy_artifacts(
    files: dict[str, Path]
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Classify known historical artifact families from captured bytes (R-014).

    Returns the parsed legacy artifacts keyed by tree-relative path, plus
    their human-readable classifications. A file that is not a known legacy
    family is left to the other gates — never refused merely for predating
    the strict schema.
    """
    parsed_by_rel: dict[str, dict[str, Any]] = {}
    classifications: dict[str, str] = {}
    for rel in sorted(files):
        if rel in ("profile.json", "presentation_manifest.json"):
            continue
        if not rel.endswith(".json"):
            continue
        try:
            parsed = parse_legacy_profile(files[rel].read_bytes())
        except ColmAimsError:
            continue  # not a known legacy family; other gates govern it
        parsed_by_rel[rel] = parsed
        classifications[rel] = (
            f"legacy_{parsed['legacy_family']}_aggregate"
            " (historical, aggregate-only)"
        )
    return parsed_by_rel, classifications


def _reject_empty_evaluation(cells: Any) -> None:
    """R-006/R-012: an explicitly empty evaluation errors before any report."""
    if not isinstance(cells, list):
        return
    for cell in cells:
        if (
            isinstance(cell, dict)
            and isinstance(cell.get("counts"), dict)
            and cell["counts"].get("n_pairing_population") == 0
        ):
            raise schema.EmptyEvaluationError(
                f"cell {cell.get('cell_id')!r} declares an explicitly empty"
                " evaluation (n_pairing_population == 0); refused before any"
                " report is emitted (R-006/R-012)"
            )


def run_verifier(
    tree: Path,
    *,
    mode: str,
    receipts_dir: Path,
    expectations: Path | None = None,
) -> VerificationReport:
    """Run one verifier pass over an artifact tree; never mutate inputs."""
    if mode not in ("source", "release"):
        raise ColmAimsError(
            f"unknown verifier mode {mode!r}; expected 'source' or 'release'"
        )
    tree = Path(tree)
    receipts_dir = Path(receipts_dir)
    tree_resolved = tree.resolve()

    files = _tree_file_map(tree) if tree.is_dir() else {}
    if not files or "profile.json" not in files:
        raise VacuousInputError(
            f"zero candidate artifacts under {tree_resolved}; expected"
            f" layout: {_EXPECTED_LAYOUT} (R-033)"
        )

    expectations_obj: dict[str, Any] | None = None
    expectations_bytes: bytes | None = None
    expectations_path: Path | None = None
    if mode == "release":
        if expectations is None:
            raise ColmAimsError(
                "release mode requires an independently anchored expectations"
                " file located outside the verified artifact tree (R-013)"
            )
        expectations_path = Path(expectations)
        if schema.resolves_inside(expectations_path, tree):
            raise ContainmentError(
                "expectations file resolves inside the verified artifact"
                " tree — self-attestation is refused; containment decisions"
                " use fully resolved, symlink-free paths (R-013)"
            )
        expectations_obj, expectations_bytes = _load_expectations(
            expectations_path
        )

    # Typed ingress (R-020) — unreadable/malformed inputs halt here.
    profile = schema.load_artifact(files["profile.json"], tree_root=tree)
    records: list[dict[str, Any]] | None = None
    if "records.jsonl" in files:
        records = schema.load_artifact(
            files["records.jsonl"], tree_root=tree
        )["records"]

    cells = profile.get("cells")
    _reject_empty_evaluation(cells)

    legs: list[dict[str, Any]] = []
    validated: list[str] = []

    # ---- shared (source minimum positive set) --------------------------
    legs.append(_pass("typed_ingress"))

    profile_valid = True
    try:
        schema.validate_profile(profile)
        legs.append(_pass("profile_validation"))
    except schema.SchemaValidationError as exc:
        profile_valid = False
        legs.append(
            _fail(
                "profile_validation",
                expected="valid strict constructed-reference profile"
                " (R-001..R-003, R-029, R-031, R-032)",
                observed=str(exc),
            )
        )

    records_valid = False
    if records is None:
        legs.append(
            _fail(
                "records_present",
                expected="records.jsonl with retained per-item records —"
                " absent records are non-certifying (R-015)",
                observed="records.jsonl absent from the artifact tree",
            )
        )
    else:
        record_errors: list[str] = []
        for record in records:
            try:
                schema.validate_record(record)
            except schema.RecordValidationError as exc:
                record_errors.append(str(exc))
        if record_errors:
            legs.append(
                _fail(
                    "records_validation",
                    expected="non-reversible per-item records (R-031)",
                    observed="; ".join(record_errors[:5]),
                )
            )
        else:
            legs.append(_pass("records_validation"))
            records_valid = True

    cells_valid = True
    if records is None:
        cells_valid = False
    elif isinstance(cells, list):
        for cell in cells:
            cell_id = (
                cell.get("cell_id", "unnamed")
                if isinstance(cell, dict)
                else "unnamed"
            )
            leg_id = f"cell_{cell_id}_validation"
            try:
                pairing.validate_cell(cell, records)
                legs.append(_pass(leg_id))
            except schema.EmptyEvaluationError:
                raise
            except (ColmAimsError, KeyError, TypeError, ValueError) as exc:
                cells_valid = False
                legs.append(
                    _fail(
                        leg_id,
                        expected="cell count/rate/key/summary/interval"
                        " recomputation identities hold"
                        " (R-005..R-011, R-015)",
                        observed=str(exc),
                    )
                )

    legacy_parsed, classifications = _classify_legacy_artifacts(files)

    try:
        closure = classify_certifiability(profile)
    except ColmAimsError:
        closure = HISTORICAL_NONCERTIFYING
    classifications["profile.json"] = closure

    artifacts_valid = profile_valid and records_valid and cells_valid
    if profile_valid and cells_valid:
        validated.append("profile.json")
    if records_valid:
        validated.append("records.jsonl")

    # ---- release-only gates --------------------------------------------
    if mode == "release":
        assert expectations_obj is not None and expectations_path is not None
        _release_legs(
            legs,
            expectations_obj,
            expectations_path,
            files,
            profile,
            records,
            legacy_parsed,
            closure,
            artifacts_valid,
        )

    failing = [leg for leg in legs if leg["outcome"] == "FAIL"]
    if failing:
        verdict = VERDICT_FAIL
    elif mode == "source":
        verdict = VERDICT_SOURCE_PASS
    else:
        verdict = VERDICT_RELEASE_PASS

    report = VerificationReport(
        mode=mode,
        verdict=verdict,
        legs=legs,
        validated_artifacts=validated,
        classifications=classifications,
    )

    payload = {
        "schema_version": receipt_mod.RECEIPT_SCHEMA_VERSION,
        "mode": mode,
        "verdict": verdict,
        "legs": legs,
        "validated_artifacts": validated,
        "classifications": classifications,
        "input_tree_sha256": _tree_digest(tree),
        "expectations_anchor_sha256": (
            hashlib.sha256(expectations_bytes).hexdigest()
            if expectations_bytes is not None
            else None
        ),
        "verifier_code_sha256": _code_digest(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    report.receipt_path = receipt_mod.emit_receipt(
        payload, receipts_dir=receipts_dir, verified_tree=tree
    )
    return report


# ---------------------------------------------------------------------------
# Release legs
# ---------------------------------------------------------------------------


def _release_legs(
    legs: list[dict[str, Any]],
    exp: dict[str, Any],
    expectations_path: Path,
    files: dict[str, Path],
    profile: dict[str, Any],
    records: list[dict[str, Any]] | None,
    legacy_parsed: dict[str, dict[str, Any]],
    closure: str,
    artifacts_valid: bool,
) -> None:
    """Append every release-only leg, in the pinned order.

    Each section owns one gate family; a section that cannot even reach its
    inputs records the failure and returns rather than skipping silently.
    """
    prov = profile.get("provenance") or {}
    base = expectations_path.parent

    ledger_doc = _anchor_legs(legs, exp, base, prov)
    _tree_file_legs(legs, exp, files)
    _binding_legs(legs, exp, files, profile, prov)

    dirty_state = prov.get("dirty_state") or {}
    _record_leg(
        legs,
        "dirty_state_clean",
        dirty_state.get("git_dirty") is False,
        expected={"git_dirty": False},
        observed={"git_dirty": dirty_state.get("git_dirty")},
    )
    _identity_resolution_legs(legs, prov)
    _model_legs(legs, prov)
    _splits_legs(legs, prov, records)
    _mc_build_legs(legs, prov)
    _record_leg(
        legs,
        "closure_certifiability",
        closure == CERTIFIABLE,
        expected=CERTIFIABLE,
        observed=closure,
    )

    _rights_legs(legs, exp, base, files)
    _manifest_legs(legs, files)
    _ledger_legs(legs, ledger_doc, files, legacy_parsed, artifacts_valid)


def _anchor_legs(
    legs: list[dict[str, Any]],
    exp: dict[str, Any],
    base: Path,
    prov: dict[str, Any],
) -> dict[str, Any] | None:
    """Anchor cross-check before any expectation is consumed (R-013).

    Returns the frozen claim ledger document when one was reachable and
    parseable, else ``None``.
    """
    anchor = exp.get("anchor")
    if not isinstance(anchor, dict):
        legs.append(
            _fail(
                "anchor",
                expected="anchor block binding a reviewed source commit and"
                " the frozen claim ledger (R-013)",
                observed="absent",
                remediation="MISSING_EXPECTATION",
            )
        )
        return None

    ledger_doc: dict[str, Any] | None = None
    ledger_rel = anchor.get("ledger_path", "ledger.json")
    ledger_path = base / str(ledger_rel)
    anchor_ledger_sha = anchor.get("ledger_sha256")
    if not ledger_path.is_file():
        legs.append(
            _fail(
                "anchor_ledger",
                expected=anchor_ledger_sha,
                observed=f"frozen ledger {ledger_rel!r} absent",
                remediation="MISSING_EXPECTATION",
            )
        )
    else:
        actual_ledger_sha = _sha256_file(ledger_path)
        _record_leg(
            legs,
            "anchor_ledger",
            actual_ledger_sha == anchor_ledger_sha,
            expected=anchor_ledger_sha,
            observed=actual_ledger_sha,
        )
        ledger_doc, ledger_err = _load_json_lenient(ledger_path)
        if ledger_err is not None:
            legs.append(
                _fail(
                    "ledger_parse",
                    expected="parseable frozen claim ledger",
                    observed=ledger_err,
                )
            )

    anchor_commit = anchor.get("source_commit")
    observed_commit = (prov.get("dirty_state") or {}).get("source_commit")
    if not schema.is_commit_sha(anchor_commit):
        legs.append(
            _fail(
                "anchor_source_commit",
                expected="full-length reviewed source commit SHA (R-013)",
                observed=anchor_commit,
                remediation="MISSING_EXPECTATION",
            )
        )
    elif anchor_commit != observed_commit:
        # String-exact identity comparison; works without a git checkout.
        legs.append(
            _fail(
                "anchor_source_commit",
                expected=anchor_commit,
                observed=observed_commit,
            )
        )
    elif _git_object_exists(anchor_commit) is False:
        legs.append(
            _fail(
                "anchor_source_commit_object",
                expected=f"commit {anchor_commit} present in the"
                " available repository (R-013)",
                observed="object not found",
            )
        )
    else:
        legs.append(_pass("anchor_source_commit"))
    return ledger_doc


def _tree_file_legs(
    legs: list[dict[str, Any]], exp: dict[str, Any], files: dict[str, Path]
) -> None:
    """Artifact-tree byte identity against the anchored hash map (R-014)."""
    declared_tree = exp.get("tree_files")
    if not isinstance(declared_tree, dict):
        legs.append(
            _fail(
                "tree_files",
                expected="tree byte-hash map in the expectations file",
                observed="absent",
                remediation="MISSING_EXPECTATION",
            )
        )
        return

    problems: list[str] = []
    actual = {rel: _sha256_file(path) for rel, path in files.items()}
    for rel, sha in sorted(declared_tree.items()):
        if rel not in actual:
            problems.append(f"declared-but-absent {rel!r}")
        elif actual[rel] != sha:
            problems.append(f"byte-hash mismatch {rel!r}")
    for rel in sorted(set(actual) - set(declared_tree)):
        problems.append(f"present-but-unanchored {rel!r}")
    _record_leg(
        legs,
        "tree_files",
        not problems,
        expected="artifact tree byte-identical to the anchored hash map"
        " (R-014)",
        observed="; ".join(problems),
    )


def _binding_legs(
    legs: list[dict[str, Any]],
    exp: dict[str, Any],
    files: dict[str, Path],
    profile: dict[str, Any],
    prov: dict[str, Any],
) -> None:
    """The thirteen independently anchored binding legs (R-012)."""
    observed_bindings: dict[str, Any] = {
        "schema_profile": {
            "profile_id": profile.get("profile_id"),
            "schema_version": profile.get("schema_version"),
            "profile_sha256": _sha256_file(files["profile.json"]),
        },
        "producer": {
            "entrypoint": prov.get("producer_entrypoint"),
            "sha256": prov.get("producer_sha256"),
            "helper_sha256s": prov.get("helper_sha256s"),
        },
        "semantic_command": prov.get("semantic_command"),
        "seeds": prov.get("seeds"),
        "dirty_state": prov.get("dirty_state"),
        "splits": prov.get("splits"),
        "calibration_identity": prov.get("calibration_identity"),
        "continuation_identity": prov.get("continuation_identity"),
        "input_hashes": prov.get("input_sha256"),
        "split_metadata_sha256": prov.get("split_metadata_sha256"),
        "mc_build": prov.get("mc_build"),
        "model": prov.get("model"),
        "runtime_packages": prov.get("runtime_packages"),
    }
    bindings = exp.get("bindings")
    if not isinstance(bindings, dict):
        legs.append(
            _fail(
                "bindings",
                expected="per-leg bindings block in the expectations file",
                observed="absent",
                remediation="MISSING_EXPECTATION",
            )
        )
        bindings = {}
    for key in BINDING_KEYS:
        leg_id = f"binding_{key}"
        observed = observed_bindings[key]
        if key not in bindings:
            legs.append(
                _fail(
                    leg_id,
                    expected="<missing anchored expectation>",
                    observed=observed,
                    remediation="MISSING_EXPECTATION",
                )
            )
            continue
        expected = bindings[key]
        if expected != observed:
            legs.append(_fail(leg_id, expected=expected, observed=observed))
            continue
        if key == "input_hashes" and isinstance(expected, dict):
            mismatched: dict[str, Any] = {}
            for fname, sha in expected.items():
                actual_sha = (
                    _sha256_file(files[fname]) if fname in files else "absent"
                )
                if actual_sha != sha:
                    mismatched[fname] = actual_sha
            if mismatched:
                legs.append(
                    _fail(leg_id, expected=expected, observed=mismatched)
                )
                continue
        legs.append(_pass(leg_id))


def _identity_resolution_legs(
    legs: list[dict[str, Any]], prov: dict[str, Any]
) -> None:
    """Closure identities must be resolved, never left UNRESOLVED (R-012)."""
    for name in _CLOSURE_IDENTITY_KEYS:
        value = prov.get(name)
        _record_leg(
            legs,
            f"binding_{name}_resolved",
            _is_resolved_identity(value),
            expected=f"resolved {name}",
            observed=value,
            remediation="AUTHOR_DECISION_REQUIRED",
        )


def _model_legs(legs: list[dict[str, Any]], prov: dict[str, Any]) -> None:
    """Immutable model revision identity + weights content hash (R-012)."""
    model = prov.get("model") or {}
    revision = model.get("revision")
    digest_manifest = model.get("byte_digest_manifest")
    if revision is not None:
        # Short hashes, tags, branch names, and bare repo ids are rejected —
        # repo ids are reassignable (R-012).
        _record_leg(
            legs,
            "model_revision_immutability",
            schema.is_commit_sha(revision),
            expected="immutable full-length 40-hex commit SHA, or a"
            " complete canonical byte-digest manifest",
            observed={"revision": revision},
        )
    else:
        complete_manifest = (
            isinstance(digest_manifest, dict)
            and bool(digest_manifest)
            and all(
                schema.is_sha256_hex(v) for v in digest_manifest.values()
            )
        )
        _record_leg(
            legs,
            "model_revision_immutability",
            complete_manifest,
            expected="immutable full-length commit SHA or complete"
            " canonical byte-digest manifest",
            observed={
                "revision": None,
                "byte_digest_manifest": digest_manifest,
            },
        )
    weights_sha = model.get("weights_sha256")
    _record_leg(
        legs,
        "model_weights_hash",
        schema.is_sha256_hex(weights_sha),
        expected="content-level sha256 of the loaded weights file",
        observed=weights_sha,
    )


def _splits_legs(
    legs: list[dict[str, Any]],
    prov: dict[str, Any],
    records: list[dict[str, Any]] | None,
) -> None:
    """Split disjointness + eval-split key-set recomputation (R-012)."""
    splits = prov.get("splits") or {}
    _record_leg(
        legs,
        "splits_zero_overlap",
        splits.get("zero_overlap") is True,
        expected={"zero_overlap": True},
        observed={"zero_overlap": splits.get("zero_overlap")},
    )
    if records is None:
        return
    eval_split = splits.get("eval") or {}
    keys = [
        r.get("item_key") for r in records if isinstance(r.get("item_key"), str)
    ]
    recomputed_hash = _keyset_sha256(keys)
    declared_hash = eval_split.get("keyset_sha256")
    declared_count = eval_split.get("count")
    _record_leg(
        legs,
        "splits_eval_recompute",
        declared_hash == recomputed_hash and declared_count == len(keys),
        expected={"keyset_sha256": declared_hash, "count": declared_count},
        observed={"keyset_sha256": recomputed_hash, "count": len(keys)},
    )


def _mc_build_legs(legs: list[dict[str, Any]], prov: dict[str, Any]) -> None:
    """MC-build freshness + coverage/retention recording (R-012)."""
    mc_build = prov.get("mc_build") or {}
    _record_leg(
        legs,
        "mc_build_freshness",
        mc_build.get("built_after_split") is True,
        expected={"built_after_split": True},
        observed={"built_after_split": mc_build.get("built_after_split")},
    )
    coverage_recorded = (
        isinstance(mc_build.get("coverage_rate"), (int, float))
        and isinstance(mc_build.get("retention_policy"), str)
        and isinstance(mc_build.get("retained_count"), int)
    )
    _record_leg(
        legs,
        "mc_build_coverage_retention",
        coverage_recorded,
        expected="coverage rate + retention policy/counts recorded",
        observed=mc_build,
    )


def _rights_legs(
    legs: list[dict[str, Any]],
    exp: dict[str, Any],
    base: Path,
    files: dict[str, Path],
) -> None:
    """Rights inventory binding + release clearance (R-026/R-035)."""
    rights_decl = exp.get("rights_inventory")
    if not isinstance(rights_decl, dict):
        legs.append(
            _fail(
                "rights_inventory",
                expected="rights inventory binding (path + sha256) in the"
                " expectations file (R-026)",
                observed="absent",
                remediation="MISSING_EXPECTATION",
            )
        )
        return

    rights_path = base / str(rights_decl.get("path", "rights.json"))
    rights_obj, rights_err = _load_json_lenient(rights_path)
    if rights_err is not None:
        legs.append(
            _fail(
                "rights_inventory",
                expected="parseable rights inventory",
                observed=rights_err,
            )
        )
        return

    actual_sha = _sha256_file(rights_path)
    _record_leg(
        legs,
        "rights_inventory_hash",
        actual_sha == rights_decl.get("sha256"),
        expected=rights_decl.get("sha256"),
        observed=actual_sha,
    )
    try:
        # R-035: rights cover every file FOUND, not merely declared.
        ledger_mod.check_rights_release(rights_obj, sorted(files))
        legs.append(_pass("rights_release"))
    except ledger_mod.RightsError as exc:
        legs.append(
            _fail(
                "rights_release",
                expected="every included path VERIFIED_ALLOWED and"
                " inventoried (R-026)",
                observed=str(exc),
                remediation="AUTHOR_DECISION_REQUIRED",
            )
        )


def _manifest_legs(
    legs: list[dict[str, Any]], files: dict[str, Path]
) -> None:
    """Presentation manifest reconciliation (R-033/R-035)."""
    if "presentation_manifest.json" not in files:
        legs.append(
            _fail(
                "presentation_manifest_present",
                expected="presentation_manifest.json in the artifact tree",
                observed="absent",
            )
        )
        return

    manifest_obj, manifest_err = _load_json_lenient(
        files["presentation_manifest.json"]
    )
    if manifest_err is None:
        unknown_keys = sorted(set(manifest_obj) - _MANIFEST_KEYS)
        if unknown_keys:
            manifest_err = f"unknown manifest key(s) {unknown_keys}"
    if manifest_err is not None:
        legs.append(
            _fail(
                "presentation_manifest_parse",
                expected="typed presentation manifest",
                observed=manifest_err,
            )
        )
        return

    declared = [
        a.get("path")
        for a in manifest_obj.get("artifacts", [])
        if isinstance(a, dict) and isinstance(a.get("path"), str)
    ]
    allowlist = [
        p
        for p in manifest_obj.get("allowlist_undeclared", [])
        if isinstance(p, str)
    ]
    _record_leg(
        legs,
        "manifest_nonempty",
        bool(declared),
        expected=">=1 manifest-declared artifact (R-033)",
        observed="0 declared artifacts",
    )
    ghosts = sorted(p for p in declared if p not in files)
    _record_leg(
        legs,
        "manifest_declared_absent",
        not ghosts,
        expected="every manifest-declared artifact present",
        observed=ghosts,
    )
    undeclared = sorted(
        rel
        for rel in files
        if rel not in declared
        and rel not in allowlist
        and rel != "presentation_manifest.json"
    )
    _record_leg(
        legs,
        "manifest_undeclared_present",
        not undeclared,
        expected="no present-but-undeclared file without an explicit"
        " per-file allowlist entry (R-035)",
        observed=undeclared,
    )


def _ledger_legs(
    legs: list[dict[str, Any]],
    ledger_doc: dict[str, Any] | None,
    files: dict[str, Path],
    legacy_parsed: dict[str, dict[str, Any]],
    artifacts_valid: bool,
) -> None:
    """Claim-ledger validation + per-row status recomputation (R-012/R-033)."""
    if ledger_doc is None:
        legs.append(
            _fail(
                "ledger_present",
                expected="frozen claim ledger reachable via the anchor",
                observed="absent or unparseable",
                remediation="MISSING_EXPECTATION",
            )
        )
        return
    try:
        ledger_mod.validate_ledger(ledger_doc)
        legs.append(_pass("ledger_validation"))
    except ledger_mod.LedgerValidationError as exc:
        legs.append(
            _fail(
                "ledger_validation",
                expected="structurally valid claim ledger (R-023..R-025,"
                " R-030)",
                observed=str(exc),
            )
        )
    rows = ledger_doc.get("rows") or []
    _record_leg(
        legs,
        "ledger_nonempty",
        bool(rows),
        expected=">=1 retained claim-ledger row (R-033)",
        observed="empty ledger rows",
    )
    for row in rows:
        if not isinstance(row, dict):
            continue
        status = row.get("status")
        if status == "EXTERNAL":
            continue  # R-024: EXTERNAL rows are immune to repo tooling.
        claim_id = row.get("claim_id", "unnamed")
        recomputed = _recompute_row_status(
            row, files, legacy_parsed, artifacts_valid
        )
        _record_leg(
            legs,
            f"ledger_row_{claim_id}_recompute",
            _STATUS_STRENGTH.get(status, 0) <= _STATUS_STRENGTH[recomputed],
            expected="recorded status no stronger than the recomputed"
            f" status {recomputed!r} (R-012)",
            observed={
                "claim_id": claim_id,
                "recorded": status,
                "recomputed": recomputed,
            },
        )


def _recompute_row_status(
    row: dict[str, Any],
    files: dict[str, Path],
    legacy_parsed: dict[str, dict[str, Any]],
    artifacts_valid: bool,
) -> str:
    """Recompute a non-EXTERNAL claim row's status from current verification
    (R-012). The recomputed status is the strongest level the current run can
    support — never taken from the recorded row."""
    if row.get("rights_status") != "VERIFIED_ALLOWED":
        return "UNVERIFIED"
    artifact = row.get("artifact_id")
    if not isinstance(artifact, str) or artifact not in files:
        return "UNVERIFIED"
    if artifact in legacy_parsed:
        claim_kind = (
            "per_item_paired"
            if row.get("estimand") == "signed_index_shift_mc_minus_ref"
            else "aggregate"
        )
        if not legacy_certifies(legacy_parsed[artifact], claim_kind):
            return "UNVERIFIED"
        return "PASS"
    if not artifacts_valid:
        return "UNVERIFIED"
    return "PASS"
