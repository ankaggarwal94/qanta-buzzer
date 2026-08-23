"""R-081 single-use launcher: the ONLY sanctioned path to the ONE run.

Validates the PRE_RUN_READY certificate byte-for-byte against the author's
activation digest, re-proves the live repository state against the
certified head, refuses ambient provenance overrides, re-verifies both
model snapshots, consumes the single-use exception via a CREATE-ONCE
ledger, launches the producer EXACTLY once into a fresh quarantine
directory with a pinned environment, and promotes the quarantine to the
final destination only after a mandatory parity-comparator PASS.

Spec: .correctless/specs/camera-ready-aims-evidence-2.md R-081/R-082
(operational-rejection repair, 2026-08-22).

Error taxonomy: every PRE-launch defect raises ``LaunchRefusal`` (the run
never started; the message names the refusal class); POST-launch defects
(nonzero exit, comparator FAIL) raise ``RunFailed`` with the quarantine
left intact and a STOP report written beside the outputs. Both are
``schema.ColmAimsError`` subclasses.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from . import phase4, schema

# reproducibility/colm_aims_2026/phase4_launcher.py -> repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# R-081 class (3)/(6): ambient provenance overrides — PRESENCE refuses,
# value ignored (an ambient EMPTY status would fake-clean the guard).
AMBIENT_OVERRIDE_VARS = ("MODAL_HOST_GIT_STATUS", "MODAL_HOST_GIT_COMMIT")
AMBIENT_ENV_PREFIX = "MODAL_HOST"

# R-081 (3): deterministic, offline, single-threaded child environment.
LAUNCH_ENV_PINS = {
    "PYTHONHASHSEED": "0",
    "OMP_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
}

LAUNCHER_CONFIG_KEYS = (
    "certificate_path",
    "activation_digest",
    "quarantine_dir",
    "promote_to",
    "ledger_path",
    "snapshot_manifest_path",
    "snapshot_dirs",
    "anchor_path",
)

STOP_REPORT_NAME = "STOP_REPORT.json"


class LaunchRefusal(schema.ColmAimsError):
    """Pre-launch refusal (R-081): the producer was never started."""


class RunFailed(schema.ColmAimsError):
    """Post-launch failure (R-081): nonzero exit or comparator FAIL —
    quarantine left intact, STOP report written, nothing promoted."""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_run_git(cmd: list[str]) -> str:
    """Production git runner: subprocess in the repository root."""
    completed = subprocess.run(
        [str(part) for part in cmd],
        cwd=str(_REPO_ROOT),
        capture_output=True,
        text=True,
        check=True,
    )
    return completed.stdout


def _default_launch(argv: list[str], env: dict[str, str]) -> int:
    """Production launcher: run the producer subprocess from the repo root
    (the certificate command carries repo-relative paths) and return its
    exit code."""
    completed = subprocess.run(
        [str(part) for part in argv], env=dict(env), cwd=str(_REPO_ROOT)
    )
    return completed.returncode


def _compose_argv(
    command: list[str], quarantine_dir: Path, activation_digest: str
) -> tuple[list[str], str | None]:
    """Compose the child argv FROM the certificate's recorded command.

    ONLY output paths are remapped: the ``--out`` value to
    ``quarantine_dir/<basename>`` and the ``--records-out`` value to the
    quarantine directory ITSELF (amended R-080: --records-out is the
    PARENT; the producer's exporter owns the ``records/`` segment).
    ``--certificate-digest <activation_digest>`` is appended. Every other
    token is preserved verbatim. Returns ``(argv, out_basename)``.
    """
    argv: list[str] = []
    out_basename: str | None = None
    index = 0
    total = len(command)
    while index < total:
        token = command[index]
        if token in ("--out", "--records-out"):
            if index + 1 >= total:
                raise LaunchRefusal(
                    f"certificate command carries a dangling {token} flag —"
                    " argv composition refused (R-081)"
                )
            value = command[index + 1]
            if token == "--out":
                out_basename = Path(value).name
                argv.extend([token, str(quarantine_dir / out_basename)])
            else:
                argv.extend([token, str(quarantine_dir)])
            index += 2
            continue
        if token.startswith("--out=") or token.startswith("--records-out="):
            flag, _, value = token.partition("=")
            if flag == "--out":
                out_basename = Path(value).name
                argv.append(f"--out={quarantine_dir / out_basename}")
            else:
                argv.append(f"--records-out={quarantine_dir}")
            index += 1
            continue
        argv.append(token)
        index += 1
    argv.extend(["--certificate-digest", activation_digest])
    return argv, out_basename


def _build_default_compare(anchor_path: Path, out_basename: str | None):
    """Documented default comparator (unpinned by tests): load the frozen
    anchor from ``anchor_path`` and the regenerated export from the
    quarantine (the remapped ``--out`` basename), then run
    ``phase4.compare_parity``."""
    if out_basename is None:
        raise LaunchRefusal(
            "certificate command carries no --out flag — the default"
            " comparator cannot locate the regenerated export (R-081)"
        )

    def compare(quarantine_dir: Path) -> dict[str, Any]:
        anchor = schema.parse_json_bytes_strict(
            schema.read_regular_file_bytes(Path(anchor_path))
        )
        regenerated = schema.parse_json_bytes_strict(
            schema.read_regular_file_bytes(Path(quarantine_dir) / out_basename)
        )
        return phase4.compare_parity(anchor, regenerated)

    return compare


def _write_stop_report(quarantine_dir: Path, payload: dict[str, Any]) -> None:
    quarantine_dir = Path(quarantine_dir)
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    report_path = quarantine_dir / STOP_REPORT_NAME
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def validate_and_launch(
    config: dict[str, Any],
    *,
    run_git: Any = None,
    launch: Any = None,
    compare: Any = None,
    now: Any = None,
) -> dict[str, Any]:
    """Single-use, fail-closed launch of the ONE authorized run (R-081).

    Pre-launch refusal classes, in execution order (``launch`` NEVER fires
    on any refusal): (1) certificate bytes sha256 != activation digest —
    checked FIRST, before parsing; (2) ``ready`` not identically True
    (bool-safe); (6) ambient ``MODAL_HOST_GIT_STATUS``/``_GIT_COMMIT``
    present at all; (3) live ``git rev-parse HEAD`` != certificate commit;
    (4) live tree != certificate tree; (5) live TRACKED-dirty state
    (untracked-only ``??`` lines are the signed tracked-clean +
    untracked-disclosure convention and do NOT refuse); (7) snapshot
    re-verification failure; (9) pre-existing quarantine/promote
    destination (stale workspace — checked BEFORE the ledger so a
    workspace mistake never burns the single-use exception); (8) ledger
    create-once via ``os.open(O_CREAT|O_EXCL)``, recording the activation
    digest BEFORE launch.

    Post-launch: nonzero exit -> ``RunFailed`` + STOP report in the intact
    quarantine, nothing promoted; zero exit -> comparator invoked
    MANDATORILY; PASS -> single atomic ``os.rename`` quarantine ->
    promote_to; FAIL -> ``RunFailed`` + STOP report + quarantine intact.
    """
    if not isinstance(config, dict):
        raise LaunchRefusal("launcher config must be an object (R-081)")
    missing = sorted(k for k in LAUNCHER_CONFIG_KEYS if k not in config)
    if missing:
        raise LaunchRefusal(
            f"launcher config missing key(s) {missing} (R-081)"
        )
    if run_git is None:
        run_git = _default_run_git
    if launch is None:
        launch = _default_launch
    if now is None:
        now = _utc_now_iso

    certificate_path = Path(config["certificate_path"])
    activation_digest = str(config["activation_digest"])
    # F-4: resolve the workspace paths at entry — a relative quarantine
    # would otherwise split between the launcher's cwd (mkdir/promote) and
    # the child's cwd (the repo root), stranding the run's outputs.
    quarantine_dir = Path(config["quarantine_dir"]).resolve()
    promote_to = Path(config["promote_to"]).resolve()
    ledger_path = Path(config["ledger_path"]).resolve()

    # (1) Activation digest over the RAW BYTES — before any parse, before
    # any semantic check (bytes first, semantics second).
    try:
        certificate_bytes = schema.read_regular_file_bytes(certificate_path)
    except schema.ColmAimsError as exc:
        raise LaunchRefusal(
            f"certificate digest check failed — certificate bytes"
            f" unreadable: {exc} (R-081)"
        ) from exc
    observed_digest = hashlib.sha256(certificate_bytes).hexdigest()
    if observed_digest != activation_digest:
        raise LaunchRefusal(
            f"certificate digest mismatch: certificate bytes sha256"
            f" {observed_digest} != activation digest {activation_digest}"
            " (R-081)"
        )

    # (2) Hardened parse + ready identity (bool-safe: 1 is not True).
    try:
        certificate = schema.parse_json_bytes_strict(certificate_bytes)
    except (schema.ColmAimsError, ValueError) as exc:
        raise LaunchRefusal(
            f"certificate parse failed: {exc} (R-081)"
        ) from exc
    if not isinstance(certificate, dict):
        raise LaunchRefusal(
            "certificate must be a JSON object (R-081)"
        )
    if certificate.get("ready") is not True:
        raise LaunchRefusal(
            "certificate ready is not identically True (found"
            f" {certificate.get('ready')!r}) — a non-ready certificate"
            " never launches (R-081)"
        )

    components = certificate.get("components")
    components = components if isinstance(components, dict) else {}
    repo = components.get("repo")
    repo = repo if isinstance(repo, dict) else {}
    certificate_commit = repo.get("commit")
    certificate_tree = repo.get("tree_sha256")
    environment = components.get("environment")
    environment = environment if isinstance(environment, dict) else {}
    command = environment.get("command")
    if (
        not isinstance(command, list)
        or not command
        or not all(isinstance(token, str) and token for token in command)
    ):
        raise LaunchRefusal(
            "certificate components.environment.command missing or"
            " malformed — argv must be composed FROM the certificate"
            " (R-081)"
        )

    # (6) Ambient provenance overrides: PRESENCE refuses (even empty) —
    # checked before any git output is trusted.
    for var in AMBIENT_OVERRIDE_VARS:
        if var in os.environ:
            raise LaunchRefusal(
                f"ambient provenance override {var} is present in the"
                " environment (MODAL_HOST overrides are refused even when"
                " empty — provenance laundering) (R-081)"
            )

    # (3) Live commit == certified commit (runner-sourced).
    live_commit = str(run_git(["git", "rev-parse", "HEAD"])).strip()
    if live_commit != certificate_commit:
        raise LaunchRefusal(
            f"live repository commit {live_commit!r} != certificate commit"
            f" {certificate_commit!r} (R-081)"
        )
    # (4) Live tree == certified tree.
    live_tree = str(run_git(["git", "rev-parse", "HEAD^{tree}"])).strip()
    if live_tree != certificate_tree:
        raise LaunchRefusal(
            f"live repository tree {live_tree!r} != certificate tree"
            f" {certificate_tree!r} (R-081)"
        )
    # (5) Live TRACKED-dirty state; untracked-only "??" porcelain lines are
    # the signed tracked-clean + untracked-disclosure convention.
    status_output = str(
        run_git(["git", "status", "--porcelain", "--untracked-files=no"])
    )
    tracked_lines = [
        line
        for line in status_output.splitlines()
        if line.strip() and not line.startswith("??")
    ]
    if tracked_lines:
        raise LaunchRefusal(
            "live repository is tracked-dirty"
            f" ({tracked_lines[:5]!r}) — the certified clean-state proof no"
            " longer holds (R-081)"
        )

    # (7) Snapshot re-verification against the frozen manifest.
    try:
        manifest = phase4.load_model_snapshot_manifest(
            Path(config["snapshot_manifest_path"])
        )
    except schema.ColmAimsError as exc:
        raise LaunchRefusal(
            f"snapshot manifest failed to load: {exc} (R-081)"
        ) from exc
    snapshot_dirs = config["snapshot_dirs"]
    snapshot_dirs = snapshot_dirs if isinstance(snapshot_dirs, dict) else {}
    for role in sorted(phase4.SNAPSHOT_ROLES):
        snapshot_dir = snapshot_dirs.get(role)
        if snapshot_dir is None:
            raise LaunchRefusal(
                f"snapshot directory for role {role!r} is not configured"
                " (R-081)"
            )
        try:
            phase4.verify_snapshot_dir(
                manifest["roles"][role], Path(snapshot_dir)
            )
        except schema.ColmAimsError as exc:
            raise LaunchRefusal(
                f"snapshot re-verification failed for role {role!r}: {exc}"
                " (R-081)"
            ) from exc

    # Argv/env composition + default comparator resolution happen BEFORE
    # the ledger so every composition defect refuses without consuming the
    # single-use exception.
    argv, out_basename = _compose_argv(
        command, quarantine_dir, activation_digest
    )

    # (F-1, R-082) Staged inputs must live OUTSIDE the repository tree —
    # an in-repo untracked staged file passes every other gate, then trips
    # the producer's committed-writer git-pathspec guard AFTER scoring and
    # burns the single-use exception (the P0-1 trap, the exact defect of
    # rejected certificate 8731ad00). Refused here, pre-ledger, from BOTH
    # sources: the certificate's staged_inputs component and every staged
    # path the composed argv would hand the child (relative forms resolve
    # against the child's cwd, the repo root).
    staged_candidates: list[str] = []
    for entry in components.get("staged_inputs") or []:
        if isinstance(entry, dict) and entry.get("path"):
            staged_candidates.append(str(entry["path"]))
    for i, token in enumerate(argv):
        if token == "--staged-input" and i + 1 < len(argv):
            staged_candidates.append(
                str(argv[i + 1]).partition("=")[2].rpartition(":")[0]
            )
        elif token.startswith("--staged-input="):
            staged_candidates.append(
                token.partition("=")[2].partition("=")[2].rpartition(":")[0]
            )
        elif token == "--calibration" and i + 1 < len(argv):
            staged_candidates.append(str(argv[i + 1]))
    for raw_path in staged_candidates:
        if not raw_path:
            continue
        resolved = Path(raw_path)
        if not resolved.is_absolute():
            resolved = _REPO_ROOT / resolved
        if schema.resolves_inside(resolved, _REPO_ROOT):
            raise LaunchRefusal(
                f"staged input {raw_path!r} resolves inside the repository"
                " tree — staged inputs live OUTSIDE the repo (identity by"
                " hash, never location; R-082, the P0-1 ledger-burn trap)"
            )
    if compare is None:
        compare = _build_default_compare(
            Path(config["anchor_path"]), out_basename
        )
    child_env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(AMBIENT_ENV_PREFIX)
    }
    child_env.update(LAUNCH_ENV_PINS)

    # (9) Workspace — fully materialized BEFORE the ledger (F-2: no
    # workspace defect may consume the single-use exception; the mkdir's
    # exist_ok=False doubles as the staleness check, and an unwritable
    # parent refuses here instead of burning the ledger).
    if promote_to.exists():
        raise LaunchRefusal(
            f"promote destination {promote_to} already exists — stale"
            " workspace; promotion is a single atomic rename (R-081)"
        )
    if not promote_to.parent.is_dir():
        raise LaunchRefusal(
            f"promote destination parent {promote_to.parent} does not exist"
            " — pre-flight refused so a PASS can always promote (R-081)"
        )
    try:
        quarantine_dir.mkdir(parents=True, exist_ok=False)
        _created_quarantine = True
    except FileExistsError as exc:
        raise LaunchRefusal(
            f"quarantine directory {quarantine_dir} already exists — stale"
            " workspace; the run writes into a FRESH quarantine (R-081)"
        ) from exc
    except OSError as exc:
        raise LaunchRefusal(
            f"quarantine directory {quarantine_dir} cannot be created"
            f" ({exc.__class__.__name__}) — workspace refused pre-ledger"
            " (R-081/F-2)"
        ) from exc
    if quarantine_dir.stat().st_dev != promote_to.parent.stat().st_dev:
        quarantine_dir.rmdir()
        raise LaunchRefusal(
            "quarantine and promote destination live on different devices —"
            " os.rename cannot promote atomically across devices (R-081)"
        )

    # (8) Single-use consumption: CREATE-ONCE ledger via O_CREAT|O_EXCL,
    # recording the activation digest BEFORE launch.
    ledger_payload = (
        json.dumps(
            {
                "activation_digest": activation_digest,
                "certificate_path": str(certificate_path),
                "certificate_commit": certificate_commit,
                "certificate_tree": certificate_tree,
                "argv": argv,
                "consumed_at": now(),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
        + "\n"
    )
    try:
        ledger_fd = os.open(
            str(ledger_path),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o644,
        )
    except FileExistsError as exc:
        # Ledger refusals stay side-effect-free: remove the quarantine this
        # call just created (empty by construction at this point).
        if _created_quarantine:
            quarantine_dir.rmdir()
        raise LaunchRefusal(
            f"exception ledger {ledger_path} already exists — the"
            " single-use exception was already consumed; no second run"
            " without a new recorded amendment (R-081)"
        ) from exc
    except OSError as exc:
        if _created_quarantine:
            quarantine_dir.rmdir()
        raise LaunchRefusal(
            f"exception ledger {ledger_path} is unwritable"
            f" ({exc.__class__.__name__}) (R-081)"
        ) from exc
    try:
        os.write(ledger_fd, ledger_payload.encode("utf-8"))
    finally:
        os.close(ledger_fd)

    # Launch EXACTLY once. A crash inside the launch callable still gets a
    # STOP report (F-3): the ledger is consumed, so the triage artifact
    # must exist precisely on the messiest failures.
    try:
        exit_code = launch(list(argv), dict(child_env))
    except Exception as exc:
        _write_stop_report(
            quarantine_dir,
            {
                "reason": "launch_crash",
                "error": f"{exc.__class__.__name__}: {exc}",
                "activation_digest": activation_digest,
                "argv": argv,
                "stopped_at": now(),
            },
        )
        raise RunFailed(
            f"producer launch crashed ({exc.__class__.__name__}) —"
            f" quarantine left intact at {quarantine_dir}, STOP report"
            " written, nothing promoted (R-081)"
        ) from exc
    if type(exit_code) is not int or exit_code != 0:
        _write_stop_report(
            quarantine_dir,
            {
                "reason": "nonzero_exit",
                "exit_code": exit_code,
                "activation_digest": activation_digest,
                "argv": argv,
                "stopped_at": now(),
            },
        )
        raise RunFailed(
            f"producer run exited nonzero ({exit_code!r}) — quarantine left"
            f" intact at {quarantine_dir}, STOP report written, nothing"
            " promoted (R-081)"
        )

    # Mandatory comparator on a zero exit. A comparator crash gets a STOP
    # report too (F-3) — fail-closed with the triage artifact present.
    try:
        result = compare(quarantine_dir)
    except Exception as exc:
        _write_stop_report(
            quarantine_dir,
            {
                "reason": "comparator_crash",
                "error": f"{exc.__class__.__name__}: {exc}",
                "activation_digest": activation_digest,
                "stopped_at": now(),
            },
        )
        raise RunFailed(
            f"parity comparator crashed ({exc.__class__.__name__}) —"
            f" promotion blocked; quarantine left intact at"
            f" {quarantine_dir} with a STOP report (R-081)"
        ) from exc
    result = result if isinstance(result, dict) else {}
    if result.get("verdict") == "PASS":
        # Single atomic promotion.
        os.rename(str(quarantine_dir), str(promote_to))
        return {
            "promoted_to": str(promote_to),
            "activation_digest": activation_digest,
            "exit_code": 0,
            "verdict": "PASS",
            "argv": argv,
        }
    _write_stop_report(
        quarantine_dir,
        {
            "reason": "parity_comparator_fail",
            "verdict": result.get("verdict"),
            "checked": result.get("checked"),
            "failures": result.get("failures"),
            "activation_digest": activation_digest,
            "stopped_at": now(),
        },
    )
    raise RunFailed(
        f"parity comparator verdict {result.get('verdict')!r} — promotion"
        f" blocked; quarantine left intact at {quarantine_dir} with a STOP"
        " report (R-081)"
    )
