"""Focused, model-free tests for the executable Phase-4 launcher."""
from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from reproducibility.colm_aims_2026 import phase4_launcher as launcher
from reproducibility.colm_aims_2026 import phase4_driver_d7b as driver
from reproducibility.colm_aims_2026 import phase4_finalize_release as finalizer


HOST_IDENTITY = {"os": "SyntheticOS 1.0 ()", "arch": "synthetic64"}
LOCK_TEXT = "example-package==1.0\n"
STAGED_DATA_FILES = {
    "eval_split": "test_dataset.json",
    "fit_split": "val_dataset.json",
    "mc_dataset": "mc_dataset.json",
    "answer_profiles": "answer_profiles.json",
    "build_metadata": "build_metadata.json",
    "split_metadata": "split_metadata.json",
}
OPERATOR_STAGED_LABELS = (
    "fit_split",
    "mc_dataset",
    "answer_profiles",
    "build_metadata",
    "split_metadata",
)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_accepted_certificate(path: Path, data: bytes) -> None:
    """Write a synthetic certificate bundle with an exact acceptance marker."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    (path.parent / launcher.CERTIFICATE_GENERATION_SUMMARY_NAME).write_bytes(
        b"{}\n"
    )
    tree_sha256 = finalizer._directory_tree_sha256(path.parent)
    finalizer._accepted_marker_path(path.parent).write_bytes(
        finalizer._accepted_marker_bytes(path.parent, tree_sha256)
    )


def _refresh_accepted_certificate(path: Path, data: bytes) -> None:
    """Replace a pre-launch test fixture and rebind its synthetic marker."""
    path.write_bytes(data)
    tree_sha256 = finalizer._directory_tree_sha256(path.parent)
    finalizer._accepted_marker_path(path.parent).write_bytes(
        finalizer._accepted_marker_bytes(path.parent, tree_sha256)
    )


def _json_config(tmp_path: Path) -> dict[str, object]:
    return {
        "certificate_path": "certificate.json",
        "activation_digest": "a" * 64,
        "quarantine_dir": "quarantine",
        "promote_to": "promoted",
        "ledger_path": "ledger.json",
        "snapshot_manifest_path": "manifest.json",
        "snapshot_dirs": {
            "primary_scorer": "snapshots/primary",
            "disjoint_selector": "snapshots/disjoint",
        },
        "anchor_path": "anchor.json",
    }


def _write_config(tmp_path: Path, payload: object) -> Path:
    path = tmp_path / "launcher_config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_config_loader_requires_an_exact_closed_json_object(tmp_path):
    valid = _json_config(tmp_path)
    loaded = launcher._load_launcher_config(_write_config(tmp_path, valid))
    assert loaded == valid

    extra = dict(valid, ignored_key=True)
    with pytest.raises(launcher.LaunchRefusal, match="unknown key"):
        launcher._load_launcher_config(_write_config(tmp_path, extra))

    missing = dict(valid)
    missing.pop("anchor_path")
    with pytest.raises(launcher.LaunchRefusal, match="missing key"):
        launcher._load_launcher_config(_write_config(tmp_path, missing))

    with pytest.raises(launcher.LaunchRefusal, match="JSON object"):
        launcher._load_launcher_config(_write_config(tmp_path, []))


def test_cli_and_contract_disclose_exact_process_trust_boundary():
    help_text = launcher._build_parser().format_help()
    assert "not a sandbox" in help_text
    assert "no surviving producer descendants" in help_text

    repo_root = Path(launcher.__file__).resolve().parents[2]
    surfaces = {
        "spec": repo_root
        / ".correctless/specs/camera-ready-aims-evidence-2.md",
        "readme": repo_root / "reproducibility/colm_aims_2026/README.md",
        "feature": repo_root
        / "docs/features/camera-ready-aims-evidence-v2.md",
        "decision": repo_root
        / "decision_record_phase4_process_trust_boundary_2026-08-26.md",
    }
    for label, path in surfaces.items():
        text = path.read_text("utf-8")
        assert "not a sandbox" in text, label
        assert "no surviving producer descendants" in text, label
        assert launcher.phase4.PHASE4_PROCESS_TRUST_MODEL_ID in text, label

    assert "not a sandbox" in (launcher.__doc__ or "")
    assert "not ACL, principal, or process isolation" in (
        launcher._materialize_private_promotion_tree.__doc__ or ""
    )
    architecture = (repo_root / ".correctless/ARCHITECTURE.md").read_text(
        "utf-8"
    )
    assert "published artifacts are immutable once written" not in architecture
    assert "not filesystem ACL immutability" in architecture
    closure_source = (
        repo_root / "reproducibility/colm_aims_2026/closure.py"
    ).read_text("utf-8")
    assert "staged immutable envelope" not in closure_source
    assert "not an OS-level immutability" in closure_source


def test_launcher_and_driver_share_the_pending_guard_name():
    assert (
        launcher.ACCEPTANCE_PENDING_NAME == driver.ACCEPTANCE_PENDING_NAME
    )


def test_config_loader_closes_snapshot_roles_and_path_types(tmp_path):
    invalid_roles = _json_config(tmp_path)
    invalid_roles["snapshot_dirs"] = {
        "primary_scorer": "snapshots/primary",
        "disjoint_selector": "snapshots/disjoint",
        "shadow": "snapshots/shadow",
    }
    with pytest.raises(launcher.LaunchRefusal, match="unknown role"):
        launcher._load_launcher_config(
            _write_config(tmp_path, invalid_roles)
        )

    invalid_path = _json_config(tmp_path)
    invalid_path["ledger_path"] = None
    with pytest.raises(launcher.LaunchRefusal, match="ledger_path"):
        launcher._load_launcher_config(_write_config(tmp_path, invalid_path))


def test_main_prints_indented_json_on_pass(tmp_path, monkeypatch, capsys):
    config_path = _write_config(tmp_path, _json_config(tmp_path))
    expected = {
        "verdict": "PASS",
        "promoted_to": "promoted",
        "exit_code": 0,
    }
    monkeypatch.setattr(
        launcher, "validate_and_launch", lambda config: expected
    )

    assert launcher.main(["--config", str(config_path)]) == launcher.EXIT_PASS
    captured = capsys.readouterr()
    assert json.loads(captured.out) == expected
    assert captured.out.startswith("{\n  ")
    assert captured.err == ""


@pytest.mark.parametrize(
    ("error", "exit_code", "label"),
    [
        (
            launcher.LaunchRefusal("certificate refused"),
            launcher.EXIT_LAUNCH_REFUSAL,
            "LaunchRefusal",
        ),
        (
            launcher.RunFailed("producer failed"),
            launcher.EXIT_RUN_FAILED,
            "RunFailed",
        ),
    ],
)
def test_main_returns_typed_nonzero_errors(
    tmp_path, monkeypatch, capsys, error, exit_code, label
):
    config_path = _write_config(tmp_path, _json_config(tmp_path))

    def refuse(_config):
        raise error

    monkeypatch.setattr(launcher, "validate_and_launch", refuse)
    assert launcher.main(["--config", str(config_path)]) == exit_code
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith(f"{label}:")


def test_python_m_interface_is_executable_without_loading_a_model(tmp_path):
    config_path = _write_config(tmp_path, [])
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "reproducibility.colm_aims_2026.phase4_launcher",
            "--config",
            str(config_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == launcher.EXIT_LAUNCH_REFUSAL
    assert "LaunchRefusal:" in completed.stderr
    assert completed.stdout == ""


def _runtime_fixture(
    tmp_path: Path,
    monkeypatch,
    *,
    data_dir: Path | None = None,
    data_dir_equals: bool = False,
):
    real_assemble_certificate = launcher.phase4.assemble_certificate
    monkeypatch.setattr(
        launcher.phase4,
        "assemble_certificate",
        lambda components: {
            "schema_version": launcher.phase4.CERT_SCHEMA_VERSION,
            "ready": True,
            "failing_checks": [],
            "components": components,
        },
    )
    executable = tmp_path / "synthetic-python"
    executable.write_bytes(b"synthetic executable\n")

    manifest_path = (
        launcher._REPO_ROOT / launcher.phase4.SNAPSHOT_MANIFEST_RELPATH
    ).resolve()
    manifest_bytes = manifest_path.read_bytes()
    snapshot_dirs = {
        "primary_scorer": tmp_path / "primary_snapshot",
        "disjoint_selector": tmp_path / "disjoint_snapshot",
    }
    for snapshot_dir in snapshot_dirs.values():
        snapshot_dir.mkdir()

    anchor_path = (
        launcher._REPO_ROOT / launcher.phase4.PARITY_ANCHOR_RELPATH
    ).resolve()
    anchor_bytes = anchor_path.read_bytes()

    staged_dir = tmp_path / "staged_inputs"
    staged_dir.mkdir()
    staged_paths = {
        "calibration_train": staged_dir / "calibration_train.json",
        **{
            label: staged_dir / filename
            for label, filename in STAGED_DATA_FILES.items()
        },
    }
    staged_digests = {}
    for label, staged_path in staged_paths.items():
        staged_bytes = json.dumps({"label": label}).encode("utf-8") + b"\n"
        staged_path.write_bytes(staged_bytes)
        staged_digests[label] = _sha256(staged_bytes)

    eligibility_path = (
        launcher._REPO_ROOT / launcher.phase4.ELIGIBILITY_ARTIFACT_RELPATH
    ).resolve()
    eligibility_bytes = eligibility_path.read_bytes()
    staged_digests.update(launcher.phase4.R082_STAGED_INPUT_SHA256)
    eligible_keys = [f"item-{index:04d}" for index in range(2249)]
    loaded_eligibility = {
        "eligible_keys": eligible_keys,
        "horizon_map": {key: 2 for key in eligible_keys},
        "pairing_population_keyset_sha256": (
            launcher.phase4.ELIGIBILITY_KEYSET_SHA256
        ),
        "horizon_map_sha256": (
            launcher.phase4.ELIGIBILITY_HORIZON_MAP_SHA256
        ),
        "derived_from": {
            "test_dataset_sha256": (
                launcher.phase4.ELIGIBILITY_TEST_DATASET_SHA256
            )
        },
    }
    monkeypatch.setattr(
        launcher.phase4,
        "load_pairing_eligibility",
        lambda _path: loaded_eligibility,
    )
    real_staged_hash = launcher._sha256_regular_file

    def fixture_staged_hash(path, *, label):
        candidate = Path(path).resolve()
        for staged_label, staged_path in staged_paths.items():
            if candidate == staged_path.resolve():
                return launcher.phase4.R082_STAGED_INPUT_SHA256[staged_label]
        return real_staged_hash(path, label=label)

    monkeypatch.setattr(
        launcher, "_sha256_regular_file", fixture_staged_hash
    )

    if data_dir is None:
        data_dir = staged_dir
    command = [
        str(executable),
        "scripts/stopdff_fair_qa_retest.py",
        "--snapshot-manifest",
        str(manifest_path),
        "--primary-model-path",
        str(snapshot_dirs["primary_scorer"]),
        "--disjoint-model-path",
        str(snapshot_dirs["disjoint_selector"]),
    ]
    if data_dir_equals:
        command.append(f"--data-dir={data_dir}")
    else:
        command.extend(["--data-dir", str(data_dir)])
    command.extend(
        [
            "--calibration",
            str(staged_paths["calibration_train"]),
            "--fit-split",
            "val",
            "--eval-split",
            "test",
            "--reward-schedule",
            "power_mark",
            "--qa-arms",
            "idealized,krandom,khard,kdisjoint,klex",
            "--calibrations",
            "shared,performat",
            "--num-bootstrap",
            "1000",
            "--n-test",
            "0",
            "--n-val",
            "0",
            "--seed",
            "1",
            "--eligibility",
            str(eligibility_path),
        ]
    )
    for label in OPERATOR_STAGED_LABELS:
        command.extend(
            [
                "--staged-input",
                f"{label}={staged_paths[label]}:{staged_digests[label]}",
            ]
        )
    command.extend(
        [
            "--records-out",
            "phase4_run_output",
            "--out",
            "phase4_run_output/stopdff_fair_qa_regenerated.json",
        ]
    )

    quarantine_dir = tmp_path / "quarantine"
    promote_to = tmp_path / "promoted"
    ledger_path = tmp_path / "ledger.json"

    commit = "1" * 40
    tree = "2" * 40
    certificate = {
        "schema_version": launcher.phase4.CERT_SCHEMA_VERSION,
        "ready": True,
        "failing_checks": [],
        "components": {
            "repo": {
                "commit": commit,
                "tree_sha256": tree,
                "dirty": False,
                "untracked_disclosure": [],
                "root_realpath": str(launcher._REPO_ROOT.resolve()),
            },
            "content_hashes": {
                key: {
                    "artifact_path": str(
                        (launcher._REPO_ROOT / relpath).resolve()
                    ),
                    "sha256": _sha256(
                        (launcher._REPO_ROOT / relpath).read_bytes()
                    ),
                }
                for key, relpath in (
                    launcher.phase4.CONTENT_HASH_RELPATHS.items()
                )
            },
            "environment": {
                "command": command,
                "interpreter_realpath": str(executable),
                "os": HOST_IDENTITY["os"],
                "arch": HOST_IDENTITY["arch"],
                "environment_lock_sha256": _sha256(
                    LOCK_TEXT.encode("utf-8")
                ),
                "seeds": [1],
                "pythonhashseed": "0",
                "thread_settings": dict(
                    launcher.phase4.PHASE4_THREAD_SETTINGS
                ),
                "quarantine_dir": str(quarantine_dir),
                "promote_to": str(promote_to),
                "exception_ledger_path": str(ledger_path),
            },
            "eligibility": {
                "digest": loaded_eligibility[
                    "pairing_population_keyset_sha256"
                ],
                "horizon_map_sha256": loaded_eligibility[
                    "horizon_map_sha256"
                ],
                "artifact_path": str(eligibility_path),
                "artifact_sha256": _sha256(eligibility_bytes),
                "test_dataset_sha256": staged_digests["eval_split"],
            },
            "parity": {"anchor_sha256": _sha256(anchor_bytes)},
            "qa012": {"manifest_sha256": launcher._QA012_SHA256},
            "snapshots": {
                "artifact_path": str(manifest_path),
                "artifact_sha256": launcher.phase4.SNAPSHOT_MANIFEST_SHA256,
                "primary_scorer": dict(
                    launcher.phase4.EXPECTED_SNAPSHOT_IDENTITIES[
                        "primary_scorer"
                    ]
                ),
                "disjoint_selector": dict(
                    launcher.phase4.EXPECTED_SNAPSHOT_IDENTITIES[
                        "disjoint_selector"
                    ]
                ),
            },
            "staged_inputs": [
                {
                    "label": label,
                    "path": str(staged_paths[label]),
                    "expected_sha256": staged_digests[label],
                    "observed_sha256": staged_digests[label],
                }
                for label in (
                    "calibration_train",
                    "eval_split",
                    "fit_split",
                    "mc_dataset",
                    "answer_profiles",
                    "build_metadata",
                    "split_metadata",
                )
            ],
        },
    }
    certificate_bytes = (
        json.dumps(certificate, sort_keys=True) + "\n"
    ).encode("utf-8")
    certificate_path = tmp_path / "certificate" / "certificate.json"
    _write_accepted_certificate(certificate_path, certificate_bytes)

    config = {
        "certificate_path": certificate_path,
        "activation_digest": _sha256(certificate_bytes),
        "quarantine_dir": quarantine_dir,
        "promote_to": promote_to,
        "ledger_path": ledger_path,
        "snapshot_manifest_path": manifest_path,
        "snapshot_dirs": snapshot_dirs,
        "anchor_path": anchor_path,
    }
    manifest = {
        "roles": {
            role: dict(identity)
            for role, identity in (
                launcher.phase4.EXPECTED_SNAPSHOT_IDENTITIES.items()
            )
        }
    }
    monkeypatch.setattr(
        launcher.phase4,
        "load_model_snapshot_manifest",
        lambda _path: manifest,
    )
    monkeypatch.setattr(
        launcher.phase4,
        "verify_snapshot_dir",
        lambda _entry, _path: None,
    )
    real_capture_regular_file = launcher._capture_regular_file

    def capture_fixture_file(
        source,
        destination,
        *,
        expected_sha256,
        label,
        expected_size=None,
    ):
        candidate = Path(source).resolve()
        if any(
            candidate == staged_path.resolve()
            for staged_path in staged_paths.values()
        ):
            destination = Path(destination)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(Path(source).read_bytes())
            return None
        return real_capture_regular_file(
            source,
            destination,
            expected_sha256=expected_sha256,
            label=label,
            expected_size=expected_size,
        )

    monkeypatch.setattr(
        launcher, "_capture_regular_file", capture_fixture_file
    )
    monkeypatch.setattr(
        launcher,
        "_capture_snapshot_role",
        lambda _entry, _source, destination: Path(destination).mkdir(
            parents=True, exist_ok=False
        ),
    )

    def run_git(command_argv):
        if command_argv == ["git", "rev-parse", "HEAD"]:
            return commit + "\n"
        if command_argv == ["git", "rev-parse", "HEAD^{tree}"]:
            return tree + "\n"
        if command_argv == [
            "git",
            "status",
            "--porcelain",
            "--untracked-files=no",
        ]:
            return ""
        if command_argv == list(launcher._GIT_UNTRACKED_STATUS_COMMAND):
            return ""
        raise AssertionError(command_argv)

    probes = {
        "run_git": run_git,
        "resolve_executable": lambda _token: executable,
        "host_identity": lambda: dict(HOST_IDENTITY),
        "probe_environment_lock": lambda _interpreter: LOCK_TEXT,
        "verified_eligibility": loaded_eligibility,
        "real_assemble_certificate": real_assemble_certificate,
        "real_staged_hash": real_staged_hash,
    }
    return config, certificate, staged_paths["fit_split"], probes


def _call_launcher(config, probes, *, launch=None, compare=None, **kwargs):
    supplied_launch = launch or (lambda _argv, _env: 0)

    def launch_with_bound_outputs(argv, env):
        exit_code = supplied_launch(argv, env)
        records_dir = Path(config["quarantine_dir"]) / "records"
        if exit_code == 0 and not records_dir.exists():
            _write_valid_default_outputs(argv, probes["verified_eligibility"])
        return exit_code

    return launcher.validate_and_launch(
        config,
        run_git=probes["run_git"],
        resolve_executable=probes["resolve_executable"],
        host_identity=probes["host_identity"],
        probe_environment_lock=probes["probe_environment_lock"],
        launch=launch_with_bound_outputs,
        compare=compare
        or (lambda _quarantine: {"verdict": "PASS", "checked": 194}),
        **kwargs,
    )


def _call_default_launcher(config, probes, *, launch, **kwargs):
    return launcher.validate_and_launch(
        config,
        run_git=probes["run_git"],
        resolve_executable=probes["resolve_executable"],
        host_identity=probes["host_identity"],
        probe_environment_lock=probes["probe_environment_lock"],
        launch=launch,
        **kwargs,
    )


def _rewrite_ready_certificate(config, certificate) -> None:
    """Persist a deliberately hand-crafted ready:true certificate."""
    assert certificate["ready"] is True
    certificate_bytes = (
        json.dumps(certificate, sort_keys=True) + "\n"
    ).encode("utf-8")
    _refresh_accepted_certificate(
        Path(config["certificate_path"]), certificate_bytes
    )
    config["activation_digest"] = _sha256(certificate_bytes)


def test_launcher_requires_positive_certificate_acceptance(
    tmp_path, monkeypatch
):
    config, _certificate, _fit_split, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    finalizer._accepted_marker_path(
        Path(config["certificate_path"]).parent
    ).unlink()
    launches = []

    with pytest.raises(launcher.LaunchRefusal, match="positive acceptance"):
        _call_launcher(
            config,
            probes,
            launch=lambda argv, env: launches.append((argv, env)) or 0,
        )

    assert launches == []
    assert not Path(config["ledger_path"]).exists()


def test_launcher_rejects_pending_certificate_publication(
    tmp_path, monkeypatch
):
    config, _certificate, _fit_split, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    certificate_dir = Path(config["certificate_path"]).parent
    finalizer._pending_guard_path(certificate_dir).write_bytes(
        finalizer._pending_guard_bytes(certificate_dir)
    )
    launches = []

    with pytest.raises(launcher.LaunchRefusal, match="pending"):
        _call_launcher(
            config,
            probes,
            launch=lambda argv, env: launches.append((argv, env)) or 0,
        )

    assert launches == []
    assert not Path(config["ledger_path"]).exists()


def test_launcher_rejects_certificate_tree_not_bound_by_marker(
    tmp_path, monkeypatch
):
    config, _certificate, _fit_split, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    summary_path = (
        Path(config["certificate_path"]).parent
        / launcher.CERTIFICATE_GENERATION_SUMMARY_NAME
    )
    summary_path.write_bytes(b'{"changed":true}\n')
    launches = []

    with pytest.raises(launcher.LaunchRefusal, match="exact tree"):
        _call_launcher(
            config,
            probes,
            launch=lambda argv, env: launches.append((argv, env)) or 0,
        )

    assert launches == []
    assert not Path(config["ledger_path"]).exists()


def _historical_cell(cell_id: str) -> str:
    reference, calibration = cell_id.rsplit("__", 1)
    if calibration == "format_specific":
        calibration = "performat"
    return f"{reference}+{calibration}"


def _write_valid_default_outputs(
    argv: list[str],
    eligibility: dict[str, object],
    *,
    mutate=None,
) -> None:
    """Write a complete model-free ten-cell producer export."""
    out_path = Path(argv[argv.index("--out") + 1])
    records_parent = Path(argv[argv.index("--records-out") + 1])
    activation_digest = argv[argv.index("--certificate-digest") + 1]
    records_root = records_parent / "records"
    records_root.mkdir()

    horizon_map = eligibility["horizon_map"]
    rows = [
        {
            "item_key": item_key,
            "trajectory_horizon": horizon_map[item_key],
            "mc_event_status": "FINITE_STOP",
            "mc_stop_step": 0,
            "mc_terminal_imputation": "NONE",
            "ref_event_status": "FINITE_STOP",
            "ref_stop_step": 0,
            "ref_terminal_imputation": "NONE",
        }
        for item_key in eligibility["eligible_keys"]
    ]
    record_bytes = b"".join(
        (
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        for row in rows
    )
    record_digest = _sha256(record_bytes)
    exported_records = {}
    for cell_id in launcher.schema.CELL_IDS:
        (records_root / f"{cell_id}.jsonl").write_bytes(record_bytes)
        exported_records[cell_id] = {
            "path": f"records/{cell_id}.jsonl",
            "sha256": record_digest,
            "n_items": launcher.schema.EXPECTED_COMPLETE_PAIRS,
            "historical_cell": _historical_cell(cell_id),
            "policy": "dp",
        }
    payload = {
        "metadata": {
            "phase4": {
                "certificate_digest": activation_digest,
                "exported_records": exported_records,
            }
        },
        "regenerated": True,
    }
    if mutate is not None:
        mutate(payload, records_root)
    out_path.write_text(json.dumps(payload), encoding="utf-8")


def test_malicious_ready_certificate_cannot_omit_component_label(
    tmp_path, monkeypatch
):
    config, certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    staged_inputs = certificate["components"]["staged_inputs"]
    certificate["components"]["staged_inputs"] = [
        entry
        for entry in staged_inputs
        if entry["label"] != "build_metadata"
    ]
    _rewrite_ready_certificate(config, certificate)

    launch_calls = []
    with pytest.raises(
        launcher.LaunchRefusal,
        match="missing required label 'build_metadata'",
    ):
        _call_launcher(
            config,
            probes,
            launch=lambda argv, env: launch_calls.append((argv, env)),
        )
    assert launch_calls == []
    assert not Path(config["ledger_path"]).exists()
    assert not Path(config["quarantine_dir"]).exists()


def test_malicious_ready_certificate_cannot_omit_operator_binding(
    tmp_path, monkeypatch
):
    config, certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    command = certificate["components"]["environment"]["command"]
    index = next(
        index
        for index, token in enumerate(command)
        if token == "--staged-input"
        and command[index + 1].startswith("build_metadata=")
    )
    del command[index : index + 2]
    _rewrite_ready_certificate(config, certificate)

    with pytest.raises(
        launcher.LaunchRefusal,
        match="missing operator label 'build_metadata'",
    ):
        _call_launcher(config, probes)
    assert not Path(config["ledger_path"]).exists()
    assert not Path(config["quarantine_dir"]).exists()


def test_malicious_ready_certificate_cannot_drift_split_selector(
    tmp_path, monkeypatch
):
    config, certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    command = certificate["components"]["environment"]["command"]
    command[command.index("--fit-split") + 1] = "alternate"
    _rewrite_ready_certificate(config, certificate)

    with pytest.raises(launcher.LaunchRefusal, match="fit_split"):
        _call_launcher(config, probes)
    assert not Path(config["ledger_path"]).exists()
    assert not Path(config["quarantine_dir"]).exists()


def test_streaming_staged_hash_is_raw_binary_on_windows(tmp_path):
    raw = b"prefix\r\ncontrol-z:\x1a\x00binary:\xff\xfe\r\nsuffix\n"
    staged = tmp_path / "binary_staged_input.bin"
    staged.write_bytes(raw)
    assert launcher._sha256_regular_file(
        staged, label="binary staged input"
    ) == _sha256(raw)


@pytest.mark.parametrize("equals_spelling", [False, True])
def test_data_dir_inside_repo_refuses_pre_ledger_in_both_spellings(
    tmp_path, monkeypatch, equals_spelling
):
    config, _certificate, _staged_path, probes = _runtime_fixture(
        tmp_path,
        monkeypatch,
        data_dir=launcher._REPO_ROOT / "data" / "processed",
        data_dir_equals=equals_spelling,
    )
    launch_calls = []
    with pytest.raises(launcher.LaunchRefusal, match="R-082"):
        _call_launcher(
            config,
            probes,
            launch=lambda argv, env: launch_calls.append((argv, env)),
        )
    assert launch_calls == []
    assert not Path(config["ledger_path"]).exists()
    assert not Path(config["quarantine_dir"]).exists()


def test_runtime_interpreter_and_host_are_certificate_bound(
    tmp_path, monkeypatch
):
    config, _certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    probes["host_identity"] = lambda: {
        "os": HOST_IDENTITY["os"],
        "arch": "different-arch",
    }
    with pytest.raises(launcher.LaunchRefusal, match="live host arch"):
        _call_launcher(config, probes)
    assert not Path(config["ledger_path"]).exists()

    probes["host_identity"] = lambda: dict(HOST_IDENTITY)
    probes["resolve_executable"] = lambda _token: tmp_path / "other-python"
    with pytest.raises(launcher.LaunchRefusal, match="interpreter realpath"):
        _call_launcher(config, probes)
    assert not Path(config["ledger_path"]).exists()


def test_environment_lock_mismatch_refuses_pre_ledger(tmp_path, monkeypatch):
    config, _certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    probes["probe_environment_lock"] = lambda _interpreter: "drifted==2\n"
    with pytest.raises(launcher.LaunchRefusal, match="pip-freeze"):
        _call_launcher(config, probes)
    assert not Path(config["ledger_path"]).exists()


@pytest.mark.parametrize("variable", launcher.AMBIENT_PYTHON_INJECTION_VARS)
def test_ambient_python_import_injection_refuses_even_when_empty(
    tmp_path, monkeypatch, variable
):
    config, _certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    monkeypatch.setenv(variable, "")
    launch_calls = []
    environment_probe_calls = []
    probes["probe_environment_lock"] = lambda interpreter: (
        environment_probe_calls.append(interpreter)
    )

    with pytest.raises(launcher.LaunchRefusal, match=variable):
        _call_launcher(
            config,
            probes,
            launch=lambda argv, env: launch_calls.append((argv, env)),
        )
    assert launch_calls == []
    assert environment_probe_calls == []
    assert not Path(config["ledger_path"]).exists()
    assert not Path(config["quarantine_dir"]).exists()


def test_default_environment_probe_uses_certified_interpreter_and_lf_bytes(
    tmp_path, monkeypatch
):
    interpreter = tmp_path / "python"
    calls = []

    class Completed:
        stdout = b"alpha==1\r\nbeta==2\rgamma==3\n"

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return Completed()

    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    observed = launcher._default_probe_environment_lock(interpreter)
    assert observed == b"alpha==1\nbeta==2\ngamma==3\n"
    assert calls[0][0] == [str(interpreter), "-m", "pip", "freeze"]
    assert calls[0][1]["cwd"] == str(launcher._REPO_ROOT)
    assert calls[0][1]["capture_output"] is True
    assert calls[0][1]["check"] is True
    probe_env = calls[0][1]["env"]
    assert probe_env["PYTHONNOUSERSITE"] == "1"
    assert not any(
        key.upper().startswith("PYTHON") and key not in launcher.LAUNCH_ENV_PINS
        for key in probe_env
    )


def test_fresh_ledger_config_substitution_refuses_pre_ledger(
    tmp_path, monkeypatch
):
    config, _certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    certified_ledger = Path(config["ledger_path"])
    substituted_ledger = tmp_path / "fresh-ledger.json"
    config["ledger_path"] = substituted_ledger

    with pytest.raises(launcher.LaunchRefusal, match="exactly match"):
        _call_launcher(config, probes)
    assert not certified_ledger.exists()
    assert not substituted_ledger.exists()
    assert not Path(config["quarantine_dir"]).exists()


@pytest.mark.parametrize("defect", ["nested", "inside_repo"])
def test_certified_launch_workspace_topology_refuses_pre_ledger(
    tmp_path, monkeypatch, defect
):
    config, certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    environment = certificate["components"]["environment"]
    if defect == "nested":
        quarantine = tmp_path / "workspace"
        promote = quarantine / "promoted"
        environment["quarantine_dir"] = str(quarantine)
        environment["promote_to"] = str(promote)
        config["quarantine_dir"] = quarantine
        config["promote_to"] = promote
    else:
        quarantine = launcher._REPO_ROOT / "launcher-test-quarantine"
        environment["quarantine_dir"] = str(quarantine)
        config["quarantine_dir"] = quarantine
    _rewrite_ready_certificate(config, certificate)

    with pytest.raises(
        launcher.LaunchRefusal, match="launch-workspace refusal"
    ):
        _call_launcher(config, probes)
    assert not Path(config["ledger_path"]).exists()
    assert not quarantine.exists()


def test_missing_workspace_parent_refuses_pre_ledger(tmp_path, monkeypatch):
    config, certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    quarantine = tmp_path / "missing-parent" / "quarantine"
    certificate["components"]["environment"]["quarantine_dir"] = str(
        quarantine
    )
    config["quarantine_dir"] = quarantine
    _rewrite_ready_certificate(config, certificate)

    with pytest.raises(launcher.LaunchRefusal, match="parent .*does not"):
        _call_launcher(config, probes)
    assert not Path(config["ledger_path"]).exists()
    assert not quarantine.exists()


def test_workspace_symlink_redirect_refuses_pre_ledger(tmp_path, monkeypatch):
    config, _certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    promote_to = Path(config["promote_to"])
    try:
        promote_to.symlink_to(
            tmp_path / "missing-redirect-target", target_is_directory=True
        )
    except OSError as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")
    launch_calls = []

    with pytest.raises(launcher.LaunchRefusal, match="canonical resolved path"):
        _call_launcher(
            config,
            probes,
            launch=lambda argv, env: launch_calls.append((argv, env)),
        )
    assert launch_calls == []
    assert not Path(config["ledger_path"]).exists()
    assert not Path(config["quarantine_dir"]).exists()
    assert promote_to.is_symlink()


def test_eligibility_certificate_pins_are_recomputed_pre_ledger(
    tmp_path, monkeypatch
):
    config, certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    certificate["components"]["eligibility"]["digest"] = "a" * 64
    _rewrite_ready_certificate(config, certificate)

    with pytest.raises(
        launcher.LaunchRefusal,
        match="eligibility.digest.*canonical",
    ):
        _call_launcher(config, probes)
    assert not Path(config["ledger_path"]).exists()
    assert not Path(config["quarantine_dir"]).exists()


def test_substituted_command_eligibility_refuses_pre_ledger(
    tmp_path, monkeypatch
):
    config, certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    substituted = tmp_path / "substituted_eligibility.json"
    substituted.write_text("{}\n", encoding="utf-8")
    command = certificate["components"]["environment"]["command"]
    command[command.index("--eligibility") + 1] = str(substituted)
    _rewrite_ready_certificate(config, certificate)

    with pytest.raises(launcher.LaunchRefusal, match="--eligibility path"):
        _call_launcher(config, probes)
    assert not Path(config["ledger_path"]).exists()
    assert not Path(config["quarantine_dir"]).exists()


def test_self_consistent_substitute_eligibility_refuses_canonical_pin(
    tmp_path, monkeypatch
):
    config, certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    canonical = Path(
        certificate["components"]["eligibility"]["artifact_path"]
    )
    substitute = tmp_path / "substitute_eligibility.json"
    substitute.write_bytes(canonical.read_bytes())
    eligibility = certificate["components"]["eligibility"]
    eligibility["artifact_path"] = str(substitute)
    eligibility["artifact_sha256"] = _sha256(substitute.read_bytes())
    command = certificate["components"]["environment"]["command"]
    command[command.index("--eligibility") + 1] = str(substitute)
    _rewrite_ready_certificate(config, certificate)

    with pytest.raises(
        launcher.LaunchRefusal, match="canonical.*artifact"
    ):
        _call_launcher(config, probes)
    assert not Path(config["ledger_path"]).exists()
    assert not Path(config["quarantine_dir"]).exists()


def test_semantic_certificate_reassembly_refuses_handcrafted_ready(
    tmp_path, monkeypatch
):
    config, _certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    monkeypatch.setattr(
        launcher.phase4,
        "assemble_certificate",
        probes["real_assemble_certificate"],
    )
    with pytest.raises(
        launcher.LaunchRefusal, match="semantic reassembly is not ready"
    ):
        _call_launcher(config, probes)
    assert not Path(config["ledger_path"]).exists()
    assert not Path(config["quarantine_dir"]).exists()


def test_certificate_schema_version_requires_an_exact_integer(
    tmp_path, monkeypatch
):
    config, certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    certificate["schema_version"] = float(
        launcher.phase4.CERT_SCHEMA_VERSION
    )
    _rewrite_ready_certificate(config, certificate)

    with pytest.raises(launcher.LaunchRefusal, match="schema_version"):
        _call_launcher(config, probes)
    assert not Path(config["ledger_path"]).exists()
    assert not Path(config["quarantine_dir"]).exists()


def test_content_hashes_refuse_a_self_consistent_substituted_source(
    tmp_path, monkeypatch
):
    config, certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    entry = certificate["components"]["content_hashes"]["producer_sha256"]
    canonical = Path(entry["artifact_path"])
    substitute = tmp_path / "substituted_producer.py"
    substitute.write_bytes(canonical.read_bytes())
    entry["artifact_path"] = str(substitute)
    entry["sha256"] = _sha256(substitute.read_bytes())
    _rewrite_ready_certificate(config, certificate)

    with pytest.raises(
        launcher.LaunchRefusal, match="content_hashes.*canonical"
    ):
        _call_launcher(config, probes)
    assert not Path(config["ledger_path"]).exists()
    assert not Path(config["quarantine_dir"]).exists()


def test_content_hashes_rehash_live_source_bytes_pre_ledger(
    tmp_path, monkeypatch
):
    config, certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    producer = Path(
        certificate["components"]["content_hashes"]["producer_sha256"][
            "artifact_path"
        ]
    ).resolve()
    fixture_hash = launcher._sha256_regular_file

    def observe_mutated_source(path, *, label):
        if Path(path).resolve() == producer:
            return "0" * 64
        return fixture_hash(path, label=label)

    monkeypatch.setattr(
        launcher, "_sha256_regular_file", observe_mutated_source
    )
    with pytest.raises(launcher.LaunchRefusal, match="live SHA-256"):
        _call_launcher(config, probes)
    assert not Path(config["ledger_path"]).exists()
    assert not Path(config["quarantine_dir"]).exists()


def test_live_untracked_disclosure_must_exactly_match_certificate(
    tmp_path, monkeypatch
):
    config, _certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    base_run_git = probes["run_git"]

    def run_git(command):
        if command == list(launcher._GIT_UNTRACKED_STATUS_COMMAND):
            return "?? harmless-notes.txt\x00"
        return base_run_git(command)

    probes["run_git"] = run_git
    with pytest.raises(
        launcher.LaunchRefusal, match="does not exactly equal"
    ):
        _call_launcher(config, probes)
    assert not Path(config["ledger_path"]).exists()
    assert not Path(config["quarantine_dir"]).exists()


def test_untracked_root_import_shadow_refuses_pre_ledger(
    tmp_path, monkeypatch
):
    config, certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    shadow = "sentence_transformers.py"
    certificate["components"]["repo"]["untracked_disclosure"] = [shadow]
    _rewrite_ready_certificate(config, certificate)
    base_run_git = probes["run_git"]

    def run_git(command):
        if command == list(launcher._GIT_UNTRACKED_STATUS_COMMAND):
            return f"?? {shadow}\x00"
        return base_run_git(command)

    probes["run_git"] = run_git
    launch_calls = []
    with pytest.raises(
        launcher.LaunchRefusal, match="import-capable"
    ):
        _call_launcher(
            config,
            probes,
            launch=lambda argv, env: launch_calls.append((argv, env)),
        )
    assert launch_calls == []
    assert not Path(config["ledger_path"]).exists()
    assert not Path(config["quarantine_dir"]).exists()


def test_live_hash_bound_untracked_orchestration_is_admissible(
    tmp_path, monkeypatch
):
    config, certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    orchestration = launcher.phase4.CONTENT_HASH_RELPATHS[
        "orchestration_sha256"
    ]
    certificate["components"]["repo"]["untracked_disclosure"] = [
        orchestration
    ]
    _rewrite_ready_certificate(config, certificate)
    base_run_git = probes["run_git"]

    def run_git(command):
        if command == list(launcher._GIT_UNTRACKED_STATUS_COMMAND):
            return f"?? {orchestration}\x00"
        return base_run_git(command)

    probes["run_git"] = run_git
    result = _call_launcher(config, probes)
    assert result["verdict"] == "PASS"


def test_canonical_qa012_digest_is_live_rechecked_pre_ledger(
    tmp_path, monkeypatch
):
    config, certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    certificate["components"]["qa012"]["manifest_sha256"] = "0" * 64
    _rewrite_ready_certificate(config, certificate)
    with pytest.raises(launcher.LaunchRefusal, match="QA-012|qa012"):
        _call_launcher(config, probes)
    assert not Path(config["ledger_path"]).exists()
    assert not Path(config["quarantine_dir"]).exists()


@pytest.mark.parametrize(
    "mutation",
    ["unknown_abbreviation", "wrong_script", "preexisting_digest", "pins"],
)
def test_malicious_ready_command_contract_refuses_pre_ledger(
    tmp_path, monkeypatch, mutation
):
    config, certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    environment = certificate["components"]["environment"]
    command = environment["command"]
    if mutation == "unknown_abbreviation":
        command.extend(["--num-boot", "1000"])
    elif mutation == "wrong_script":
        command[1] = "scripts/not_the_producer.py"
    elif mutation == "preexisting_digest":
        command.extend(["--certificate-digest", "a" * 64])
    else:
        environment["thread_settings"] = {"OMP_NUM_THREADS": "2"}
    _rewrite_ready_certificate(config, certificate)

    with pytest.raises(launcher.LaunchRefusal, match="exact-command refusal"):
        _call_launcher(config, probes)
    assert not Path(config["ledger_path"]).exists()
    assert not Path(config["quarantine_dir"]).exists()


@pytest.mark.parametrize(
    ("flag", "duplicate"),
    [
        ("--out", False),
        ("--out", True),
        ("--records-out", False),
        ("--records-out", True),
    ],
)
def test_output_flags_are_unique_and_required_pre_ledger(
    tmp_path, monkeypatch, flag, duplicate
):
    config, certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    command = certificate["components"]["environment"]["command"]
    index = command.index(flag)
    if duplicate:
        command.extend([flag, command[index + 1]])
    else:
        del command[index : index + 2]
    _rewrite_ready_certificate(config, certificate)

    with pytest.raises(launcher.LaunchRefusal, match=flag):
        _call_launcher(config, probes)
    assert not Path(config["ledger_path"]).exists()
    assert not Path(config["quarantine_dir"]).exists()


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--records-out", "records"),
        ("--out", "phase4_run_output/records"),
        ("--out", "phase4_run_output/NUL"),
        ("--out", "stopdff_fair_qa_regenerated.json"),
    ],
)
def test_output_arguments_are_exactly_pinned_pre_ledger(
    tmp_path, monkeypatch, flag, value
):
    config, certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    command = certificate["components"]["environment"]["command"]
    command[command.index(flag) + 1] = value
    _rewrite_ready_certificate(config, certificate)

    with pytest.raises(launcher.LaunchRefusal, match=flag):
        _call_launcher(config, probes)
    assert not Path(config["ledger_path"]).exists()
    assert not Path(config["quarantine_dir"]).exists()


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("manifest", "snapshot_manifest_path"),
        ("primary", "primary_scorer"),
        ("disjoint", "disjoint_selector"),
    ],
)
def test_external_snapshot_config_must_match_certificate_command(
    tmp_path, monkeypatch, mutation, message
):
    config, _certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    substituted = tmp_path / "substituted"
    if mutation == "manifest":
        config["snapshot_manifest_path"] = substituted
    elif mutation == "primary":
        config["snapshot_dirs"]["primary_scorer"] = substituted
    else:
        config["snapshot_dirs"]["disjoint_selector"] = substituted

    with pytest.raises(launcher.LaunchRefusal, match=message):
        _call_launcher(config, probes)
    assert not Path(config["ledger_path"]).exists()


def test_anchor_is_rehashed_against_certificate_pre_ledger(
    tmp_path, monkeypatch
):
    config, certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    certificate["components"]["parity"]["anchor_sha256"] = "0" * 64
    _rewrite_ready_certificate(config, certificate)
    with pytest.raises(
        launcher.LaunchRefusal, match="canonical.*parity-anchor pin"
    ):
        _call_launcher(config, probes)
    assert not Path(config["ledger_path"]).exists()


def test_self_consistent_substitute_anchor_refuses_pre_ledger(
    tmp_path, monkeypatch
):
    config, certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    substitute = tmp_path / "substitute_anchor.json"
    substitute.write_bytes(Path(config["anchor_path"]).read_bytes())
    config["anchor_path"] = substitute
    certificate["components"]["parity"]["anchor_sha256"] = _sha256(
        substitute.read_bytes()
    )
    _rewrite_ready_certificate(config, certificate)
    with pytest.raises(launcher.LaunchRefusal, match="canonical frozen parity"):
        _call_launcher(config, probes)
    assert not Path(config["ledger_path"]).exists()


def test_snapshot_manifest_is_rehashed_against_certificate_pre_ledger(
    tmp_path, monkeypatch
):
    config, _certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    fixture_hash = launcher._sha256_regular_file

    def manifest_drift(path, *, label):
        if Path(path).resolve() == Path(
            config["snapshot_manifest_path"]
        ).resolve():
            return "0" * 64
        return fixture_hash(path, label=label)

    monkeypatch.setattr(launcher, "_sha256_regular_file", manifest_drift)
    with pytest.raises(
        launcher.LaunchRefusal, match="snapshot manifest live SHA-256"
    ):
        _call_launcher(config, probes)
    assert not Path(config["ledger_path"]).exists()
    assert not Path(config["quarantine_dir"]).exists()


def test_snapshot_manifest_config_must_match_certificate_artifact(
    tmp_path, monkeypatch
):
    config, _certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    substitute = tmp_path / "substitute_manifest.json"
    substitute.write_text("{}\n", encoding="utf-8")
    config["snapshot_manifest_path"] = substitute
    with pytest.raises(
        launcher.LaunchRefusal, match="snapshot_manifest_path"
    ):
        _call_launcher(config, probes)
    assert not Path(config["ledger_path"]).exists()
    assert not Path(config["quarantine_dir"]).exists()


def test_self_consistent_substitute_snapshot_manifest_refuses_canonical_pin(
    tmp_path, monkeypatch
):
    config, certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    canonical = Path(config["snapshot_manifest_path"])
    substitute = tmp_path / "substitute_snapshot_manifest.json"
    substitute.write_bytes(canonical.read_bytes())
    snapshots = certificate["components"]["snapshots"]
    snapshots["artifact_path"] = str(substitute)
    snapshots["artifact_sha256"] = _sha256(substitute.read_bytes())
    command = certificate["components"]["environment"]["command"]
    command[command.index("--snapshot-manifest") + 1] = str(substitute)
    config["snapshot_manifest_path"] = substitute
    _rewrite_ready_certificate(config, certificate)

    with pytest.raises(
        launcher.LaunchRefusal, match="canonical repo manifest"
    ):
        _call_launcher(config, probes)
    assert not Path(config["ledger_path"]).exists()
    assert not Path(config["quarantine_dir"]).exists()


def test_snapshot_role_revision_must_match_loaded_manifest_pre_ledger(
    tmp_path, monkeypatch
):
    config, certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    certificate["components"]["snapshots"]["primary_scorer"][
        "hf_revision"
    ] = "c" * 40
    _rewrite_ready_certificate(config, certificate)
    with pytest.raises(
        launcher.LaunchRefusal, match="primary_scorer.*hf_revision"
    ):
        _call_launcher(config, probes)
    assert not Path(config["ledger_path"]).exists()
    assert not Path(config["quarantine_dir"]).exists()


def test_staged_inputs_are_rehashed_before_ledger(tmp_path, monkeypatch):
    config, _certificate, staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    staged_path.write_bytes(b'{"items": [1, 2]}\n')
    fixture_hash = launcher._sha256_regular_file

    def observe_mutated_fit(path, *, label):
        if Path(path).resolve() == staged_path.resolve():
            return probes["real_staged_hash"](path, label=label)
        return fixture_hash(path, label=label)

    monkeypatch.setattr(
        launcher, "_sha256_regular_file", observe_mutated_fit
    )
    with pytest.raises(launcher.LaunchRefusal, match="live SHA-256"):
        _call_launcher(config, probes)
    assert not Path(config["ledger_path"]).exists()
    assert not Path(config["quarantine_dir"]).exists()


def test_child_environment_includes_cross_platform_thread_pins(
    tmp_path, monkeypatch
):
    config, _certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    monkeypatch.setenv("PYTHONUNBUFFERED", "1")
    observed_envs = []

    def record_launch(_argv, env):
        observed_envs.append(env)
        return 0

    result = _call_launcher(config, probes, launch=record_launch)
    assert result["verdict"] == "PASS"
    assert len(observed_envs) == 1
    for name in (
        "PYTHONNOUSERSITE",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        assert observed_envs[0][name] == "1"
    for name in launcher.AMBIENT_PYTHON_INJECTION_VARS:
        assert name not in observed_envs[0]
    assert "PYTHONUNBUFFERED" not in observed_envs[0]


def test_git_probe_ignores_ambient_path(monkeypatch):
    observed = {}

    def fake_run(argv, **kwargs):
        observed["argv"] = argv
        observed["env"] = kwargs["env"]
        return subprocess.CompletedProcess(argv, 0, stdout="head\n", stderr="")

    monkeypatch.setenv("PATH", str(Path("fake-git-bin")))
    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    assert launcher._default_run_git(["git", "rev-parse", "HEAD"]) == "head\n"
    assert Path(observed["argv"][0]).is_absolute()
    assert "fake-git-bin" not in observed["argv"][0]
    assert "PATH" not in observed["env"]


def test_child_argv_uses_private_captured_inputs(
    tmp_path, monkeypatch
):
    config, certificate, staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch, data_dir_equals=True
    )
    observed = []
    original_bytes = staged_path.read_bytes()

    def observe_and_mutate_sources(argv, _env):
        observed.append(list(argv))
        staged_path.write_bytes(b"post-capture source mutation\n")
        for source_dir in config["snapshot_dirs"].values():
            (Path(source_dir) / "post-capture.bin").write_bytes(b"changed")
        return 0

    result = _call_launcher(
        config, probes, launch=observe_and_mutate_sources
    )
    assert result["verdict"] == "PASS"
    assert len(observed) == 1
    argv = observed[0]
    capture_root = (
        Path(config["promote_to"]) / launcher.CAPTURED_INPUTS_DIRNAME
    )
    assert argv[argv.index("--calibration") + 1].startswith(
        str(Path(config["quarantine_dir"]) / launcher.CAPTURED_INPUTS_DIRNAME)
    )
    assert next(
        token for token in argv if token.startswith("--data-dir=")
    ).split("=", 1)[1].startswith(
        str(Path(config["quarantine_dir"]) / launcher.CAPTURED_INPUTS_DIRNAME)
    )
    for flag in (
        "--eligibility",
        "--snapshot-manifest",
        "--primary-model-path",
        "--disjoint-model-path",
    ):
        assert argv[argv.index(flag) + 1].startswith(
            str(
                Path(config["quarantine_dir"])
                / launcher.CAPTURED_INPUTS_DIRNAME
            )
        )
    for index, token in enumerate(argv):
        if token != "--staged-input":
            continue
        staged_spec = argv[index + 1]
        assert f"={Path(config['quarantine_dir'])}" in staged_spec
        assert launcher.CAPTURED_INPUTS_DIRNAME in staged_spec
    assert (
        capture_root / "data" / launcher.phase4.R082_DATA_FILENAMES["fit_split"]
    ).read_bytes() == original_bytes
    ledger = json.loads(Path(config["ledger_path"]).read_text("utf-8"))
    assert ledger["argv"] == argv
    assert certificate["components"]["environment"]["command"] != argv


def test_persistent_captured_input_mutation_blocks_promotion(
    tmp_path, monkeypatch
):
    config, _certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )

    def mutate_captured_input(argv, _env):
        captured = Path(argv[argv.index("--calibration") + 1])
        captured.chmod(stat.S_IREAD | stat.S_IWRITE)
        captured.write_bytes(b"child mutation\n")
        return 0

    with pytest.raises(launcher.RunFailed, match="changed during execution"):
        _call_launcher(config, probes, launch=mutate_captured_input)
    quarantine = Path(config["quarantine_dir"])
    report = json.loads(
        (quarantine / launcher.STOP_REPORT_NAME).read_text("utf-8")
    )
    assert report["reason"] == "captured_input_drift"
    assert not Path(config["promote_to"]).exists()


def test_capture_regular_file_authenticates_copied_bytes(tmp_path):
    source = tmp_path / "source.bin"
    destination = tmp_path / "private" / "copy.bin"
    content = b"held-descriptor bytes\n"
    source.write_bytes(content)
    launcher._capture_regular_file(
        source,
        destination,
        expected_sha256=_sha256(content),
        expected_size=len(content),
        label="test input",
    )
    source.write_bytes(b"replacement bytes\n")
    assert destination.read_bytes() == content

    with pytest.raises(launcher.LaunchRefusal, match="copied SHA-256"):
        launcher._capture_regular_file(
            source,
            tmp_path / "private" / "wrong.bin",
            expected_sha256="0" * 64,
            label="wrong digest",
        )


@pytest.mark.parametrize("stage", ["launch", "comparator"])
@pytest.mark.parametrize("interrupt", [KeyboardInterrupt(), SystemExit(17)])
def test_interrupt_after_ledger_claim_writes_stop_report(
    tmp_path, monkeypatch, stage, interrupt
):
    config, _certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )

    def raise_interrupt(*_args):
        raise interrupt

    kwargs = (
        {"launch": raise_interrupt}
        if stage == "launch"
        else {"compare": raise_interrupt}
    )
    expected_reason = "launch_crash" if stage == "launch" else "comparator_crash"
    with pytest.raises(launcher.RunFailed) as caught:
        _call_launcher(config, probes, **kwargs)
    assert isinstance(caught.value.__cause__, type(interrupt))
    assert Path(config["ledger_path"]).is_file()
    quarantine = Path(config["quarantine_dir"])
    report = json.loads(
        (quarantine / launcher.STOP_REPORT_NAME).read_text("utf-8")
    )
    assert report["reason"] == expected_reason
    assert report["activation_digest"] == config["activation_digest"]
    assert not Path(config["promote_to"]).exists()


@pytest.mark.parametrize("stage", ["launch", "comparator"])
def test_preacceptance_stop_failure_never_masks_primary_runfailed(
    tmp_path, monkeypatch, stage
):
    config, _certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )

    def primary_failure(*_args):
        raise OSError(f"synthetic {stage} failure")

    monkeypatch.setattr(
        launcher,
        "_write_stop_report",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("synthetic STOP failure")
        ),
    )
    kwargs = (
        {"launch": primary_failure}
        if stage == "launch"
        else {"compare": primary_failure}
    )
    with pytest.raises(
        launcher.RunFailed, match="STOP report publication failed"
    ) as caught:
        _call_launcher(config, probes, **kwargs)

    assert isinstance(caught.value.__cause__, OSError)
    assert str(caught.value.__cause__) == f"synthetic {stage} failure"
    assert not Path(config["promote_to"]).exists()
    assert not (
        Path(config["quarantine_dir"]) / launcher.ACCEPTANCE_MARKER_NAME
    ).exists()


def test_partial_ledger_writes_are_completed_and_fsynced(
    tmp_path, monkeypatch
):
    config, _certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    writes = []
    fsyncs = []
    parent_syncs = []

    def partial_write(fd, data):
        chunk = bytes(data[:7])
        writes.append(chunk)
        return launcher.os.write(fd, chunk)

    result = _call_launcher(
        config,
        probes,
        ledger_write=partial_write,
        ledger_fsync=lambda fd: fsyncs.append(fd),
        ledger_parent_sync=lambda parent: parent_syncs.append(Path(parent)),
    )
    assert result["verdict"] == "PASS"
    assert len(writes) > 1
    assert len(fsyncs) == 1
    assert parent_syncs == [Path(config["ledger_path"]).parent]
    ledger = json.loads(Path(config["ledger_path"]).read_text("utf-8"))
    assert ledger["activation_digest"] == config["activation_digest"]


@pytest.mark.parametrize(
    "failure", ["zero_write", "write_error", "fsync", "parent_sync"]
)
def test_ledger_durability_failure_never_launches_and_preserves_claim(
    tmp_path, monkeypatch, failure
):
    config, _certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    launch_calls = []
    calls = {"write": 0}

    def ledger_write(fd, data):
        calls["write"] += 1
        if failure == "zero_write":
            return 0
        if calls["write"] == 1:
            return launcher.os.write(fd, bytes(data[:9]))
        raise OSError("synthetic ledger write failure")

    def ledger_fsync(_fd):
        if failure == "fsync":
            raise OSError("synthetic ledger fsync failure")

    def ledger_parent_sync(_parent):
        if failure == "parent_sync":
            raise OSError("synthetic ledger parent-sync failure")

    if failure in {"fsync", "parent_sync"}:
        ledger_write = launcher.os.write

    with pytest.raises(launcher.RunFailed, match="durably written"):
        _call_launcher(
            config,
            probes,
            launch=lambda argv, env: launch_calls.append((argv, env)),
            ledger_write=ledger_write,
            ledger_fsync=ledger_fsync,
            ledger_parent_sync=ledger_parent_sync,
        )
    assert launch_calls == []
    assert Path(config["ledger_path"]).exists()
    quarantine = Path(config["quarantine_dir"])
    assert quarantine.is_dir()
    report = json.loads(
        (quarantine / launcher.STOP_REPORT_NAME).read_text("utf-8")
    )
    assert report["reason"] == "ledger_write_failure"
    assert report["activation_digest"] == config["activation_digest"]
    assert not Path(config["promote_to"]).exists()


def test_default_comparator_uses_preledger_cached_anchor(
    tmp_path, monkeypatch
):
    config, _certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    compared = []

    def compare_parity(anchor, regenerated):
        compared.append((anchor, regenerated))
        return {"verdict": "PASS", "checked": 194, "failures": []}

    monkeypatch.setattr(launcher.phase4, "compare_parity", compare_parity)

    def launch_and_mutate_anchor(argv, _env):
        _write_valid_default_outputs(
            argv, probes["verified_eligibility"]
        )
        assert Path(config["ledger_path"]).is_file()
        config["anchor_path"] = tmp_path / "post-ledger-substitute.json"
        return 0

    result = launcher.validate_and_launch(
        config,
        run_git=probes["run_git"],
        resolve_executable=probes["resolve_executable"],
        host_identity=probes["host_identity"],
        probe_environment_lock=probes["probe_environment_lock"],
        launch=launch_and_mutate_anchor,
    )
    assert result["verdict"] == "PASS"
    assert len(compared) == 1
    assert compared[0][0]["artifact_type"] == "parity_anchor"
    assert compared[0][1]["regenerated"] is True


@pytest.mark.parametrize(
    ("defect", "message"),
    [
        ("missing_certificate_digest", "certificate_digest"),
        ("wrong_certificate_digest", "certificate_digest"),
        ("missing_record_entry", "keyset"),
        ("extra_record_entry", "keyset"),
        ("missing_record_file", "file set"),
        ("extra_record_file", "file set"),
        ("tampered_record_file", "SHA-256"),
        ("wrong_record_path", "artifact-relative"),
        ("missing_entry_field", "non-closed"),
        ("extra_entry_field", "non-closed"),
        ("wrong_n_items", "n_items"),
        ("wrong_historical_cell", "historical_cell"),
        ("wrong_policy", "policy"),
        ("wrong_item_key", "ineligible"),
        ("invalid_record_row", "canonical schema"),
        ("excluded_record", "excluded rather than a complete pair"),
    ],
)
def test_zero_exit_malformed_record_export_never_promotes(
    tmp_path, monkeypatch, defect, message
):
    config, _certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    first_cell = launcher.schema.CELL_IDS[0]

    def mutate(payload, records_root):
        phase4_metadata = payload["metadata"]["phase4"]
        exported = phase4_metadata["exported_records"]
        entry = exported[first_cell]
        record_path = records_root / f"{first_cell}.jsonl"
        if defect == "missing_certificate_digest":
            phase4_metadata.pop("certificate_digest")
        elif defect == "wrong_certificate_digest":
            phase4_metadata["certificate_digest"] = "0" * 64
        elif defect == "missing_record_entry":
            exported.pop(first_cell)
        elif defect == "extra_record_entry":
            exported["unexpected__cell"] = dict(entry)
        elif defect == "missing_record_file":
            record_path.unlink()
        elif defect == "extra_record_file":
            (records_root / "unexpected.jsonl").write_text(
                "{}\n", encoding="utf-8"
            )
        elif defect == "tampered_record_file":
            with record_path.open("ab") as handle:
                handle.write(b"{}\n")
        elif defect == "wrong_record_path":
            entry["path"] = str(record_path)
        elif defect == "missing_entry_field":
            entry.pop("policy")
        elif defect == "extra_entry_field":
            entry["untrusted"] = True
        elif defect == "wrong_n_items":
            entry["n_items"] = launcher.schema.EXPECTED_COMPLETE_PAIRS - 1
        elif defect == "wrong_historical_cell":
            entry["historical_cell"] = "wrong+shared"
        elif defect == "wrong_policy":
            entry["policy"] = "myopic"
        elif defect in {
            "wrong_item_key",
            "invalid_record_row",
            "excluded_record",
        }:
            lines = record_path.read_text("utf-8").splitlines()
            first = json.loads(lines[0])
            if defect == "wrong_item_key":
                first["item_key"] = "not-eligible"
            else:
                if defect == "invalid_record_row":
                    first["mc_stop_step"] = True
                else:
                    first["excluded"] = True
                    first["exclusion_reason"] = "UNKNOWN_NOT_INFERRED"
            lines[0] = json.dumps(
                first, sort_keys=True, separators=(",", ":")
            )
            mutated = ("\n".join(lines) + "\n").encode("utf-8")
            record_path.write_bytes(mutated)
            entry["sha256"] = _sha256(mutated)

    launch_calls = []

    def zero_exit(argv, _env):
        launch_calls.append(list(argv))
        _write_valid_default_outputs(
            argv,
            probes["verified_eligibility"],
            mutate=mutate,
        )
        return 0

    monkeypatch.setattr(
        launcher.phase4,
        "compare_parity",
        lambda *_args: pytest.fail("parity must follow record validation"),
    )
    with pytest.raises(launcher.RunFailed, match="comparator crashed"):
        _call_default_launcher(config, probes, launch=zero_exit)

    assert len(launch_calls) == 1
    assert Path(config["ledger_path"]).is_file()
    quarantine = Path(config["quarantine_dir"])
    assert quarantine.is_dir()
    stop = json.loads(
        (quarantine / launcher.STOP_REPORT_NAME).read_text("utf-8")
    )
    assert stop["reason"] == "comparator_crash"
    assert message in stop["error"]
    assert not Path(config["promote_to"]).exists()


def test_promotion_oserror_writes_stop_report_and_raises_runfailed(
    tmp_path, monkeypatch
):
    config, _certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )

    def fail_rename(_source, _destination):
        raise OSError("synthetic rename failure")

    monkeypatch.setattr(launcher.os, "rename", fail_rename)
    with pytest.raises(launcher.RunFailed, match="atomic promotion"):
        _call_launcher(config, probes)

    quarantine = Path(config["quarantine_dir"])
    assert quarantine.is_dir()
    stop_report = json.loads(
        (quarantine / launcher.STOP_REPORT_NAME).read_text("utf-8")
    )
    assert stop_report["reason"] == "promotion_crash"
    assert stop_report["activation_digest"] == config["activation_digest"]
    assert Path(config["ledger_path"]).is_file()
    assert not Path(config["promote_to"]).exists()


def test_staged_tree_sync_failure_prevents_promotion(tmp_path, monkeypatch):
    config, _certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    quarantine = Path(config["quarantine_dir"])
    promote_to = Path(config["promote_to"])

    def fail_sync(root):
        root = Path(root)
        assert root != quarantine
        assert root.parent == quarantine
        assert root.name.startswith(launcher.PRIVATE_PROMOTION_PREFIX)
        receipt_path = root / launcher.LAUNCH_RECEIPT_NAME
        receipt = json.loads(receipt_path.read_text("utf-8"))
        assert (root / receipt["export_basename"]).is_file()
        assert all(
            (root / "records" / f"{cell_id}.jsonl").is_file()
            for cell_id in launcher.schema.CELL_IDS
        )
        raise OSError("synthetic staged fsync failure")

    monkeypatch.setattr(launcher.fileio, "fsync_tree", fail_sync)
    monkeypatch.setattr(
        launcher,
        "publish_dir_create_once",
        lambda *_args, **_kwargs: pytest.fail(
            "publication must not run after staged sync failure"
        ),
    )

    with pytest.raises(launcher.RunFailed, match="atomic promotion failed"):
        _call_launcher(config, probes)

    assert quarantine.is_dir()
    assert not promote_to.exists()
    report = json.loads(
        (quarantine / launcher.STOP_REPORT_NAME).read_text("utf-8")
    )
    assert report["reason"] == "promotion_crash"
    assert report["promotion_committed"] is False
    assert "synthetic staged fsync failure" in report["error"]
    assert not any(
        child.name.startswith(launcher.PRIVATE_PROMOTION_PREFIX)
        for child in quarantine.iterdir()
    )


@pytest.mark.skipif(os.name != "nt", reason="requires a real NTFS junction")
@pytest.mark.parametrize("junction_at", ["root", "child"])
def test_captured_input_snapshot_never_scans_windows_junction(
    tmp_path, monkeypatch, junction_at
):
    root = tmp_path / "captured"
    root.mkdir()
    (root / "local.bin").write_bytes(b"local")
    external = tmp_path / "external"
    external.mkdir()
    (external / "outside.bin").write_bytes(b"outside")
    if junction_at == "root":
        (root / "local.bin").unlink()
        root.rmdir()
        junction = root
    else:
        junction = root / "junction"
    completed = subprocess.run(
        [
            os.environ.get("COMSPEC", "cmd.exe"),
            "/d",
            "/c",
            "mklink",
            "/J",
            str(junction),
            str(external),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        pytest.skip(f"junction creation unavailable: {completed.stderr}")

    real_scandir = os.scandir
    scanned = []

    def record_scandir(path):
        scanned.append(Path(path))
        return real_scandir(path)

    monkeypatch.setattr(launcher.os, "scandir", record_scandir)
    try:
        with pytest.raises(launcher.LaunchRefusal, match="reparse"):
            launcher._captured_input_snapshot(root)
        assert junction not in scanned
    finally:
        os.rmdir(junction)


@pytest.mark.skipif(os.name != "nt", reason="requires a real NTFS junction")
def test_private_cleanup_never_scans_or_chmods_through_windows_junction(
    tmp_path, monkeypatch
):
    quarantine = tmp_path / "quarantine"
    candidate = quarantine / f"{launcher.PRIVATE_PROMOTION_PREFIX}candidate"
    candidate.mkdir(parents=True)
    (candidate / "local.bin").write_bytes(b"local")
    external = tmp_path / "external"
    external.mkdir()
    external_file = external / "outside.bin"
    external_file.write_bytes(b"outside")
    junction = candidate / "junction"
    completed = subprocess.run(
        [
            os.environ.get("COMSPEC", "cmd.exe"),
            "/d",
            "/c",
            "mklink",
            "/J",
            str(junction),
            str(external),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        pytest.skip(f"junction creation unavailable: {completed.stderr}")

    real_scandir = os.scandir
    real_chmod = Path.chmod
    scanned = []
    chmodded = []

    def record_scandir(path):
        scanned.append(Path(path))
        return real_scandir(path)

    def record_chmod(path, mode, *args, **kwargs):
        chmodded.append(Path(path))
        return real_chmod(path, mode, *args, **kwargs)

    monkeypatch.setattr(launcher.os, "scandir", record_scandir)
    monkeypatch.setattr(Path, "chmod", record_chmod)
    try:
        launcher._remove_private_promotion_tree(candidate, quarantine)
        assert candidate.is_dir()
        assert junction not in scanned
        assert chmodded == []
    finally:
        os.rmdir(junction)


def test_private_cleanup_defect_never_masks_materialization_failure(
    tmp_path, monkeypatch
):
    quarantine = tmp_path / "quarantine"
    quarantine.mkdir()
    source = tmp_path / "source"
    source.mkdir()
    out_basename = "export.json"
    output_snapshot = {out_basename: b"{}\n"}
    output_snapshot.update(
        {
            f"records/{cell_id}.jsonl": b"{}\n"
            for cell_id in launcher.schema.CELL_IDS
        }
    )

    monkeypatch.setattr(
        launcher,
        "_capture_regular_file",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            ValueError("primary materialization failure")
        ),
    )
    monkeypatch.setattr(
        launcher,
        "_closed_tree_regular_files",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            RuntimeError("cleanup failure")
        ),
    )

    with pytest.raises(ValueError, match="primary materialization failure"):
        launcher._materialize_private_promotion_tree(
            quarantine,
            captured_inputs={
                "root": source,
                "snapshot": {
                    "input.bin": {"size": 1, "sha256": "a" * 64}
                },
            },
            activation_digest="b" * 64,
            ledger_path=tmp_path / "ledger.json",
            out_basename=out_basename,
            comparator_result={
                "verdict": "PASS",
                "checked": 194,
                "failures": [],
            },
            comparator_output_snapshot=output_snapshot,
        )


def test_publish_uses_private_snapshot_when_original_output_mutates(
    tmp_path, monkeypatch
):
    config, _certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    quarantine = Path(config["quarantine_dir"])
    promote_to = Path(config["promote_to"])
    real_publish = launcher.publish_dir_create_once

    monkeypatch.setattr(
        launcher.phase4,
        "compare_parity",
        lambda *_args: {"verdict": "PASS", "checked": 194, "failures": []},
    )

    held = {}

    def zero_exit(argv, _env):
        _write_valid_default_outputs(argv, probes["verified_eligibility"])
        first_cell = launcher.schema.CELL_IDS[0]
        held["writer"] = (
            quarantine / "records" / f"{first_cell}.jsonl"
        ).open("ab")
        return 0

    observed = {}

    def mutate_original_then_publish(staged, destination, **kwargs):
        staged = Path(staged)
        assert staged != quarantine
        assert staged.parent == quarantine
        first_cell = launcher.schema.CELL_IDS[0]
        candidate_record = staged / "records" / f"{first_cell}.jsonl"
        original_record = quarantine / "records" / f"{first_cell}.jsonl"
        observed["validated"] = candidate_record.read_bytes()
        handle = held.pop("writer")
        try:
            handle.write(b"post-validation-open-handle-drift\n")
            handle.flush()
            os.fsync(handle.fileno())
        finally:
            handle.close()
        assert original_record.read_bytes() != observed["validated"]
        return real_publish(staged, destination, **kwargs)

    monkeypatch.setattr(
        launcher, "publish_dir_create_once", mutate_original_then_publish
    )

    result = _call_default_launcher(config, probes, launch=zero_exit)

    assert result["verdict"] == "PASS"
    assert not quarantine.exists()
    first_cell = launcher.schema.CELL_IDS[0]
    promoted_record = promote_to / "records" / f"{first_cell}.jsonl"
    assert promoted_record.read_bytes() == observed["validated"]
    receipt = json.loads(
        (promote_to / launcher.LAUNCH_RECEIPT_NAME).read_text("utf-8")
    )
    assert (
        receipt["process_trust_model"]
        == launcher.phase4.PHASE4_PROCESS_TRUST_MODEL_ID
    )
    assert receipt["records_sha256"][first_cell] == _sha256(
        observed["validated"]
    )
    marker = json.loads(
        (promote_to / launcher.ACCEPTANCE_MARKER_NAME).read_text("utf-8")
    )
    assert marker == {
        "schema_version": launcher.schema.SCHEMA_VERSION,
        "marker_type": "phase4_launch_accepted",
        "activation_digest": config["activation_digest"],
        "launch_receipt_sha256": _sha256(
            (promote_to / launcher.LAUNCH_RECEIPT_NAME).read_bytes()
        ),
    }
    assert {child.name for child in promote_to.iterdir()} == {
        launcher.ACCEPTANCE_MARKER_NAME,
        launcher.CAPTURED_INPUTS_DIRNAME,
        launcher.LAUNCH_RECEIPT_NAME,
        "records",
        receipt["export_basename"],
    }


def test_post_fsync_output_drift_prevents_promotion(tmp_path, monkeypatch):
    config, _certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    quarantine = Path(config["quarantine_dir"])
    promote_to = Path(config["promote_to"])
    real_fsync_tree = launcher.fileio.fsync_tree

    def sync_then_mutate(root):
        real_fsync_tree(root)
        root = Path(root)
        assert root != quarantine
        assert root.parent == quarantine
        first_cell = launcher.schema.CELL_IDS[0]
        with (root / "records" / f"{first_cell}.jsonl").open(
            "ab"
        ) as handle:
            handle.write(b"post-fsync-drift\n")

    monkeypatch.setattr(launcher.fileio, "fsync_tree", sync_then_mutate)
    monkeypatch.setattr(
        launcher,
        "publish_dir_create_once",
        lambda *_args, **_kwargs: pytest.fail(
            "publication must not run after post-fsync output drift"
        ),
    )

    with pytest.raises(launcher.RunFailed, match="atomic promotion failed"):
        _call_launcher(config, probes)

    assert quarantine.is_dir()
    assert not promote_to.exists()
    report = json.loads(
        (quarantine / launcher.STOP_REPORT_NAME).read_text("utf-8")
    )
    assert report["reason"] == "promotion_crash"
    assert "changed after the comparator" in report["error"]


def test_promotion_destination_race_never_replaces_empty_incumbent(
    tmp_path, monkeypatch
):
    config, _certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    real_publish = launcher.publish_dir_create_once
    promote_to = Path(config["promote_to"])

    def race_then_publish(staged, destination, **kwargs):
        promote_to.mkdir()
        return real_publish(staged, destination, **kwargs)

    monkeypatch.setattr(
        launcher, "publish_dir_create_once", race_then_publish
    )
    with pytest.raises(launcher.RunFailed, match="nothing promoted"):
        _call_launcher(config, probes)

    assert promote_to.is_dir()
    assert list(promote_to.iterdir()) == []
    quarantine = Path(config["quarantine_dir"])
    assert quarantine.is_dir()
    report = json.loads(
        (quarantine / launcher.STOP_REPORT_NAME).read_text("utf-8")
    )
    assert report["reason"] == "promotion_crash"
    assert report["promotion_committed"] is False


def test_receipt_failure_never_reclaims_peer_destination_claim(
    tmp_path, monkeypatch
):
    config, _certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    promote_to = Path(config["promote_to"])

    def peer_claim_then_receipt_failure(*_args, **_kwargs):
        promote_to.mkdir()
        raise OSError("synthetic receipt failure after peer claim")

    monkeypatch.setattr(
        launcher, "_write_launch_receipt", peer_claim_then_receipt_failure
    )
    with pytest.raises(launcher.RunFailed, match="nothing promoted"):
        _call_launcher(config, probes)

    assert promote_to.is_dir()
    assert list(promote_to.iterdir()) == []
    quarantine = Path(config["quarantine_dir"])
    assert quarantine.is_dir()
    report = json.loads(
        (quarantine / launcher.STOP_REPORT_NAME).read_text("utf-8")
    )
    assert report["reason"] == "promotion_crash"
    assert report["promotion_committed"] is False


def test_post_rename_sync_failure_records_truthful_committed_state(
    tmp_path, monkeypatch
):
    config, _certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    quarantine = Path(config["quarantine_dir"])
    promote_to = Path(config["promote_to"])

    def rename_then_fail(staged, destination, **_kwargs):
        launcher.os.rename(staged, destination)
        raise OSError("synthetic parent fsync failure")

    monkeypatch.setattr(
        launcher, "publish_dir_create_once", rename_then_fail
    )
    with pytest.raises(
        launcher.RunFailed, match="private promotion committed.*cleanup failed"
    ):
        _call_launcher(config, probes)

    assert quarantine.exists()
    assert promote_to.is_dir()
    report = json.loads(
        (promote_to / launcher.STOP_REPORT_NAME).read_text("utf-8")
    )
    assert report["reason"] == "promotion_durability_failure"
    assert report["promotion_committed"] is True
    assert report["activation_digest"] == config["activation_digest"]
    assert not (promote_to / launcher.ACCEPTANCE_MARKER_NAME).exists()


def test_postcommit_quarantine_cleanup_failure_never_returns_pass(
    tmp_path, monkeypatch
):
    config, _certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    quarantine = Path(config["quarantine_dir"])
    promote_to = Path(config["promote_to"])
    real_rmtree = launcher.shutil.rmtree

    def fail_original_cleanup(path, *args, **kwargs):
        if Path(path) == quarantine:
            raise OSError("synthetic original-quarantine cleanup failure")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(launcher.shutil, "rmtree", fail_original_cleanup)
    with pytest.raises(
        launcher.RunFailed, match="private promotion committed.*cleanup failed"
    ):
        _call_launcher(config, probes)

    assert quarantine.is_dir()
    assert promote_to.is_dir()
    report = json.loads(
        (promote_to / launcher.STOP_REPORT_NAME).read_text("utf-8")
    )
    assert report["reason"] == "post_promotion_cleanup_failure"
    assert report["promotion_committed"] is True
    assert "original-quarantine cleanup failure" in report["error"]
    assert not (promote_to / launcher.ACCEPTANCE_MARKER_NAME).exists()


def test_cleanup_and_stop_write_failures_still_leave_no_acceptance(
    tmp_path, monkeypatch
):
    config, _certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    quarantine = Path(config["quarantine_dir"])
    promote_to = Path(config["promote_to"])
    real_rmtree = launcher.shutil.rmtree

    def fail_original_cleanup(path, *args, **kwargs):
        if Path(path) == quarantine:
            raise OSError("synthetic cleanup EIO")
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(launcher.shutil, "rmtree", fail_original_cleanup)
    monkeypatch.setattr(
        launcher,
        "_write_stop_report",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("synthetic STOP ENOSPC")
        ),
    )
    with pytest.raises(
        launcher.RunFailed, match="STOP report publication also failed"
    ):
        _call_launcher(config, probes)

    assert promote_to.is_dir()
    assert (promote_to / launcher.LAUNCH_RECEIPT_NAME).is_file()
    assert not (promote_to / launcher.ACCEPTANCE_MARKER_NAME).exists()
    assert not (promote_to / launcher.STOP_REPORT_NAME).exists()


def test_quarantine_parent_sync_failure_precedes_acceptance_marker(
    tmp_path, monkeypatch
):
    config, _certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    quarantine = Path(config["quarantine_dir"])
    promote_to = Path(config["promote_to"])
    real_sync = launcher.fileio.fsync_directory

    def fail_post_cleanup_sync(path):
        if Path(path) == quarantine.parent and not quarantine.exists():
            raise OSError("synthetic quarantine-parent sync failure")
        return real_sync(path)

    monkeypatch.setattr(
        launcher.fileio, "fsync_directory", fail_post_cleanup_sync
    )
    with pytest.raises(launcher.RunFailed, match="durability"):
        _call_launcher(config, probes)

    assert promote_to.is_dir()
    assert not (promote_to / launcher.ACCEPTANCE_MARKER_NAME).exists()
    report = json.loads(
        (promote_to / launcher.STOP_REPORT_NAME).read_text("utf-8")
    )
    assert report["reason"] == "post_promotion_cleanup_failure"


def test_acceptance_marker_failure_and_stop_failure_remain_rejected(
    tmp_path, monkeypatch
):
    config, _certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    quarantine = Path(config["quarantine_dir"])
    promote_to = Path(config["promote_to"])
    monkeypatch.setattr(
        launcher,
        "_write_acceptance_marker",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("synthetic marker EIO")
        ),
    )
    monkeypatch.setattr(
        launcher,
        "_write_stop_report",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("synthetic STOP ENOSPC")
        ),
    )
    with pytest.raises(
        launcher.RunFailed, match="acceptance-marker.*STOP report"
    ):
        _call_launcher(config, probes)

    assert not quarantine.exists()
    assert promote_to.is_dir()
    assert (promote_to / launcher.LAUNCH_RECEIPT_NAME).is_file()
    assert not (promote_to / launcher.ACCEPTANCE_MARKER_NAME).exists()
    assert not (promote_to / launcher.STOP_REPORT_NAME).exists()


def test_final_marker_link_failure_remains_rejected_when_stop_also_fails(
    tmp_path, monkeypatch
):
    config, _certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    promote_to = Path(config["promote_to"])
    marker_path = promote_to / launcher.ACCEPTANCE_MARKER_NAME
    real_link = launcher.os.link

    def fail_marker_link(source, destination, *args, **kwargs):
        if Path(destination) == marker_path:
            raise OSError("synthetic final marker link failure")
        return real_link(source, destination, *args, **kwargs)

    monkeypatch.setattr(launcher.os, "link", fail_marker_link)
    monkeypatch.setattr(
        launcher,
        "_write_stop_report",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            OSError("synthetic STOP publication failure")
        ),
    )
    with pytest.raises(launcher.RunFailed, match="acceptance-marker"):
        _call_launcher(config, probes)

    assert not marker_path.exists()
    assert not (promote_to / launcher.ACCEPTANCE_PENDING_NAME).exists()
    assert not (promote_to / launcher.STOP_REPORT_NAME).exists()


@pytest.mark.parametrize("fault", ["temporary_cleanup", "directory_sync"])
@pytest.mark.parametrize("stop_report_fails", [False, True])
def test_postcommit_marker_cleanup_or_sync_uncertainty_remains_pass(
    tmp_path, monkeypatch, fault, stop_report_fails
):
    config, _certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    promote_to = Path(config["promote_to"])
    marker_path = promote_to / launcher.ACCEPTANCE_MARKER_NAME

    if fault == "temporary_cleanup":
        real_unlink = launcher.os.unlink

        def fail_marker_temporary_cleanup(path, *args, **kwargs):
            candidate = Path(path)
            if (
                candidate.parent == promote_to.parent
                and candidate.name.startswith(
                    f".{marker_path.name}.acceptance-"
                )
            ):
                raise OSError("synthetic marker temporary cleanup failure")
            return real_unlink(path, *args, **kwargs)

        monkeypatch.setattr(launcher.os, "unlink", fail_marker_temporary_cleanup)
    else:
        real_sync = launcher.fileio.fsync_directory

        def fail_marker_directory_sync(directory):
            if Path(directory) == promote_to and marker_path.is_file():
                raise OSError("synthetic marker directory sync failure")
            return real_sync(directory)

        monkeypatch.setattr(
            launcher.fileio, "fsync_directory", fail_marker_directory_sync
        )

    if stop_report_fails:
        monkeypatch.setattr(
            launcher,
            "_write_stop_report",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("synthetic STOP publication failure")
            ),
        )

    result = _call_launcher(config, probes)

    assert result["verdict"] == "PASS"
    assert marker_path.is_file()
    assert not (promote_to / launcher.ACCEPTANCE_PENDING_NAME).exists()
    assert not (promote_to / launcher.STOP_REPORT_NAME).exists()
    assert not list(promote_to.glob(f".{marker_path.name}.*"))
    if fault == "temporary_cleanup":
        assert list(
            promote_to.parent.glob(f".{marker_path.name}.acceptance-*")
        )


@pytest.mark.parametrize("stop_report_fails", [False, True])
def test_acceptance_pending_guard_unlink_failure_never_accepts(
    tmp_path, monkeypatch, stop_report_fails
):
    config, _certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    promote_to = Path(config["promote_to"])
    pending_path = promote_to / launcher.ACCEPTANCE_PENDING_NAME
    real_unlink = launcher.os.unlink

    def fail_pending_guard_unlink(path, *args, **kwargs):
        if Path(path) == pending_path:
            raise OSError("synthetic pending-guard unlink failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(launcher.os, "unlink", fail_pending_guard_unlink)
    if stop_report_fails:
        monkeypatch.setattr(
            launcher,
            "_write_stop_report",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("synthetic STOP publication failure")
            ),
        )

    with pytest.raises(launcher.RunFailed, match="acceptance-marker"):
        _call_launcher(config, probes)

    assert pending_path.is_file()
    assert not (promote_to / launcher.ACCEPTANCE_MARKER_NAME).exists()
    assert (promote_to / launcher.STOP_REPORT_NAME).exists() is (
        not stop_report_fails
    )


def test_acceptance_guard_removal_is_synced_before_pass(tmp_path, monkeypatch):
    config, _certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    promote_to = Path(config["promote_to"])
    pending_path = promote_to / launcher.ACCEPTANCE_PENDING_NAME
    marker_path = promote_to / launcher.ACCEPTANCE_MARKER_NAME
    real_sync = launcher.fileio.fsync_directory
    terminal_syncs = []

    def observe_terminal_sync(path):
        path = Path(path)
        if (
            path == promote_to
            and not pending_path.exists()
            and not marker_path.exists()
        ):
            terminal_syncs.append(path)
        return real_sync(path)

    monkeypatch.setattr(
        launcher.fileio, "fsync_directory", observe_terminal_sync
    )

    result = _call_launcher(config, probes)

    assert result["verdict"] == "PASS"
    assert terminal_syncs == [promote_to]


@pytest.mark.parametrize("stop_report_fails", [False, True])
def test_acceptance_guard_removal_sync_failure_prevents_positive_marker(
    tmp_path, monkeypatch, stop_report_fails
):
    config, _certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    promote_to = Path(config["promote_to"])
    pending_path = promote_to / launcher.ACCEPTANCE_PENDING_NAME
    marker_path = promote_to / launcher.ACCEPTANCE_MARKER_NAME
    real_sync = launcher.fileio.fsync_directory

    def fail_terminal_sync(path):
        path = Path(path)
        if (
            path == promote_to
            and not pending_path.exists()
            and not marker_path.exists()
        ):
            raise OSError("synthetic acceptance-guard removal sync failure")
        return real_sync(path)

    monkeypatch.setattr(
        launcher.fileio, "fsync_directory", fail_terminal_sync
    )
    if stop_report_fails:
        monkeypatch.setattr(
            launcher,
            "_write_stop_report",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("synthetic STOP publication failure")
            ),
        )

    with pytest.raises(launcher.RunFailed, match="acceptance-marker"):
        _call_launcher(config, probes)

    assert not marker_path.exists()
    assert not pending_path.exists()
    assert (promote_to / launcher.STOP_REPORT_NAME).exists() is (
        not stop_report_fails
    )


def test_acceptance_marker_is_written_only_after_quarantine_cleanup(
    tmp_path, monkeypatch
):
    config, _certificate, _staged_path, probes = _runtime_fixture(
        tmp_path, monkeypatch
    )
    quarantine = Path(config["quarantine_dir"])
    real_write_marker = launcher._write_acceptance_marker
    observed = {}

    def assert_cleanup_then_mark(promote_to, **kwargs):
        observed["quarantine_exists"] = quarantine.exists()
        return real_write_marker(promote_to, **kwargs)

    monkeypatch.setattr(
        launcher, "_write_acceptance_marker", assert_cleanup_then_mark
    )
    result = _call_launcher(config, probes)

    assert result["verdict"] == "PASS"
    assert observed == {"quarantine_exists": False}
    assert not (
        Path(config["promote_to"]) / launcher.ACCEPTANCE_PENDING_NAME
    ).exists()
