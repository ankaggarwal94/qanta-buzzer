"""Certificate-side R-082 external-staging regression tests.

These tests exercise only certificate assembly/gathering over synthetic paths
and bytes.  They never construct or load a model.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from reproducibility.colm_aims_2026 import phase4


_SHA = "a" * 64
_TEST_DATASET_SHA = phase4.ELIGIBILITY_TEST_DATASET_SHA256
_COMMIT = "b" * 40
_TREE = "c" * 40
_STAGED_LABELS = (
    "calibration_train",
    "eval_split",
    "fit_split",
    "mc_dataset",
    "answer_profiles",
    "build_metadata",
)
_OPERATOR_LABELS = (
    "fit_split",
    "mc_dataset",
    "answer_profiles",
    "build_metadata",
)
_DATA_FILENAMES = {
    "eval_split": "test_dataset.json",
    "fit_split": "val_dataset.json",
    "mc_dataset": "mc_dataset.json",
    "answer_profiles": "answer_profiles.json",
    "build_metadata": "build_metadata.json",
}


def _content_hashes(repo_root: Path) -> dict[str, dict[str, str]]:
    return {
        key: {
            "artifact_path": str((repo_root / relpath).resolve()),
            "sha256": hashlib.sha256(key.encode("utf-8")).hexdigest(),
        }
        for key, relpath in phase4.CONTENT_HASH_RELPATHS.items()
    }


def _materialize_audited_sources(repo_root: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for key, relpath in phase4.CONTENT_HASH_RELPATHS.items():
        path = repo_root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(f"synthetic audited source: {key}\n".encode("utf-8"))
        paths[key] = path
    return paths


def test_regular_file_hash_reads_raw_bytes_on_every_platform(
    tmp_path: Path,
) -> None:
    payload = b"line-one\r\nline-two\x1a\x00\xff\x80\r\n"
    path = tmp_path / "raw-hash-input.bin"
    path.write_bytes(payload)

    observed = phase4._sha256_regular_file(
        path,
        error_cls=phase4.StagedInputError,
        label="raw-byte regression",
    )

    assert observed == hashlib.sha256(payload).hexdigest()


def _suite_receipt(
    suite: str, interpreter: str
) -> dict[str, object]:
    selection = (
        phase4.FOCUSED_SUITE_SELECTION
        if suite == "focused"
        else phase4.FULL_SUITE_SELECTION
    )
    junit_path = Path(interpreter).parent / f"{suite}-suite.xml"
    return {
        "exit_code": 0,
        "command": [
            interpreter,
            "-m",
            "pytest",
            *selection,
            "-q",
            "-p",
            "no:cacheprovider",
            f"--junitxml={junit_path}",
        ],
        "environment_lock_sha256": "d" * 64,
        "workflow_sha256": "e" * 64,
        "interpreter_realpath": interpreter,
        "counts": {"tests": 2, "failures": 0, "errors": 0, "skipped": 0},
        "skip_identities": [],
        "junit_sha256": "1" * 64,
        "transcript_sha256": "2" * 64,
        "commit": _COMMIT,
        "tree_sha256": _TREE,
        "dirty": False,
    }


def _components(
    repo_root: Path,
    staged_path: Path | str,
    command: list[str],
) -> dict[str, object]:
    calibration = Path(staged_path)
    staged_dir = calibration.parent
    launch_root = repo_root.parent / "external-launch"
    interpreter = str(
        (repo_root.parent / "synthetic-runtime" / "python").resolve()
    )
    eligibility_path = (
        repo_root / phase4.ELIGIBILITY_ARTIFACT_RELPATH
    ).resolve()
    staged_paths = {
        "calibration_train": calibration,
        **{
            label: staged_dir / filename
            for label, filename in _DATA_FILENAMES.items()
        },
    }
    return {
        "repo": {
            "commit": _COMMIT,
            "tree_sha256": _TREE,
            "dirty": False,
            "untracked_disclosure": [],
            "root_realpath": str(repo_root.resolve()),
        },
        "content_hashes": _content_hashes(repo_root),
        "eligibility": {
            "digest": phase4.ELIGIBILITY_KEYSET_SHA256,
            "horizon_map_sha256": phase4.ELIGIBILITY_HORIZON_MAP_SHA256,
            "artifact_path": str(eligibility_path),
            "artifact_sha256": phase4.ELIGIBILITY_ARTIFACT_SHA256,
            "test_dataset_sha256": _TEST_DATASET_SHA,
        },
        "snapshots": {
            "artifact_path": str(
                (repo_root / phase4.SNAPSHOT_MANIFEST_RELPATH).resolve()
            ),
            "artifact_sha256": phase4.SNAPSHOT_MANIFEST_SHA256,
            "primary_scorer": {
                "verified": True,
                **phase4.EXPECTED_SNAPSHOT_IDENTITIES["primary_scorer"],
            },
            "disjoint_selector": {
                "verified": True,
                **phase4.EXPECTED_SNAPSHOT_IDENTITIES[
                    "disjoint_selector"
                ],
            },
        },
        "offline_flags": list(phase4.REQUIRED_OFFLINE_FLAGS),
        "staged_inputs": [
            {
                "path": str(staged_paths[label]),
                "label": label,
                "expected_sha256": phase4.R082_STAGED_INPUT_SHA256[label],
                "observed_sha256": phase4.R082_STAGED_INPUT_SHA256[label],
            }
            for label in _STAGED_LABELS
        ],
        "suite_receipts": {
            "focused": _suite_receipt("focused", interpreter),
            "full": _suite_receipt("full", interpreter),
        },
        "parity": {
            "comparator_identity": phase4.PARITY_COMPARATOR_IDENTITY,
            "artifact_path": str(
                (repo_root / phase4.PARITY_ANCHOR_RELPATH).resolve()
            ),
            "anchor_sha256": phase4.PARITY_ANCHOR_SHA256,
            "source_export_a_sha256": phase4.PARITY_SOURCE_EXPORT_A_SHA256,
        },
        "qa012": {
            "artifact_path": str(
                (repo_root / phase4.QA012_MANIFEST_RELPATH).resolve()
            ),
            "manifest_sha256": phase4.QA012_MANIFEST_SHA256,
            "manifest_type": phase4.QA012_MANIFEST_TYPE,
            "revision": phase4.QA012_MANIFEST_REVISION,
            "conventions": dict(phase4.QA012_CONVENTIONS),
        },
        "environment": {
            "interpreter_realpath": interpreter,
            "os": "synthetic-os",
            "arch": "synthetic-arch",
            "cpu": "synthetic-cpu",
            "blas": "synthetic-blas",
            "thread_settings": dict(phase4.PHASE4_THREAD_SETTINGS),
            "environment_lock_sha256": "d" * 64,
            "command": command,
            "seeds": [1],
            "pythonhashseed": "0",
            "archived_rng_pinned": False,
            "fresh_rng_pinned": True,
            "quarantine_dir": str(launch_root / "quarantine"),
            "promote_to": str(launch_root / "promoted"),
            "exception_ledger_path": str(
                launch_root / "exception-ledger.json"
            ),
        },
    }


def _bound_command(
    calibration: Path,
    eligibility_path: Path,
    repo_root: Path,
    *,
    joined: bool = False,
    fit_split: str = "val",
    eval_split: str = "test",
) -> list[str]:
    data_dir = calibration.parent
    snapshot_root = repo_root / "snapshots"
    interpreter = (
        repo_root.parent / "synthetic-runtime" / "python"
    ).resolve()
    command = [str(interpreter), phase4.PHASE4_PRODUCER_SCRIPT]
    for flag, value in (
        ("--data-dir", str(data_dir)),
        ("--calibration", str(calibration)),
        ("--eligibility", str(eligibility_path)),
        ("--fit-split", fit_split),
        ("--eval-split", eval_split),
        ("--reward-schedule", "power_mark"),
        ("--qa-arms", ",".join(phase4.PHASE4_QA_ARMS)),
        ("--calibrations", ",".join(phase4.PHASE4_CALIBRATIONS)),
        ("--num-bootstrap", "1000"),
        ("--n-test", "0"),
        ("--n-val", "0"),
        ("--seed", "1"),
        (
            "--out",
            "phase4_run_output/stopdff_fair_qa_regenerated.json",
        ),
        ("--records-out", "phase4_run_output"),
        (
            "--snapshot-manifest",
            str(repo_root / phase4.SNAPSHOT_MANIFEST_RELPATH),
        ),
        ("--primary-model-path", str(snapshot_root / "primary")),
        ("--disjoint-model-path", str(snapshot_root / "disjoint")),
    ):
        if joined:
            command.append(f"{flag}={value}")
        else:
            command.extend((flag, value))
    for label in _OPERATOR_LABELS:
        value = (
            f"{label}={data_dir / _DATA_FILENAMES[label]}:"
            f"{phase4.R082_STAGED_INPUT_SHA256[label]}"
        )
        if joined:
            command.append(f"--staged-input={value}")
        else:
            command.extend(("--staged-input", value))
    return command


def _valid_external_components(
    tmp_path: Path, *, joined: bool = False
) -> dict[str, object]:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    calibration = tmp_path / "external" / "calibration_train.json"
    eligibility_path = repo_root / phase4.ELIGIBILITY_ARTIFACT_RELPATH
    return _components(
        repo_root,
        calibration,
        _bound_command(
            calibration, eligibility_path, repo_root, joined=joined
        ),
    )


def _assert_r082_failure(certificate: dict[str, object], token: str) -> None:
    assert certificate["ready"] is False
    failures = certificate["failing_checks"]
    assert isinstance(failures, list)
    assert any(
        "R-082 external_staging" in failure and token in failure
        for failure in failures
    ), failures


def _assert_coverage_failure(
    certificate: dict[str, object], token: str
) -> None:
    assert certificate["ready"] is False
    failures = certificate["failing_checks"]
    assert isinstance(failures, list)
    assert any(
        "R-082 staged_coverage" in failure and token in failure
        for failure in failures
    ), failures


def _assert_launch_path_failure(
    certificate: dict[str, object], token: str
) -> None:
    assert certificate["ready"] is False
    failures = certificate["failing_checks"]
    assert isinstance(failures, list)
    assert any(
        "R-081 launch_paths" in failure and token in failure
        for failure in failures
    ), failures


def _assert_receipt_environment_failure(
    certificate: dict[str, object], token: str
) -> None:
    assert certificate["ready"] is False
    failures = certificate["failing_checks"]
    assert isinstance(failures, list)
    assert any(
        "R-070 receipt_environment" in failure and token in failure
        for failure in failures
    ), failures


def _assert_phase4_command_failure(
    certificate: dict[str, object], token: str
) -> None:
    assert certificate["ready"] is False
    failures = certificate["failing_checks"]
    assert isinstance(failures, list)
    assert any(
        "R-081 phase4_command" in failure and token in failure
        for failure in failures
    ), failures


def _assert_snapshot_manifest_failure(
    certificate: dict[str, object], token: str
) -> None:
    assert certificate["ready"] is False
    failures = certificate["failing_checks"]
    assert isinstance(failures, list)
    assert any(
        "R-075 snapshot_manifest" in failure and token in failure
        for failure in failures
    ), failures


def _assert_content_hash_failure(
    certificate: dict[str, object], token: str
) -> None:
    assert certificate["ready"] is False
    failures = certificate["failing_checks"]
    assert isinstance(failures, list)
    assert any(
        "content_hashes:" in failure and token in failure
        for failure in failures
    ), failures


@pytest.mark.parametrize(
    ("flag", "joined"),
    [
        ("--data-dir", False),
        ("--data-dir", True),
        ("--calibration", False),
        ("--calibration", True),
        ("--staged-input", False),
        ("--staged-input", True),
    ],
)
def test_exact_command_consumed_path_inside_repo_is_not_ready(
    tmp_path: Path, flag: str, joined: bool
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    outside = tmp_path / "outside" / "staged.json"
    inside = repo_root / "data" / "consumed.json"
    value = str(inside)
    if flag == "--staged-input":
        value = f"fit_split={inside}:{_SHA}"
    option = [f"{flag}={value}"] if joined else [flag, value]
    components = _components(
        repo_root,
        outside,
        ["python", "producer.py", *option],
    )

    certificate = phase4.assemble_certificate(components)

    _assert_r082_failure(certificate, flag)


def test_staged_component_path_inside_repo_is_not_ready(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    components = _components(
        repo_root,
        repo_root / "staging" / "input.json",
        ["python", "producer.py"],
    )

    certificate = phase4.assemble_certificate(components)

    _assert_r082_failure(certificate, "staged_inputs[0].path")


def test_relative_paths_are_resolved_from_exact_command_repo_cwd(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    outside = tmp_path / "outside" / "staged.json"
    components = _components(
        repo_root,
        outside,
        ["python", "producer.py", "--data-dir", "data/processed"],
    )

    certificate = phase4.assemble_certificate(components)

    _assert_r082_failure(certificate, "data/processed")


@pytest.mark.parametrize(
    "command",
    [
        ["python", "producer.py", "--data-dir"],
        ["python", "producer.py", "--calibration="],
        ["python", "producer.py", "--staged-input", "not-a-triple"],
    ],
)
def test_malformed_consumed_path_flag_fails_closed(
    tmp_path: Path, command: list[str]
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    components = _components(
        repo_root,
        tmp_path / "outside" / "staged.json",
        command,
    )

    certificate = phase4.assemble_certificate(components)

    _assert_r082_failure(certificate, "exact command")


def test_missing_repo_root_binding_fails_closed(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    components = _components(
        repo_root,
        tmp_path / "outside" / "staged.json",
        ["python", "producer.py"],
    )
    del components["repo"]["root_realpath"]

    certificate = phase4.assemble_certificate(components)

    _assert_r082_failure(certificate, "root_realpath")


def test_repo_untracked_disclosure_is_required(tmp_path: Path) -> None:
    components = _valid_external_components(tmp_path)
    del components["repo"]["untracked_disclosure"]

    certificate = phase4.assemble_certificate(components)

    assert certificate["ready"] is False
    assert any(
        "repo: required field 'untracked_disclosure' missing" == failure
        for failure in certificate["failing_checks"]
    )


@pytest.mark.parametrize(
    ("disclosure", "token"),
    [
        ("scratch.py", "must be a list"),
        (["../escape.py"], "normalized forward-slash repo-relative"),
        (["C:/escape.py"], "normalized forward-slash repo-relative"),
        (["scratch.py", "scratch.py"], "entries must be unique"),
        (["z.py", "a.py"], "entries must be sorted"),
    ],
)
def test_repo_untracked_disclosure_is_closed_and_normalized(
    tmp_path: Path, disclosure: object, token: str
) -> None:
    components = _valid_external_components(tmp_path)
    components["repo"]["untracked_disclosure"] = disclosure

    certificate = phase4.assemble_certificate(components)

    assert certificate["ready"] is False
    assert any(
        "repo: untracked_disclosure" in failure and token in failure
        for failure in certificate["failing_checks"]
    )


def test_untracked_porcelain_parser_preserves_nul_delimited_paths() -> None:
    status = b"?? z.py\x00?? folder/name with spaces.py\x00"

    observed = phase4.parse_untracked_porcelain_v1_z(status)

    assert observed == ["folder/name with spaces.py", "z.py"]


@pytest.mark.parametrize(
    "status",
    [
        "?? a.py\n?? b.py\n",
        " M tracked.py\x00?? untracked.py\x00",
    ],
)
def test_untracked_porcelain_parser_rejects_collapsed_or_mixed_status(
    status: str,
) -> None:
    with pytest.raises(phase4.schema.ColmAimsError):
        phase4.parse_untracked_porcelain_v1_z(status)


@pytest.mark.parametrize("joined", [False, True])
def test_exact_six_external_bindings_are_ready(
    tmp_path: Path, joined: bool
) -> None:
    components = _valid_external_components(tmp_path, joined=joined)

    certificate = phase4.assemble_certificate(components)

    assert certificate["ready"] is True, certificate["failing_checks"]


@pytest.mark.parametrize("mutation", ["missing", "unknown"])
def test_content_hashes_require_closed_canonical_source_set(
    tmp_path: Path, mutation: str
) -> None:
    components = _valid_external_components(tmp_path)
    if mutation == "missing":
        components["content_hashes"].pop("producer_sha256")
        token = "missing required key(s) ['producer_sha256']"
    else:
        components["content_hashes"]["substitute_sha256"] = {
            "artifact_path": str((tmp_path / "substitute.py").resolve()),
            "sha256": _SHA,
        }
        token = "unexpected key(s) ['substitute_sha256']"

    certificate = phase4.assemble_certificate(components)

    _assert_content_hash_failure(certificate, token)


@pytest.mark.parametrize(
    ("field", "value", "token"),
    [
        ("artifact_path", "substitute.py", "canonical repo artifact"),
        ("sha256", "not-a-digest", "sha256 is not a sha256 hex digest"),
    ],
)
def test_content_hash_entry_substitution_is_not_ready(
    tmp_path: Path, field: str, value: str, token: str
) -> None:
    components = _valid_external_components(tmp_path)
    entry = components["content_hashes"]["producer_sha256"]
    entry[field] = (
        str((tmp_path / value).resolve())
        if field == "artifact_path"
        else value
    )

    certificate = phase4.assemble_certificate(components)

    _assert_content_hash_failure(certificate, token)


@pytest.mark.parametrize("mutation", ["missing_field", "extra_field"])
def test_content_hash_entries_are_closed_objects(
    tmp_path: Path, mutation: str
) -> None:
    components = _valid_external_components(tmp_path)
    entry = components["content_hashes"]["producer_sha256"]
    if mutation == "missing_field":
        del entry["artifact_path"]
        token = "missing field(s) ['artifact_path']"
    else:
        entry["caller_path"] = str(tmp_path / "substitute.py")
        token = "unexpected field(s) ['caller_path']"

    certificate = phase4.assemble_certificate(components)

    _assert_content_hash_failure(certificate, token)


def test_gather_content_hashes_binds_and_hashes_only_canonical_sources(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    configured = _materialize_audited_sources(repo_root)

    gathered = phase4._gather_content_hashes(repo_root, configured)

    assert set(gathered) == set(phase4.CONTENT_HASH_RELPATHS)
    for key, path in configured.items():
        assert gathered[key] == {
            "artifact_path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }


def test_gather_content_hashes_refuses_caller_path_substitution(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    configured = _materialize_audited_sources(repo_root)
    substitute = tmp_path / "substitute-producer.py"
    substitute.write_bytes(b"caller-selected substitute\n")
    configured["producer_sha256"] = substitute

    with pytest.raises(
        phase4.schema.ColmAimsError,
        match="producer_sha256.*canonical repo artifact",
    ):
        phase4._gather_content_hashes(repo_root, configured)


@pytest.mark.parametrize("mutation", ["missing", "unknown"])
def test_gather_content_hashes_refuses_noncanonical_key_set(
    tmp_path: Path, mutation: str
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    configured = _materialize_audited_sources(repo_root)
    if mutation == "missing":
        configured.pop("producer_sha256")
    else:
        configured["caller_selected_sha256"] = tmp_path / "substitute.py"

    with pytest.raises(
        phase4.schema.ColmAimsError,
        match="exact canonical key set",
    ):
        phase4._gather_content_hashes(repo_root, configured)


@pytest.mark.parametrize(
    ("canonical", "unsupported"),
    [
        ("--data-dir", "--data_dir"),
        ("--staged-input", "--staged_input"),
        ("--fit-split", "--fit_split"),
        ("--eval-split", "--eval_split"),
        ("--reward-schedule", "--reward_schedule"),
        ("--qa-arms", "--qa_arms"),
        ("--num-bootstrap", "--num_bootstrap"),
        ("--n-test", "--n_test"),
        ("--n-val", "--n_val"),
        ("--records-out", "--records_out"),
        ("--snapshot-manifest", "--snapshot_manifest"),
        ("--primary-model-path", "--primary_model_path"),
        ("--disjoint-model-path", "--disjoint_model_path"),
    ],
)
@pytest.mark.parametrize("joined", [False, True])
def test_underscore_flag_aliases_are_rejected_by_certificate(
    tmp_path: Path,
    canonical: str,
    unsupported: str,
    joined: bool,
) -> None:
    components = _valid_external_components(tmp_path, joined=joined)
    command = components["environment"]["command"]
    for index, token in enumerate(command):
        if token == canonical or token.startswith(f"{canonical}="):
            command[index] = token.replace(canonical, unsupported, 1)
            break
    else:
        raise AssertionError(f"missing canonical test flag {canonical}")

    certificate = phase4.assemble_certificate(components)

    _assert_phase4_command_failure(
        certificate, f"{unsupported}: unsupported"
    )


@pytest.mark.parametrize(
    "flag",
    [
        "--reward-schedule",
        "--qa-arms",
        "--calibrations",
        "--num-bootstrap",
        "--n-test",
        "--n-val",
        "--seed",
        "--out",
        "--records-out",
        "--snapshot-manifest",
        "--primary-model-path",
        "--disjoint-model-path",
    ],
)
def test_phase4_run_shape_flag_is_explicit_and_unique(
    tmp_path: Path, flag: str
) -> None:
    components = _valid_external_components(tmp_path)
    command = components["environment"]["command"]
    index = command.index(flag)
    del command[index : index + 2]

    certificate = phase4.assemble_certificate(components)

    _assert_phase4_command_failure(
        certificate, f"{flag} must appear exactly once"
    )


@pytest.mark.parametrize(
    "flag",
    [
        "--reward-schedule",
        "--qa-arms",
        "--calibrations",
        "--num-bootstrap",
        "--n-test",
        "--n-val",
        "--seed",
        "--out",
        "--records-out",
        "--snapshot-manifest",
        "--primary-model-path",
        "--disjoint-model-path",
    ],
)
def test_duplicate_phase4_run_shape_flag_is_rejected(
    tmp_path: Path, flag: str
) -> None:
    components = _valid_external_components(tmp_path)
    command = components["environment"]["command"]
    index = command.index(flag)
    command.extend((flag, command[index + 1]))

    certificate = phase4.assemble_certificate(components)

    _assert_phase4_command_failure(
        certificate, f"{flag} must appear exactly once"
    )


@pytest.mark.parametrize(
    ("flag", "invalid"),
    [
        ("--reward-schedule", "linear"),
        ("--qa-arms", "idealized,krandom,khard,kdisjoint"),
        (
            "--qa-arms",
            "idealized,krandom,khard,kdisjoint,klex,klex",
        ),
        ("--calibrations", "shared"),
        ("--calibrations", "shared,performat,shared"),
        ("--num-bootstrap", "999"),
        ("--n-test", "1"),
        ("--n-val", "1"),
        ("--seed", "2"),
        ("--out", ""),
        ("--out", "phase4_run_output"),
        ("--out", "NUL"),
        ("--records-out", ""),
        ("--records-out", "records"),
        ("--snapshot-manifest", ""),
        ("--primary-model-path", ""),
        ("--disjoint-model-path", ""),
    ],
)
def test_invalid_phase4_run_shape_value_is_rejected(
    tmp_path: Path, flag: str, invalid: str
) -> None:
    components = _valid_external_components(tmp_path)
    command = components["environment"]["command"]
    command[command.index(flag) + 1] = invalid

    certificate = phase4.assemble_certificate(components)

    _assert_phase4_command_failure(certificate, flag)


@pytest.mark.parametrize("joined", [False, True])
def test_certificate_digest_is_reserved_for_launcher(
    tmp_path: Path, joined: bool
) -> None:
    components = _valid_external_components(tmp_path)
    command = components["environment"]["command"]
    if joined:
        command.append(f"--certificate-digest={'0' * 64}")
    else:
        command.extend(("--certificate-digest", "0" * 64))

    certificate = phase4.assemble_certificate(components)

    _assert_phase4_command_failure(
        certificate, "must not contain --certificate-digest"
    )


def test_producer_command_interpreter_must_match_certified_runtime(
    tmp_path: Path,
) -> None:
    components = _valid_external_components(tmp_path)
    components["environment"]["command"][0] = str(
        tmp_path / "different-runtime" / "python"
    )

    certificate = phase4.assemble_certificate(components)

    _assert_phase4_command_failure(
        certificate, "does not resolve to certified interpreter_realpath"
    )


def test_producer_script_argv_is_exactly_pinned(tmp_path: Path) -> None:
    components = _valid_external_components(tmp_path)
    components["environment"]["command"][1] = "producer.py"

    certificate = phase4.assemble_certificate(components)

    _assert_phase4_command_failure(certificate, "command[1] must be exactly")


@pytest.mark.parametrize(
    "extra",
    [
        ["--unknown-option", "value"],
        ["--num-boot=1000"],
        ["stray-positional"],
    ],
)
def test_non_allowlisted_abbreviated_and_positional_argv_is_rejected(
    tmp_path: Path, extra: list[str]
) -> None:
    components = _valid_external_components(tmp_path)
    components["environment"]["command"].extend(extra)

    certificate = phase4.assemble_certificate(components)

    _assert_phase4_command_failure(certificate, "exact command")


@pytest.mark.parametrize(
    ("field", "invalid", "token"),
    [
        ("seeds", [2], "seeds must be exactly [1]"),
        ("pythonhashseed", "1", "pythonhashseed must be exactly '0'"),
        (
            "thread_settings",
            {"OMP_NUM_THREADS": "1"},
            "thread_settings must equal the exact Phase-4 pin map",
        ),
    ],
)
def test_environment_determinism_pins_are_exact(
    tmp_path: Path, field: str, invalid: object, token: str
) -> None:
    components = _valid_external_components(tmp_path)
    components["environment"][field] = invalid

    certificate = phase4.assemble_certificate(components)

    assert certificate["ready"] is False
    assert any(token in failure for failure in certificate["failing_checks"])


@pytest.mark.parametrize("mutation", ["missing", "wrong"])
def test_python_no_user_site_is_in_the_exact_signed_environment_map(
    tmp_path: Path, mutation: str
) -> None:
    components = _valid_external_components(tmp_path)
    settings = components["environment"]["thread_settings"]
    if mutation == "missing":
        del settings["PYTHONNOUSERSITE"]
    else:
        settings["PYTHONNOUSERSITE"] = "0"

    certificate = phase4.assemble_certificate(components)

    assert certificate["ready"] is False
    assert any(
        "thread_settings must equal the exact Phase-4 pin map" in failure
        for failure in certificate["failing_checks"]
    )


@pytest.mark.parametrize("field", phase4.R081_LAUNCH_PATH_FIELDS)
def test_launch_path_environment_fields_are_required(
    tmp_path: Path, field: str
) -> None:
    components = _valid_external_components(tmp_path)
    del components["environment"][field]

    certificate = phase4.assemble_certificate(components)

    _assert_launch_path_failure(certificate, field)
    assert any(
        f"environment: required field {field!r} missing" == failure
        for failure in certificate["failing_checks"]
    )


@pytest.mark.parametrize("field", phase4.R081_LAUNCH_PATH_FIELDS)
def test_launch_paths_must_be_absolute(
    tmp_path: Path, field: str
) -> None:
    components = _valid_external_components(tmp_path)
    components["environment"][field] = f"relative/{field}"

    certificate = phase4.assemble_certificate(components)

    _assert_launch_path_failure(certificate, f"{field} must be an absolute")


@pytest.mark.parametrize("field", phase4.R081_LAUNCH_PATH_FIELDS)
def test_launch_paths_must_use_their_canonical_resolved_spelling(
    tmp_path: Path, field: str
) -> None:
    components = _valid_external_components(tmp_path)
    original = Path(components["environment"][field])
    components["environment"][field] = str(
        original.parent / "dot-segment" / ".." / original.name
    )

    certificate = phase4.assemble_certificate(components)

    _assert_launch_path_failure(certificate, "canonical resolved path")


@pytest.mark.parametrize("field", phase4.R081_LAUNCH_PATH_FIELDS)
def test_launch_paths_must_be_external_to_operational_repo(
    tmp_path: Path, field: str
) -> None:
    components = _valid_external_components(tmp_path)
    repo_root = Path(components["repo"]["root_realpath"])
    components["environment"][field] = str(repo_root / field)

    certificate = phase4.assemble_certificate(components)

    _assert_launch_path_failure(certificate, f"{field}")
    _assert_launch_path_failure(certificate, "inside the repository root")


@pytest.mark.parametrize("relation", ["equal", "quarantine_nested", "promote_nested"])
def test_quarantine_and_promotion_paths_must_be_disjoint(
    tmp_path: Path, relation: str
) -> None:
    components = _valid_external_components(tmp_path)
    environment = components["environment"]
    quarantine = Path(environment["quarantine_dir"])
    promote = Path(environment["promote_to"])
    if relation == "equal":
        environment["promote_to"] = str(quarantine)
    elif relation == "quarantine_nested":
        environment["quarantine_dir"] = str(promote / "nested")
    else:
        environment["promote_to"] = str(quarantine / "nested")

    certificate = phase4.assemble_certificate(components)

    _assert_launch_path_failure(certificate, "must be disjoint")


@pytest.mark.parametrize("workspace_field", ["quarantine_dir", "promote_to"])
@pytest.mark.parametrize("relation", ["equal", "ledger_nested", "ledger_ancestor"])
def test_exception_ledger_must_be_disjoint_from_each_workspace(
    tmp_path: Path, workspace_field: str, relation: str
) -> None:
    components = _valid_external_components(tmp_path)
    environment = components["environment"]
    workspace = Path(environment[workspace_field])
    if relation == "equal":
        environment["exception_ledger_path"] = str(workspace)
    elif relation == "ledger_nested":
        environment["exception_ledger_path"] = str(
            workspace / "ledger.json"
        )
    else:
        ledger = workspace.parent / f"{workspace_field}-ledger-ancestor"
        environment["exception_ledger_path"] = str(ledger)
        environment[workspace_field] = str(ledger / "nested-workspace")

    certificate = phase4.assemble_certificate(components)

    _assert_launch_path_failure(certificate, "exception_ledger_path")
    _assert_launch_path_failure(certificate, "disjoint")


@pytest.mark.parametrize("suite", phase4.SUITE_RECEIPT_NAMES)
@pytest.mark.parametrize(
    ("field", "token"),
    [
        ("interpreter_realpath", "interpreter_realpath"),
        ("environment_lock_sha256", "environment_lock_sha256"),
        ("command", "command[0]"),
    ],
)
def test_each_suite_receipt_is_bound_to_certificate_runtime(
    tmp_path: Path, suite: str, field: str, token: str
) -> None:
    components = _valid_external_components(tmp_path)
    receipt = components["suite_receipts"][suite]
    if field == "interpreter_realpath":
        receipt[field] = str(tmp_path / "different-runtime" / "python")
    elif field == "environment_lock_sha256":
        receipt[field] = "0" * 64
    else:
        receipt[field][0] = str(tmp_path / "different-runtime" / "python")

    certificate = phase4.assemble_certificate(components)

    _assert_receipt_environment_failure(certificate, f"{suite} receipt")
    _assert_receipt_environment_failure(certificate, token)


def test_receipt_command_interpreter_may_use_equivalent_normalized_path(
    tmp_path: Path,
) -> None:
    components = _valid_external_components(tmp_path)
    interpreter = Path(
        components["environment"]["interpreter_realpath"]
    )
    equivalent = interpreter.parent / "nested" / ".." / interpreter.name
    for receipt in components["suite_receipts"].values():
        receipt["command"][0] = str(equivalent)

    certificate = phase4.assemble_certificate(components)

    assert certificate["ready"] is True, certificate["failing_checks"]


@pytest.mark.parametrize("suite", phase4.SUITE_RECEIPT_NAMES)
@pytest.mark.parametrize("field", ["junit_sha256", "transcript_sha256"])
def test_suite_receipt_evidence_digests_are_required_and_typed(
    tmp_path: Path, suite: str, field: str
) -> None:
    components = _valid_external_components(tmp_path)
    del components["suite_receipts"][suite][field]

    certificate = phase4.assemble_certificate(components)

    assert certificate["ready"] is False
    assert any(
        f"suite_receipts: {suite}" in failure and field in failure
        for failure in certificate["failing_checks"]
    )


@pytest.mark.parametrize("field", ["junit_sha256", "transcript_sha256"])
def test_suite_receipt_evidence_digest_rejects_malformed_value(
    tmp_path: Path, field: str
) -> None:
    components = _valid_external_components(tmp_path)
    components["suite_receipts"]["full"][field] = "not-a-digest"

    certificate = phase4.assemble_certificate(components)

    assert certificate["ready"] is False
    assert any(
        "suite_receipts: full" in failure and field in failure
        for failure in certificate["failing_checks"]
    )


@pytest.mark.parametrize(
    "counts",
    [
        {"tests": 0, "failures": 0, "errors": 0, "skipped": 0},
        {"tests": 2, "failures": 0, "errors": 0, "skipped": 2},
        {"tests": True, "failures": 0, "errors": 0, "skipped": 0},
        {
            "tests": 2,
            "failures": 0,
            "errors": 0,
            "skipped": 0,
            "passed": 2,
        },
    ],
)
def test_suite_receipt_counts_cannot_be_vacuous_or_laundered(
    tmp_path: Path, counts: dict[str, object]
) -> None:
    components = _valid_external_components(tmp_path)
    components["suite_receipts"]["focused"]["counts"] = counts

    certificate = phase4.assemble_certificate(components)

    assert certificate["ready"] is False
    assert any(
        "suite_receipts: focused counts" in failure
        for failure in certificate["failing_checks"]
    )


@pytest.mark.parametrize(
    "mutation",
    ["python-c", "wrong-focused", "wrong-full", "unknown-option"],
)
def test_suite_receipt_command_cannot_substitute_a_vacuous_pass(
    tmp_path: Path, mutation: str
) -> None:
    components = _valid_external_components(tmp_path)
    environment = components["environment"]
    interpreter = environment["interpreter_realpath"]
    if mutation == "python-c":
        suite = "focused"
        command = [interpreter, "-c", "pass"]
    elif mutation == "wrong-focused":
        suite = "focused"
        command = list(components["suite_receipts"][suite]["command"])
        command[3] = "tests/unrelated.py"
    elif mutation == "wrong-full":
        suite = "full"
        command = list(components["suite_receipts"][suite]["command"])
        command[3] = "tests/test_phase4_certificate_external_staging.py"
    else:
        suite = "full"
        command = list(components["suite_receipts"][suite]["command"])
        command.insert(-1, "--collect-only")
    components["suite_receipts"][suite]["command"] = command

    certificate = phase4.assemble_certificate(components)

    _assert_receipt_environment_failure(certificate, f"{suite} receipt command")


@pytest.mark.parametrize("label", _STAGED_LABELS)
def test_each_required_component_label_is_mandatory(
    tmp_path: Path, label: str
) -> None:
    components = _valid_external_components(tmp_path)
    components["staged_inputs"] = [
        entry
        for entry in components["staged_inputs"]
        if entry["label"] != label
    ]

    certificate = phase4.assemble_certificate(components)

    _assert_coverage_failure(certificate, f"missing required label {label!r}")


def test_duplicate_component_label_is_rejected(tmp_path: Path) -> None:
    components = _valid_external_components(tmp_path)
    duplicate = dict(components["staged_inputs"][2])
    components["staged_inputs"].append(duplicate)

    certificate = phase4.assemble_certificate(components)

    _assert_coverage_failure(certificate, "duplicate component label 'fit_split'")


def test_unknown_component_label_is_rejected(tmp_path: Path) -> None:
    components = _valid_external_components(tmp_path)
    components["staged_inputs"][2]["label"] = "unknown_input"

    certificate = phase4.assemble_certificate(components)

    _assert_coverage_failure(certificate, "unknown component label 'unknown_input'")


@pytest.mark.parametrize("label", _STAGED_LABELS)
def test_component_path_must_match_exact_command_binding(
    tmp_path: Path, label: str
) -> None:
    components = _valid_external_components(tmp_path)
    entry = next(
        item for item in components["staged_inputs"] if item["label"] == label
    )
    entry["path"] = str(tmp_path / "external" / f"wrong-{label}.json")

    certificate = phase4.assemble_certificate(components)

    _assert_coverage_failure(certificate, f"component {label!r} path")


@pytest.mark.parametrize(
    "flag",
    [
        "--data-dir",
        "--calibration",
        "--eligibility",
        "--fit-split",
        "--eval-split",
    ],
)
def test_required_path_option_must_appear_exactly_once(
    tmp_path: Path, flag: str
) -> None:
    components = _valid_external_components(tmp_path)
    command = components["environment"]["command"]
    index = command.index(flag)
    del command[index : index + 2]

    certificate = phase4.assemble_certificate(components)

    _assert_coverage_failure(certificate, f"{flag} must appear exactly once")


@pytest.mark.parametrize(
    "flag",
    [
        "--data-dir",
        "--calibration",
        "--eligibility",
        "--fit-split",
        "--eval-split",
    ],
)
def test_duplicate_path_option_is_rejected(tmp_path: Path, flag: str) -> None:
    components = _valid_external_components(tmp_path)
    command = components["environment"]["command"]
    index = command.index(flag)
    command.extend((flag, command[index + 1]))

    certificate = phase4.assemble_certificate(components)

    _assert_coverage_failure(certificate, f"{flag} must appear exactly once")


@pytest.mark.parametrize(
    ("flag", "component_label", "changed_value"),
    [
        ("--fit-split", "fit_split", "train"),
        ("--eval-split", "eval_split", "holdout"),
    ],
)
def test_changed_selector_cannot_escape_component_binding(
    tmp_path: Path,
    flag: str,
    component_label: str,
    changed_value: str,
) -> None:
    components = _valid_external_components(tmp_path)
    command = components["environment"]["command"]
    command[command.index(flag) + 1] = changed_value

    certificate = phase4.assemble_certificate(components)

    _assert_coverage_failure(
        certificate, f"component {component_label!r} path"
    )
    required = "val" if component_label == "fit_split" else "test"
    _assert_coverage_failure(certificate, f"must equal {required!r}")


def test_eligibility_command_path_must_match_certified_artifact(
    tmp_path: Path,
) -> None:
    components = _valid_external_components(tmp_path)
    command = components["environment"]["command"]
    command[command.index("--eligibility") + 1] = str(
        tmp_path / "different-eligibility.json"
    )

    certificate = phase4.assemble_certificate(components)

    _assert_coverage_failure(
        certificate, "--eligibility path"
    )


def test_calibration_digest_must_equal_frozen_phase4_pin(
    tmp_path: Path,
) -> None:
    components = _valid_external_components(tmp_path)
    calibration = next(
        entry
        for entry in components["staged_inputs"]
        if entry["label"] == "calibration_train"
    )
    calibration["expected_sha256"] = _SHA
    calibration["observed_sha256"] = _SHA

    certificate = phase4.assemble_certificate(components)

    _assert_coverage_failure(certificate, "frozen pin")


@pytest.mark.parametrize("label", _STAGED_LABELS)
def test_self_consistent_alternate_staged_digest_is_rejected(
    tmp_path: Path, label: str
) -> None:
    components = _valid_external_components(tmp_path)
    alternate = "0" * 64
    entry = next(
        item for item in components["staged_inputs"] if item["label"] == label
    )
    entry["expected_sha256"] = alternate
    entry["observed_sha256"] = alternate
    if label == "eval_split":
        components["eligibility"]["test_dataset_sha256"] = alternate
    if label in _OPERATOR_LABELS:
        command = components["environment"]["command"]
        value_index = next(
            index + 1
            for index, token in enumerate(command)
            if token == "--staged-input"
            and command[index + 1].startswith(f"{label}=")
        )
        path = command[value_index].partition("=")[2].rpartition(":")[0]
        command[value_index] = f"{label}={path}:{alternate}"

    certificate = phase4.assemble_certificate(components)

    _assert_coverage_failure(
        certificate, f"component {label!r} expected_sha256"
    )
    _assert_coverage_failure(certificate, "frozen pin")


def test_eval_digest_must_match_eligibility_derived_test_pin(
    tmp_path: Path,
) -> None:
    components = _valid_external_components(tmp_path)
    eval_split = next(
        entry
        for entry in components["staged_inputs"]
        if entry["label"] == "eval_split"
    )
    eval_split["expected_sha256"] = _SHA
    eval_split["observed_sha256"] = _SHA

    certificate = phase4.assemble_certificate(components)

    _assert_coverage_failure(certificate, "test_dataset_sha256")


@pytest.mark.parametrize(
    "field",
    ["artifact_path", "artifact_sha256", "test_dataset_sha256"],
)
def test_eligibility_certificate_binding_fields_are_required(
    tmp_path: Path, field: str
) -> None:
    components = _valid_external_components(tmp_path)
    del components["eligibility"][field]

    certificate = phase4.assemble_certificate(components)

    assert certificate["ready"] is False
    assert any(
        f"eligibility: {field}" in failure
        or f"eligibility {field}" in failure
        for failure in certificate["failing_checks"]
    )


@pytest.mark.parametrize(
    "field",
    [
        "digest",
        "horizon_map_sha256",
        "artifact_sha256",
        "test_dataset_sha256",
    ],
)
def test_self_consistent_alternate_eligibility_pins_are_rejected(
    tmp_path: Path, field: str
) -> None:
    components = _valid_external_components(tmp_path)
    alternate = "0" * 64
    components["eligibility"][field] = alternate
    if field == "test_dataset_sha256":
        eval_split = next(
            entry
            for entry in components["staged_inputs"]
            if entry["label"] == "eval_split"
        )
        eval_split["expected_sha256"] = alternate
        eval_split["observed_sha256"] = alternate

    certificate = phase4.assemble_certificate(components)

    assert certificate["ready"] is False
    assert any(
        f"eligibility: {field}" in failure
        and "canonical" in failure
        for failure in certificate["failing_checks"]
    )


def test_alternate_eligibility_path_is_rejected_even_when_command_agrees(
    tmp_path: Path,
) -> None:
    components = _valid_external_components(tmp_path)
    alternate = (tmp_path / "repo" / "alternate-eligibility.json").resolve()
    components["eligibility"]["artifact_path"] = str(alternate)
    command = components["environment"]["command"]
    command[command.index("--eligibility") + 1] = str(alternate)

    certificate = phase4.assemble_certificate(components)

    _assert_coverage_failure(certificate, "not the canonical repo artifact")


@pytest.mark.parametrize("field", ["artifact_path", "artifact_sha256"])
def test_snapshot_manifest_certificate_binding_fields_are_required(
    tmp_path: Path, field: str
) -> None:
    components = _valid_external_components(tmp_path)
    del components["snapshots"][field]

    certificate = phase4.assemble_certificate(components)

    assert certificate["ready"] is False
    assert any(
        f"snapshots: {field}" in failure
        or f"snapshots {field}" in failure
        for failure in certificate["failing_checks"]
    )


def test_snapshot_manifest_command_path_must_match_certified_artifact(
    tmp_path: Path,
) -> None:
    components = _valid_external_components(tmp_path)
    command = components["environment"]["command"]
    command[command.index("--snapshot-manifest") + 1] = str(
        tmp_path / "different-snapshot-manifest.json"
    )

    certificate = phase4.assemble_certificate(components)

    _assert_snapshot_manifest_failure(certificate, "does not match")


@pytest.mark.parametrize("field", ["artifact_path", "artifact_sha256"])
def test_alternate_snapshot_manifest_identity_is_rejected(
    tmp_path: Path, field: str
) -> None:
    components = _valid_external_components(tmp_path)
    if field == "artifact_path":
        alternate = (tmp_path / "repo" / "alternate-manifest.json").resolve()
        components["snapshots"][field] = str(alternate)
        command = components["environment"]["command"]
        command[command.index("--snapshot-manifest") + 1] = str(alternate)
    else:
        components["snapshots"][field] = "0" * 64

    certificate = phase4.assemble_certificate(components)

    assert certificate["ready"] is False
    assert any(
        "snapshots" in failure and "canonical" in failure
        for failure in certificate["failing_checks"]
    )


@pytest.mark.parametrize("role", phase4.SNAPSHOT_ROLES)
@pytest.mark.parametrize("field", ["model_name", "hf_revision"])
def test_snapshot_role_identity_must_match_loaded_canonical_manifest(
    tmp_path: Path, role: str, field: str
) -> None:
    components = _valid_external_components(tmp_path)
    components["snapshots"][role][field] = (
        "alternate-model" if field == "model_name" else "0" * 40
    )

    certificate = phase4.assemble_certificate(components)

    assert certificate["ready"] is False
    assert any(
        f"snapshots: {role} {field}" in failure
        and "canonical loaded-manifest identity" in failure
        for failure in certificate["failing_checks"]
    )


@pytest.mark.parametrize(
    "field",
    [
        "comparator_identity",
        "anchor_sha256",
        "source_export_a_sha256",
        "artifact_path",
    ],
)
def test_alternate_parity_anchor_binding_is_rejected(
    tmp_path: Path, field: str
) -> None:
    components = _valid_external_components(tmp_path)
    if field == "artifact_path":
        components["parity"][field] = str(
            (tmp_path / "repo" / "alternate-anchor.json").resolve()
        )
    elif field == "comparator_identity":
        components["parity"][field] = "alternate.compare"
    else:
        components["parity"][field] = "0" * 64

    certificate = phase4.assemble_certificate(components)

    assert certificate["ready"] is False
    assert any(
        "parity" in failure and field in failure
        for failure in certificate["failing_checks"]
    )


@pytest.mark.parametrize(
    ("field", "alternate"),
    [
        ("manifest_sha256", "0" * 64),
        ("artifact_path", "alternate-qa012.json"),
        ("manifest_type", "alternate_inventory"),
        ("revision", 2),
        ("conventions", {}),
    ],
)
def test_rev2_or_alternate_qa012_manifest_is_rejected(
    tmp_path: Path, field: str, alternate: object
) -> None:
    components = _valid_external_components(tmp_path)
    if field == "artifact_path":
        alternate = str((tmp_path / "repo" / str(alternate)).resolve())
    components["qa012"][field] = alternate

    certificate = phase4.assemble_certificate(components)

    assert certificate["ready"] is False
    assert any(
        "qa012" in failure and field in failure
        for failure in certificate["failing_checks"]
    )


@pytest.mark.parametrize("flag", ["--fit-split", "--eval-split"])
def test_selector_must_be_a_simple_nonempty_token(
    tmp_path: Path, flag: str
) -> None:
    components = _valid_external_components(tmp_path)
    command = components["environment"]["command"]
    command[command.index(flag) + 1] = "../escape"

    certificate = phase4.assemble_certificate(components)

    _assert_coverage_failure(certificate, "simple non-empty split token")


@pytest.mark.parametrize("label", _OPERATOR_LABELS)
def test_each_operator_digest_binding_is_mandatory(
    tmp_path: Path, label: str
) -> None:
    components = _valid_external_components(tmp_path)
    command = components["environment"]["command"]
    index = next(
        index
        for index, token in enumerate(command)
        if token == "--staged-input" and command[index + 1].startswith(f"{label}=")
    )
    del command[index : index + 2]

    certificate = phase4.assemble_certificate(components)

    _assert_coverage_failure(certificate, f"missing operator label {label!r}")


def test_duplicate_operator_label_is_rejected(tmp_path: Path) -> None:
    components = _valid_external_components(tmp_path)
    command = components["environment"]["command"]
    index = next(
        index
        for index, token in enumerate(command)
        if token == "--staged-input"
        and command[index + 1].startswith("fit_split=")
    )
    command.extend(("--staged-input", command[index + 1]))

    certificate = phase4.assemble_certificate(components)

    _assert_coverage_failure(certificate, "duplicate operator label 'fit_split'")


def test_unknown_operator_label_is_rejected(tmp_path: Path) -> None:
    components = _valid_external_components(tmp_path)
    command = components["environment"]["command"]
    command.extend(
        (
            "--staged-input",
            f"unknown_input={tmp_path / 'external' / 'unknown.json'}:{_SHA}",
        )
    )

    certificate = phase4.assemble_certificate(components)

    _assert_coverage_failure(certificate, "unknown operator label 'unknown_input'")


@pytest.mark.parametrize("contradiction", ["path", "digest"])
def test_operator_binding_must_match_component_exactly(
    tmp_path: Path, contradiction: str
) -> None:
    components = _valid_external_components(tmp_path)
    command = components["environment"]["command"]
    value_index = next(
        index + 1
        for index, token in enumerate(command)
        if token == "--staged-input"
        and command[index + 1].startswith("fit_split=")
    )
    if contradiction == "path":
        command[value_index] = (
            f"fit_split={tmp_path / 'external' / 'wrong-fit.json'}:{_SHA}"
        )
    else:
        path = command[value_index].partition("=")[2].rpartition(":")[0]
        command[value_index] = f"fit_split={path}:{'f' * 64}"

    certificate = phase4.assemble_certificate(components)

    _assert_coverage_failure(
        certificate, f"operator 'fit_split' {contradiction}"
    )


def test_generator_writes_not_ready_for_in_repo_consumed_path(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    components = _components(
        repo_root,
        tmp_path / "outside" / "staged.json",
        ["python", "producer.py", "--calibration", "staged/input.json"],
    )
    certificate_path = tmp_path / "pre_run_ready.json"

    result = phase4.generate_pre_run_ready(components, certificate_path)

    assert result["ready"] is False
    written = json.loads(certificate_path.read_text(encoding="utf-8"))
    _assert_r082_failure(written, "--calibration")


def test_gather_binds_configured_repo_root_and_blocks_in_repo_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    source_repo = Path(phase4.__file__).resolve().parents[2]
    canonical_paths: dict[str, Path] = {}
    for name, relpath in (
        ("eligibility", phase4.ELIGIBILITY_ARTIFACT_RELPATH),
        ("snapshots", phase4.SNAPSHOT_MANIFEST_RELPATH),
        ("parity", phase4.PARITY_ANCHOR_RELPATH),
        ("qa012", phase4.QA012_MANIFEST_RELPATH),
    ):
        source = source_repo / relpath
        destination = repo_root / relpath
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        canonical_paths[name] = destination
    staged = repo_root / "staged" / "input.json"
    staged.parent.mkdir()
    staged.write_bytes(b"synthetic staged bytes\n")
    staged_sha = hashlib.sha256(staged.read_bytes()).hexdigest()
    content_paths = _materialize_audited_sources(repo_root)

    monkeypatch.setattr(
        phase4,
        "load_pairing_eligibility",
        lambda _path: {
            "pairing_population_keyset_sha256": phase4.ELIGIBILITY_KEYSET_SHA256,
            "horizon_map_sha256": phase4.ELIGIBILITY_HORIZON_MAP_SHA256,
            "derived_from": {
                "test_dataset_sha256": _TEST_DATASET_SHA,
            },
        },
    )
    monkeypatch.setattr(
        phase4,
        "load_model_snapshot_manifest",
        lambda _path: {
            "roles": {
                "primary_scorer": {
                    **phase4.EXPECTED_SNAPSHOT_IDENTITIES["primary_scorer"],
                },
                "disjoint_selector": {
                    **phase4.EXPECTED_SNAPSHOT_IDENTITIES[
                        "disjoint_selector"
                    ],
                },
            }
        },
    )
    monkeypatch.setattr(phase4, "verify_snapshot_dir", lambda _role, _path: None)

    receipt_paths: dict[str, Path] = {}
    receipt_interpreter = str(
        (tmp_path / "synthetic-runtime" / "python").resolve()
    )
    for name in phase4.SUITE_RECEIPT_NAMES:
        receipt_path = tmp_path / f"{name}.json"
        receipt_path.write_text(
            json.dumps(_suite_receipt(name, receipt_interpreter)),
            encoding="utf-8",
        )
        receipt_paths[name] = receipt_path

    commands: list[list[str]] = []

    def run(command: list[str]) -> str:
        commands.append(list(command))
        if "status" in command:
            if "--untracked-files=all" in command:
                return "?? z.py\x00?? a/b.py\x00"
            return ""
        if "HEAD^{tree}" in command:
            return _TREE + "\n"
        if "rev-parse" in command:
            return _COMMIT + "\n"
        raise AssertionError(f"unexpected command: {command}")

    config = {
        "repo_root": repo_root,
        "eligibility_path": canonical_paths["eligibility"],
        "snapshot_manifest_path": canonical_paths["snapshots"],
        "snapshot_dirs": {
            "primary_scorer": tmp_path / "primary",
            "disjoint_selector": tmp_path / "selector",
        },
        "parity_anchor_path": canonical_paths["parity"],
        "qa012_manifest_path": canonical_paths["qa012"],
        "staged_plan": [
            {
                "label": "synthetic_input",
                "path": staged,
                "expected_sha256": staged_sha,
            }
        ],
        "suite_receipt_paths": receipt_paths,
        "content_hash_paths": content_paths,
        "environment": _components(
            repo_root,
            tmp_path / "outside" / "staged.json",
            ["python", "producer.py"],
        )["environment"],
        "offline_flags": list(phase4.REQUIRED_OFFLINE_FLAGS),
    }

    components = phase4.gather_certificate_components(config, run=run)
    certificate = phase4.assemble_certificate(components)

    assert components["repo"]["root_realpath"] == str(repo_root.resolve())
    assert components["repo"]["untracked_disclosure"] == ["a/b.py", "z.py"]
    assert [
        "git",
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=no",
    ] in commands
    assert [
        "git",
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    ] in commands
    for key, path in content_paths.items():
        assert components["content_hashes"][key] == {
            "artifact_path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    assert components["eligibility"] == {
        "digest": phase4.ELIGIBILITY_KEYSET_SHA256,
        "horizon_map_sha256": phase4.ELIGIBILITY_HORIZON_MAP_SHA256,
        "artifact_path": str(canonical_paths["eligibility"].resolve()),
        "artifact_sha256": phase4.ELIGIBILITY_ARTIFACT_SHA256,
        "test_dataset_sha256": _TEST_DATASET_SHA,
    }
    assert components["snapshots"]["artifact_path"] == str(
        canonical_paths["snapshots"].resolve()
    )
    assert (
        components["snapshots"]["artifact_sha256"]
        == phase4.SNAPSHOT_MANIFEST_SHA256
    )
    assert components["parity"] == {
        "comparator_identity": phase4.PARITY_COMPARATOR_IDENTITY,
        "artifact_path": str(canonical_paths["parity"].resolve()),
        "anchor_sha256": phase4.PARITY_ANCHOR_SHA256,
        "source_export_a_sha256": phase4.PARITY_SOURCE_EXPORT_A_SHA256,
    }
    assert components["qa012"] == {
        "artifact_path": str(canonical_paths["qa012"].resolve()),
        "manifest_sha256": phase4.QA012_MANIFEST_SHA256,
        "manifest_type": phase4.QA012_MANIFEST_TYPE,
        "revision": phase4.QA012_MANIFEST_REVISION,
        "conventions": phase4.QA012_CONVENTIONS,
    }
    _assert_r082_failure(certificate, "staged_inputs[0].path")
