"""Focused regression oracles for the PR-30 successor repair set."""
from __future__ import annotations

import gzip
import json
import math
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.stopdff_v5 import adapter_build, checker, selftest, sweep
from scripts.stopdff_v5.identity import (
    build_manifest,
    compute_id,
    sha256_bytes,
    sha256_file,
)
from scripts.stopdff_v5.manifests import (
    ENVIRONMENT_PACKAGES,
    RAW_INPUT_ROLES,
    environment_contract_identity,
    raw_input_identity,
)
from tests.harness_control_plane import _load_modal_runner


EXPECTED_ENVIRONMENT_PACKAGES = (
    "numpy",
    "scipy",
    "scikit-learn",
    "pandas",
    "matplotlib",
    "sentence-transformers",
    "torch",
    "transformers",
    "huggingface_hub",
)


def test_run_spec_envelope_is_closed_in_both_public_entrypoints(
    tmp_path: Path,
) -> None:
    built = selftest.build_valid_package(tmp_path)
    spec_path = built["run_root"] / "run_spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    spec["unbound_attestation"] = {"trusted": True}
    spec_path.write_text(
        json.dumps(spec, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    standalone = checker.validate_spec(
        spec_path,
        require_final_profile=False,
    )
    packaged = checker.validate_run(
        built["run_root"],
        backend="modal",
        adapter_bundle=built["adapter_bundle"],
    )

    assert not standalone.passed
    assert not packaged.passed
    assert any("canonical envelope" in error for error in standalone.errors)
    assert any("canonical envelope" in error for error in packaged.errors)


@pytest.mark.parametrize("backend", [None, 7, "custom"])
def test_validate_run_rejects_unknown_backend_before_any_evidence_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    backend: object,
) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("invalid backend reached the evidence reader")

    monkeypatch.setattr(checker, "_validate_run_impl", forbidden)
    result = checker.validate_run(
        tmp_path / "unread-run",
        backend=backend,  # type: ignore[arg-type]
        adapter_bundle=tmp_path / "unread-adapter",
    )
    assert not result.passed
    assert result.errors == ["backend must be exactly 'local' or 'modal'"]


def test_validate_run_normalizes_oserror_but_not_process_control(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(*_args, **_kwargs):
        raise PermissionError("evidence denied")

    monkeypatch.setattr(checker, "_validate_run_impl", unavailable)
    result = checker.validate_run(
        tmp_path / "run",
        backend="modal",
        adapter_bundle=tmp_path / "adapter",
    )
    assert not result.passed
    assert any("PermissionError" in error for error in result.errors)

    def cancelled(*_args, **_kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(checker, "_validate_run_impl", cancelled)
    with pytest.raises(KeyboardInterrupt):
        checker.validate_run(
            tmp_path / "run",
            backend="modal",
            adapter_bundle=tmp_path / "adapter",
        )


def test_public_validators_normalize_corrupted_deflate_evidence(
    tmp_path: Path,
) -> None:
    built = selftest.build_valid_package(tmp_path)
    corrupted_gzip = bytes.fromhex(
        "1f8b08000000000000ff07000000000000000000"
    )
    (built["adapter_bundle"] / "fit_rows.jsonl.gz").write_bytes(
        corrupted_gzip
    )

    adapter_result = checker.validate_adapter(built["adapter_bundle"])
    run_result = checker.validate_run(
        built["run_root"],
        backend="modal",
        adapter_bundle=built["adapter_bundle"],
    )
    run_spec = json.loads(
        (built["run_root"] / "run_spec.json").read_text(encoding="utf-8")
    )
    bootstrap_plan = json.loads(
        (built["run_root"] / "bootstrap_plan.json").read_text(
            encoding="utf-8"
        )
    )

    assert not adapter_result.passed
    assert not run_result.passed
    assert any("cannot be decoded" in error for error in adapter_result.errors)
    assert run_result.errors
    with pytest.raises(ValueError, match="adapter rows cannot be decoded"):
        checker.resolve_run_binding(
            run_spec_manifest=run_spec,
            adapter_bundle=built["adapter_bundle"],
            bootstrap_plan_manifest=bootstrap_plan,
        )


def test_resume_and_acceptance_share_exact_attempt_bytes(tmp_path: Path) -> None:
    built = selftest.build_valid_package(tmp_path)
    attempts_path = built["run_root"] / "attempts.jsonl"
    canonical = attempts_path.read_bytes()
    record = json.loads(canonical.decode("utf-8"))
    surplus = {**record, "unbound": True}
    reversed_record = {
        key: record[key]
        for key in reversed(tuple(record))
    }
    mutations = {
        "missing-newline": canonical[:-1],
        "blank-record": canonical + b"\n",
        "surplus-field": (
            json.dumps(surplus, sort_keys=True) + "\n"
        ).encode("utf-8"),
        "reordered-fields": (
            json.dumps(reversed_record, sort_keys=False) + "\n"
        ).encode("utf-8"),
    }

    for label, payload in mutations.items():
        attempts_path.write_bytes(payload)
        with pytest.raises(ValueError, match="attempt history"):
            sweep._load_attempt_history(attempts_path)
        result = checker.validate_run(
            built["run_root"],
            backend="modal",
            adapter_bundle=built["adapter_bundle"],
        )
        assert not result.passed, label
        assert any("attempt" in error for error in result.errors), label

    attempts_path.write_bytes(canonical)
    _, records = sweep._load_attempt_history(attempts_path)
    assert records == [record]


def test_torch_changes_environment_identity() -> None:
    assert ENVIRONMENT_PACKAGES == EXPECTED_ENVIRONMENT_PACKAGES
    versions = {name: "1.0" for name in EXPECTED_ENVIRONMENT_PACKAGES}
    first = compute_id(
        environment_contract_identity(
            python_version="3.11.9",
            package_versions=versions,
        )
    )
    versions["torch"] = "2.0"
    second = compute_id(
        environment_contract_identity(
            python_version="3.11.9",
            package_versions=versions,
        )
    )
    assert first != second


def test_category_is_required_and_bound_before_scoring(tmp_path: Path) -> None:
    split_path = tmp_path / "val.json"
    split_path.write_text(
        json.dumps(
            [
                {
                    "qid": "v",
                    "question": "Validation?",
                    "answer_primary": "Validation",
                    "category": "   ",
                }
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="invalid category"):
        adapter_build._dataset_index(split_path, split="val")

    val = {
        "v": {
            "text": "validation?",
            "answer": "validation",
            "category": "History",
        }
    }
    test = {
        "t": {
            "text": "test?",
            "answer": "test",
            "category": "Science",
        }
    }
    questions = [
        {
            "qid": "v",
            "question": "Validation?",
            "answer_primary": "Validation",
            "category": "Literature",
        },
        {
            "qid": "t",
            "question": "Test?",
            "answer_primary": "Test",
            "category": "Science",
        },
    ]
    with pytest.raises(ValueError, match="category does not match"):
        adapter_build._validate_split_bindings(val, test, questions)


def test_checker_rejects_whitespace_category_after_rehash(
    tmp_path: Path,
) -> None:
    built = selftest.build_valid_package(tmp_path)
    bundle = built["adapter_bundle"]
    rows = checker.load_jsonl_gz(bundle / "fit_rows.jsonl.gz")
    first_item = rows[0]["item_id"]
    for row in rows:
        if row["item_id"] == first_item:
            row["category"] = "   "
    payload = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    ).encode("utf-8")
    rows_path = bundle / "fit_rows.jsonl.gz"
    rows_path.write_bytes(gzip.compress(payload, mtime=0))
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["identity"]["fit_rows_sha256"] = sha256_file(rows_path)
    manifest["id"] = compute_id(manifest["identity"])
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    result = checker.validate_adapter(bundle)
    assert not result.passed
    assert any("invalid category" in error for error in result.errors)


def _scoring_question(
    qid: str,
    first: str,
    second: str,
    answer: str,
    distractor: str,
) -> dict:
    return {
        "qid": qid,
        "question": f"{first} {second}",
        "answer_primary": answer,
        "cumulative_prefixes": [first, f"{first} {second}"],
        "options": [answer, distractor],
        "gold_index": 0,
        "category": "Test",
    }


def test_adapter_scores_all_retained_texts_in_one_fixed_batch() -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.calls = 0

        def encode(
            self,
            values,
            *,
            batch_size,
            convert_to_numpy,
            show_progress_bar,
        ):
            self.calls += 1
            assert batch_size == adapter_build._ENCODE_BATCH_SIZE
            assert convert_to_numpy is True
            assert show_progress_bar is False
            return np.asarray(
                [
                    [
                        float(len(value) + 1),
                        float(sum(map(ord, value)) % 17 + 1),
                        float(
                            sum(
                                index * ord(ch)
                                for index, ch in enumerate(value)
                            )
                            % 19
                            + 1
                        ),
                    ]
                    for value in values
                ],
                dtype=np.float64,
            )

    questions = [
        (
            _scoring_question(
                "q1", "alpha", "beta", "Mercury", "Venus"
            ),
            "val",
        ),
        (
            _scoring_question(
                "q2", "gamma", "delta", "Copper", "Silver"
            ),
            "test",
        ),
    ]
    batched_model = FakeModel()
    actual = adapter_build._score_questions_rows(questions, batched_model)
    legacy_model = FakeModel()
    expected = [
        row
        for question, split in questions
        for row in adapter_build._score_question_rows(
            question,
            legacy_model,
            split,
        )
    ]

    assert batched_model.calls == 1
    assert legacy_model.calls == len(questions)
    assert actual == expected


def test_ece_is_derived_from_serialized_platt_parameters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sklearn import linear_model

    class BoundaryCrossingLogisticRegression:
        full_coefficient = 1.00000049
        full_intercept = -2.19722451

        def __init__(self, **_kwargs) -> None:
            self.coef_ = np.asarray([[self.full_coefficient]])
            self.intercept_ = np.asarray([self.full_intercept])

        def fit(self, _scores, _labels):
            return self

        def predict_proba(self, scores):
            logits = (
                np.asarray(scores, dtype=np.float64)[:, 0]
                * self.full_coefficient
                + self.full_intercept
            )
            probabilities = 1.0 / (
                1.0 + np.exp(-np.clip(logits, -500.0, 500.0))
            )
            return np.column_stack((1.0 - probabilities, probabilities))

    monkeypatch.setattr(
        linear_model,
        "LogisticRegression",
        BoundaryCrossingLogisticRegression,
    )
    fractions = {"early": 0.1, "mid": 0.5, "late": 0.9}
    fit_rows: list[dict] = []
    eval_rows: list[dict] = []
    for fraction in fractions.values():
        for index in range(12):
            fit_rows.append(
                {
                    "format": "MC",
                    "split": "val",
                    "prefix_fraction": fraction,
                    "raw_similarity": (index - 5.5) / 6.0,
                    "correct": int(index >= 6),
                }
            )
        for score, label in (
            (0.0, 1),
            (0.2, 0),
            (0.25, 0),
            (0.3, 0),
            (0.35, 0),
        ):
            eval_rows.append(
                {
                    "format": "MC",
                    "split": "test",
                    "prefix_fraction": fraction,
                    "raw_similarity": score,
                    "correct": label,
                }
            )

    calibration = adapter_build.derive_bound_calibration(
        fit_rows=fit_rows,
        eval_rows=eval_rows,
        model_snapshot_id="a" * 64,
        fit_rows_sha256="b" * 64,
    )
    eval_by_fraction = {
        fraction: [
            row
            for row in eval_rows
            if row["prefix_fraction"] == fraction
        ]
        for fraction in fractions.values()
    }
    def independent_ece(
        coefficient: float,
        intercept: float,
        rows: list[dict],
    ) -> float:
        probabilities = np.asarray(
            [
                1.0
                / (
                    1.0
                    + math.exp(
                        -max(
                            -500.0,
                            min(
                                500.0,
                                coefficient * row["raw_similarity"]
                                + intercept,
                            ),
                        )
                    )
                )
                for row in rows
            ]
        )
        labels = np.asarray(
            [row["correct"] for row in rows],
            dtype=np.float64,
        )
        value = 0.0
        for index in range(10):
            lower = index / 10.0
            upper = (index + 1) / 10.0
            selected = (probabilities >= lower) & (
                probabilities <= upper
                if index == 9
                else probabilities < upper
            )
            if selected.any():
                value += float(selected.mean()) * abs(
                    float(labels[selected].mean())
                    - float(probabilities[selected].mean())
                )
        return value

    for phase, fraction in fractions.items():
        block = calibration["per_bucket"][phase]
        serialized_ece = independent_ece(
            block["platt_coef"],
            block["platt_intercept"],
            eval_by_fraction[fraction],
        )
        predecessor_ece = independent_ece(
            BoundaryCrossingLogisticRegression.full_coefficient,
            BoundaryCrossingLogisticRegression.full_intercept,
            eval_by_fraction[fraction],
        )
        assert block["ece"] == round(serialized_ece, 6)
        assert round(serialized_ece, 6) != round(predecessor_ece, 6)


class _FakeVolume:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.uploads = 0
        self.upload_sources: list[str] = []

    def listdir(self, remote_dir: str, *, recursive: bool):
        assert recursive is True
        names = sorted(
            name
            for name in self.files
            if name.startswith(remote_dir.rstrip("/") + "/")
        )
        if not names:
            raise FileNotFoundError(remote_dir)
        return [SimpleNamespace(path=name) for name in names]

    def batch_upload(self, *, force: bool):
        assert force is False
        volume = self

        class Batch:
            def __enter__(self):
                volume.uploads += 1
                return self

            def __exit__(self, *_args):
                return False

            def put_directory(self, local: str, remote: str) -> None:
                volume.upload_sources.append(local)
                for path in sorted(Path(local).rglob("*")):
                    if path.is_file():
                        relative = path.relative_to(local).as_posix()
                        volume.files[f"{remote}/{relative}"] = path.read_bytes()

        return Batch()

    def read_file(self, path: str):
        return iter((self.files[path],))


def _write_nested_raw_input_bundle(bundle: Path) -> dict:
    raw = bundle / "raw"
    raw.mkdir(parents=True)
    files = []
    for role in RAW_INPUT_ROLES:
        path = raw / role
        data = f"fixture for {role}\n".encode("utf-8")
        path.write_bytes(data)
        files.append(
            {
                "role": role,
                "size": len(data),
                "sha256": sha256_bytes(data),
            }
        )
    manifest = build_manifest(
        raw_input_identity(
            files=files,
            semantic_checks={
                "all_semantic_checks_pass": True,
                "question_trajectory_binding_id": "c" * 64,
            },
        )
    )
    (bundle / "raw_input_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (bundle / "raw_input_stage_record.json").write_text(
        json.dumps({"source_paths": {"private": "/not/for/upload"}}) + "\n",
        encoding="utf-8",
    )
    return manifest


def test_input_staging_is_create_once_cached_and_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_modal_runner(monkeypatch)
    bundle = tmp_path / "source-bundle"
    shutil.copytree(runner._IMAGE_SOURCE_DIR.parent, bundle)
    manifest = json.loads(
        (bundle / "source_manifest.json").read_text(encoding="utf-8")
    )
    volume = _FakeVolume()

    def verifier(remote_dir: str, kind: str) -> dict:
        assert kind == "source"
        remote_manifest = json.loads(
            volume.files[f"{remote_dir}/source_manifest.json"]
        )
        return {
            "ok": True,
            "id": remote_manifest["id"],
            "mismatches": [],
            "n_files": len(remote_manifest["identity"]["files"]),
        }

    first = runner._stage_one_input_bundle(
        bundle,
        "source",
        volume=volume,
        verifier=verifier,
    )
    second = runner._stage_one_input_bundle(
        bundle,
        "source",
        volume=volume,
        verifier=verifier,
    )
    assert first["status"] == "created"
    assert second == {**first, "status": "cached"}
    assert first["id"] == manifest["id"]
    assert volume.uploads == 1

    partial = _FakeVolume()
    remote_dir = f"inputs/source_{manifest['id']}"
    partial.files[f"{remote_dir}/partial"] = b"not a bundle"
    with pytest.raises(ValueError, match="failed remote verification"):
        runner._stage_one_input_bundle(
            bundle,
            "source",
            volume=partial,
            verifier=lambda *_args: {
                "ok": False,
                "id": manifest["id"],
                "mismatches": ["partial"],
                "n_files": 0,
            },
        )
    assert partial.uploads == 0


def test_raw_input_staging_flattens_private_copy_and_reuses_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_modal_runner(monkeypatch)
    bundle = tmp_path / "raw_inputs"
    manifest = _write_nested_raw_input_bundle(bundle)
    remote_dir = f"inputs/raw_{manifest['id']}"
    expected_paths = {
        f"{remote_dir}/raw_input_manifest.json",
        *(f"{remote_dir}/{role}" for role in RAW_INPUT_ROLES),
    }
    volume = _FakeVolume()

    def verifier(actual_remote_dir: str, kind: str) -> dict:
        assert actual_remote_dir == remote_dir
        assert kind == "raw"
        assert set(volume.files) == expected_paths
        remote_manifest = json.loads(
            volume.files[f"{remote_dir}/raw_input_manifest.json"]
        )
        for entry in remote_manifest["identity"]["files"]:
            data = volume.files[f"{remote_dir}/{entry['role']}"]
            assert len(data) == entry["size"]
            assert sha256_bytes(data) == entry["sha256"]
        return {
            "ok": True,
            "id": remote_manifest["id"],
            "mismatches": [],
            "n_files": len(remote_manifest["identity"]["files"]),
        }

    first = runner._stage_one_input_bundle(
        bundle,
        "raw",
        volume=volume,
        verifier=verifier,
    )
    assert first["status"] == "created"
    assert volume.uploads == 1
    assert len(volume.upload_sources) == 1
    assert not Path(volume.upload_sources[0]).exists()
    assert not any("/raw/" in path for path in volume.files)
    assert not any(path.endswith("raw_input_stage_record.json") for path in volume.files)

    second = runner._stage_one_input_bundle(
        bundle,
        "raw",
        volume=volume,
        verifier=verifier,
    )
    assert second == {**first, "status": "cached"}
    assert volume.uploads == 1

    partial = _FakeVolume()
    partial.files[f"{remote_dir}/raw_input_manifest.json"] = (
        bundle / "raw_input_manifest.json"
    ).read_bytes()
    with pytest.raises(ValueError, match="failed remote verification"):
        runner._stage_one_input_bundle(
            bundle,
            "raw",
            volume=partial,
            verifier=lambda *_args: {
                "ok": False,
                "id": manifest["id"],
                "mismatches": ["partial"],
                "n_files": 0,
            },
        )
    assert partial.uploads == 0


def test_raw_input_staging_rejects_local_tampering_before_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_modal_runner(monkeypatch)
    bundle = tmp_path / "raw_inputs"
    _write_nested_raw_input_bundle(bundle)
    role_path = bundle / "raw" / RAW_INPUT_ROLES[0]
    original = role_path.read_bytes()

    role_path.write_bytes(b"tampered\n")
    tampered = _FakeVolume()
    with pytest.raises(ValueError, match="file mismatch"):
        runner._stage_one_input_bundle(bundle, "raw", volume=tampered)
    assert tampered.uploads == 0

    role_path.write_bytes(original)
    (bundle / "raw" / "unlisted.json").write_text("{}\n", encoding="utf-8")
    unlisted = _FakeVolume()
    with pytest.raises(ValueError, match="inventory mismatch"):
        runner._stage_one_input_bundle(bundle, "raw", volume=unlisted)
    assert unlisted.uploads == 0


def test_create_once_control_bytes_never_replace_existing_destination(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_modal_runner(monkeypatch)
    destination = tmp_path / "receipt.json"
    runner._atomic_create_control_bytes(destination, b"first\n")
    with pytest.raises(FileExistsError, match="already exists"):
        runner._atomic_create_control_bytes(destination, b"second\n")
    assert destination.read_bytes() == b"first\n"


def test_assurance_phase_guards_prevent_duplicate_or_out_of_order_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_modal_runner(monkeypatch)
    tag = "45b7f81f-c79c-42fa-89f2-07707cc0d41c"
    runner.MNT = str(tmp_path / "volume")
    arm_path = Path(runner._p("pilots", tag, "crash_arm.json"))
    arm_path.parent.mkdir(parents=True)
    arm_path.write_text("{}\n", encoding="utf-8")
    expected = runner._assurance_expected_evidence(tag)
    arm = {
        "tag": tag,
        "source_manifest_id": runner.IMAGE_SOURCE_MANIFEST_ID,
        "runtime": {
            "container_hostname": "crashed-container",
            "function_call_id": "fc-crash",
            "input_id": "in-crash",
        },
        "exit_code": 91,
        "armed_after_attempt_start_commit": True,
    }

    def observation(state: str) -> dict:
        attempts = [expected["first_attempt"]]
        results: dict[str, dict] = {}
        if state in {"classified", "finished"}:
            results["1.json"] = expected["interrupted"]
        if state == "finished":
            attempts.append(expected["second_attempt"])
            results["2.json"] = expected["completed"]
        return {
            "tag": tag,
            "attempts": attempts,
            "results": results,
            "result_sha256": {},
            "crash_arm": arm,
            "attempts_sha256": "0" * 64,
            "crash_arm_sha256": "0" * 64,
            "run_spec": expected["run_spec"],
            "run_spec_sha256": "0" * 64,
            "bootstrap_plan": expected["bootstrap_plan"],
            "bootstrap_plan_sha256": "0" * 64,
        }

    monkeypatch.setattr(
        runner,
        "_modal_runtime_identity",
        lambda: {
            "container_hostname": "phase-container",
            "function_call_id": "fc-phase",
            "input_id": "in-phase",
        },
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("phase guard allowed sweep mutation")

    monkeypatch.setattr(sweep, "run_sweep", forbidden)
    classified = observation("classified")
    monkeypatch.setattr(runner, "_assurance_observation", lambda _tag: classified)
    result = runner.recovery_assurance(tag, "classify")
    assert result["phase"] == "classified"
    assert result["observation"] == classified

    initial = observation("initial")
    monkeypatch.setattr(runner, "_assurance_observation", lambda _tag: initial)
    with pytest.raises(ValueError, match="finish phase requires"):
        runner.recovery_assurance(tag, "finish")

    monkeypatch.setattr(runner, "_assurance_observation", lambda _tag: classified)
    with pytest.raises(ValueError, match="verify phase requires"):
        runner.recovery_assurance(tag, "verify")


# --- fresh run-root publication is create-once (PR #30) ----------------------
#
# ``_publish_fresh_initialization`` must claim ``runs/<id>`` before filling it,
# so a racing peer's empty run root fails closed instead of being silently
# replaced by a bare rename. Identity/attempt materialization is stubbed here
# to isolate the publish mechanics the fix changed.


def test_publish_fresh_initialization_publishes_into_absent_run_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Happy path across the create-once refactor: the staged run root is
    installed under its live name and the private staging holder is gone."""
    monkeypatch.setattr(
        sweep, "_run_identity_files", lambda ctx: (("run_spec.json", {"id": "x"}),)
    )
    monkeypatch.setattr(
        sweep,
        "_append_attempt",
        lambda path, attempt: Path(path).write_text("{}\n", encoding="utf-8"),
    )
    output_dir = tmp_path / "runs" / ("r" * 8)
    ctx = SimpleNamespace(output_dir=output_dir)

    sweep._publish_fresh_initialization(ctx, started_attempt={"attempt": 1})

    assert output_dir.is_dir()
    assert (output_dir / "run_spec.json").is_file()
    assert (output_dir / "attempts.jsonl").is_file()
    assert not list((tmp_path / "runs").glob(".stopdff_run_initializing_*"))


def test_publish_fresh_initialization_fails_closed_on_racing_empty_run_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A concurrent peer that claims an empty ``runs/<id>`` directory in the
    publish window must not be silently clobbered.

    Pre-fix, ``os.rename`` onto that empty peer directory replaces it (the
    create-once run-root violation). Post-fix the run root is claimed with
    ``os.mkdir`` first, so the concurrent claim collides and publication fails
    closed. Asserting the raise makes this red against the pre-fix os.rename
    code (mutation-discrimination).
    """
    monkeypatch.setattr(
        sweep, "_run_identity_files", lambda ctx: (("run_spec.json", {"id": "x"}),)
    )
    monkeypatch.setattr(
        sweep,
        "_append_attempt",
        lambda path, attempt: Path(path).write_text("{}\n", encoding="utf-8"),
    )
    output_dir = tmp_path / "runs" / ("r" * 8)
    ctx = SimpleNamespace(output_dir=output_dir)

    real_rename = os.rename

    def racing_rename(src, dst):
        # A concurrent sweep peer materializes the empty run root in the
        # check-to-publish window, then the real move proceeds.
        os.mkdir(dst)
        return real_rename(src, dst)

    monkeypatch.setattr(os, "rename", racing_rename)

    with pytest.raises(FileExistsError):
        sweep._publish_fresh_initialization(ctx, started_attempt={"attempt": 1})


@pytest.mark.skipif(
    os.name == "nt", reason="Windows does not expose repository POSIX execute bits"
)
def test_source_executable_mode_is_bound_and_rechecked_at_runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_modal_runner(monkeypatch)
    bundle = tmp_path / "bundle"
    shutil.copytree(runner._IMAGE_SOURCE_DIR.parent, bundle)
    manifest = json.loads(
        (bundle / "source_manifest.json").read_text(encoding="utf-8")
    )
    entry = manifest["identity"]["files"][0]
    bound_path = bundle / "source" / entry["path"]
    bound_path.chmod(bound_path.stat().st_mode | 0o111)
    with pytest.raises(ValueError, match="file mismatch"):
        runner._validated_local_input_bundle(bundle, "source")

    shutil.rmtree(bundle)
    shutil.copytree(runner._IMAGE_SOURCE_DIR.parent, bundle)
    source_id = manifest["id"]
    staged = tmp_path / "volume" / "inputs" / f"source_{source_id}"
    shutil.copytree(bundle, staged)
    runtime = tmp_path / "runtime"
    shutil.copytree(bundle / "source", runtime)
    runtime_path = runtime / entry["path"]
    runtime_path.chmod(runtime_path.stat().st_mode | 0o111)
    runner.MNT = str(tmp_path / "volume")
    runner.REMOTE_SRC = str(runtime)
    runner.IMAGE_SOURCE_MANIFEST_ID = source_id
    with pytest.raises(ValueError, match="executing source does not match"):
        runner._verified_executing_source(source_id)


def test_materialized_image_source_owner_controls_lifetime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_modal_runner(monkeypatch)
    bundle = tmp_path / "bundle"
    shutil.copytree(runner._IMAGE_SOURCE_DIR.parent, bundle)
    owner, source, _manifest = runner._materialize_image_source(bundle)
    root = Path(owner.name)
    assert root.is_dir()
    assert source.is_dir()
    owner.cleanup()
    assert not root.exists()

    created: list[Path] = []
    real_temporary_directory = runner.tempfile.TemporaryDirectory

    def tracked_owner(*args, **kwargs):
        result = real_temporary_directory(*args, **kwargs)
        created.append(Path(result.name))
        return result

    monkeypatch.setattr(runner.tempfile, "TemporaryDirectory", tracked_owner)
    monkeypatch.setattr(
        runner.shutil,
        "copytree",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("copy failed")),
    )
    with pytest.raises(OSError, match="copy failed"):
        runner._materialize_image_source(bundle)
    assert len(created) == 1
    assert not created[0].exists()
