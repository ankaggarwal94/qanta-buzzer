from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
DRIVER = REPO / "scripts" / "modal_stopdff_v5_assurance.py"


def _load_driver(monkeypatch: pytest.MonkeyPatch, *, times_out: bool = False):
    state = types.SimpleNamespace(
        spawns=[],
        from_ids=[],
        gets=[],
        cancellations=[],
    )

    class Call:
        def __init__(self, call_id: str, result: object) -> None:
            self.object_id = call_id
            self._result = result

        def get(self, *, timeout: float):
            state.gets.append((self.object_id, timeout))
            if times_out:
                raise TimeoutError("bounded timeout")
            return self._result

        def cancel(self, *, terminate_containers: bool) -> None:
            state.cancellations.append(
                (self.object_id, terminate_containers)
            )

    class Function:
        @classmethod
        def from_name(cls, deployment: str, function_name: str):
            assert function_name == "recovery_assurance"

            class Endpoint:
                def spawn(self, tag: str, phase: str):
                    call_id = f"fc-{phase}-{len(state.spawns) + 1}"
                    state.spawns.append((deployment, tag, phase, call_id))
                    return Call(
                        call_id,
                        {
                            "phase": phase,
                            "runtime": {
                                "container_hostname": "container",
                                "function_call_id": call_id,
                                "input_id": f"in-{phase}",
                            },
                        },
                    )

            return Endpoint()

    class FunctionCall:
        @classmethod
        def from_id(cls, call_id: str):
            state.from_ids.append(call_id)
            return Call(call_id, {"phase": "crash_rescheduled"})

    monkeypatch.setitem(
        sys.modules,
        "modal",
        types.SimpleNamespace(Function=Function, FunctionCall=FunctionCall),
    )
    name = f"_stopdff_assurance_driver_{id(monkeypatch)}_{times_out}"
    spec = importlib.util.spec_from_file_location(name, DRIVER)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module, state


def test_phase_call_is_bounded_and_receipt_binds_function_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver, state = _load_driver(monkeypatch)
    receipt = tmp_path / "classified.json"

    assert driver.main(
        [
            "classify",
            "--deployment",
            "assurance-app",
            "--tag",
            "deadbeef",
            "--timeout-seconds",
            "17",
            "--receipt",
            str(receipt),
        ]
    ) == 0

    value = json.loads(receipt.read_text(encoding="utf-8"))
    assert value["function_call_id"] == "fc-classify-1"
    assert value["result"]["runtime"]["function_call_id"] == value[
        "function_call_id"
    ]
    assert state.gets == [("fc-classify-1", 17.0)]
    assert state.cancellations == []


def test_existing_receipt_and_invalid_timeout_block_before_spawn(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver, state = _load_driver(monkeypatch)
    existing = tmp_path / "existing.json"
    existing.write_text('{"preserve":true}\n', encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        driver.main(
            [
                "classify",
                "--deployment",
                "assurance-app",
                "--tag",
                "deadbeef",
                "--receipt",
                str(existing),
            ]
        )
    with pytest.raises(ValueError, match="timeout-seconds"):
        driver.main(
            [
                "finish",
                "--deployment",
                "assurance-app",
                "--tag",
                "deadbeef",
                "--timeout-seconds",
                "0",
                "--receipt",
                str(tmp_path / "finished.json"),
            ]
        )
    assert state.spawns == []
    assert existing.read_text(encoding="utf-8") == '{"preserve":true}\n'


def test_phase_timeout_cancels_call_and_writes_no_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver, state = _load_driver(monkeypatch, times_out=True)
    receipt = tmp_path / "verified.json"

    with pytest.raises(TimeoutError, match="bounded timeout"):
        driver.main(
            [
                "verify",
                "--deployment",
                "assurance-app",
                "--tag",
                "deadbeef",
                "--timeout-seconds",
                "3",
                "--receipt",
                str(receipt),
            ]
        )
    assert state.cancellations == [("fc-verify-1", True)]
    assert not receipt.exists()


def test_write_once_uses_no_replace_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    driver, _ = _load_driver(monkeypatch)
    receipt = tmp_path / "receipt.json"
    driver._write_once(receipt, {"value": 1})
    with pytest.raises(FileExistsError, match="already exists"):
        driver._write_once(receipt, {"value": 2})
    assert json.loads(receipt.read_text(encoding="utf-8")) == {"value": 1}


@pytest.mark.parametrize(
    "payload",
    [
        b'{"schema_version":1,"deployment":"app","tag":"deadbeef",'
        b'"phase":"crash","function_call_id":"fc-first",'
        b'"function_call_id":"fc-last"}\n',
        b'{"deployment":"app","function_call_id":"fc-first",'
        b'"phase":"crash","schema_version":1,"tag":"deadbeef"}\n',
    ],
)
def test_recover_rejects_noncanonical_submission_before_call_lookup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes,
) -> None:
    driver, state = _load_driver(monkeypatch)
    submitted = tmp_path / "submitted.json"
    submitted.write_bytes(payload)

    with pytest.raises(ValueError, match="receipt"):
        driver.main(
            [
                "recover",
                "--call-receipt",
                str(submitted),
                "--receipt",
                str(tmp_path / "recovered.json"),
            ]
        )
    assert state.from_ids == []
    assert state.gets == []
    assert state.cancellations == []
