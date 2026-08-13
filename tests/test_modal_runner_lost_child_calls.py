"""Adversarial lost-response tests for Modal adapter child calls."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_stopdff_v5_control_plane import _load_modal_runner


@pytest.mark.parametrize(
    ("lost_boundary", "initial_subdirs"),
    [
        ("first", ["build_a"]),
        ("second", ["build_a", "build_b"]),
    ],
)
def test_lost_child_get_propagates_without_minting_or_reusing_receipt(
    lost_boundary: str,
    initial_subdirs: list[str],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A lost child result leaves destinations poisoned but no receipt issued."""
    runner = _load_modal_runner(monkeypatch)
    runner.MNT = str(tmp_path)
    source_id = "1" * 64
    raw_id = "2" * 64
    model_id = "3" * 64
    adapter_id = "4" * 64
    fit_hash = "5" * 64
    eval_hash = "6" * 64
    monkeypatch.setattr(
        runner,
        "_verified_executing_source",
        lambda _source_id: {"id": source_id},
    )

    from scripts.stopdff_v5 import writers

    minted: list[str] = []

    def unexpected_mint(*_args, **_kwargs):
        minted.append("receipt")
        raise AssertionError("lost child response must not reach receipt minting")

    monkeypatch.setattr(writers, "build_prerequisite_evidence", unexpected_mint)
    monkeypatch.setattr(
        writers,
        "build_evidenced_prerequisite_receipt",
        unexpected_mint,
    )

    spawn_events: list[str] = []
    get_events: list[str] = []

    def result_for(subdir: str) -> dict[str, object]:
        return {
            "adapter_id": adapter_id,
            "fit_rows_sha256": fit_hash,
            "eval_rows_sha256": eval_hash,
            "source_manifest_id": source_id,
            "raw_input_bundle_id": raw_id,
            "model_snapshot_id": model_id,
            "subdir": subdir,
            "cached": False,
        }

    class ProducerCall:
        def __init__(self, subdir: str, *, should_lose: bool):
            self.object_id = f"fc-{subdir}"
            self.subdir = subdir
            self.should_lose = should_lose

        def get(self) -> dict[str, object]:
            get_events.append(self.subdir)
            destination = tmp_path / "adapters" / self.subdir
            destination.mkdir(parents=True, exist_ok=False)
            (destination / "child-commit.marker").write_text(
                "committed before parent observed response\n",
                encoding="utf-8",
            )
            if self.should_lose:
                raise RuntimeError(f"lost {lost_boundary} child response")
            return result_for(self.subdir)

    class ExistingDestinationCall:
        def __init__(self, subdir: str):
            self.object_id = f"fc-retry-{subdir}"
            self.subdir = subdir

        def get(self) -> dict[str, object]:
            get_events.append(f"existing:{self.subdir}")
            raise FileExistsError(
                "fresh adapter build destination already exists; choose a new subdir"
            )

    def spawn(
        subdir: str,
        source: str,
        raw: str,
        model: str,
        allow_override: bool,
    ):
        assert (source, raw, model, allow_override) == (
            source_id,
            raw_id,
            model_id,
            False,
        )
        spawn_events.append(subdir)
        if (tmp_path / "adapters" / subdir).exists():
            return ExistingDestinationCall(subdir)
        should_lose = (
            lost_boundary == "first" and subdir == "build_a"
        ) or (
            lost_boundary == "second" and subdir == "build_b"
        )
        return ProducerCall(subdir, should_lose=should_lose)

    monkeypatch.setattr(runner.build_adapter, "spawn", spawn)

    with pytest.raises(RuntimeError, match=f"lost {lost_boundary} child response"):
        runner.adapter_determinism_receipt(
            "build_a",
            "build_b",
            source_id,
            raw_id,
            model_id,
        )

    assert spawn_events == initial_subdirs
    assert get_events == initial_subdirs
    assert minted == []
    assert not (tmp_path / "receipts" / "determinism").exists()
    for subdir in initial_subdirs:
        marker = tmp_path / "adapters" / subdir / "child-commit.marker"
        assert marker.read_text(encoding="utf-8").startswith("committed")

    # A same-path retry reaches the actual first call's get() and fails closed
    # on the poisoned create-once destination. It cannot reuse either old build
    # to manufacture a determinism receipt; the controller must choose its
    # attempt-scoped destination names instead.
    with pytest.raises(FileExistsError, match="destination already exists"):
        runner.adapter_determinism_receipt(
            "build_a",
            "build_b",
            source_id,
            raw_id,
            model_id,
        )

    assert spawn_events[-1] == "build_a"
    assert get_events[-1] == "existing:build_a"
    assert minted == []
    assert not (tmp_path / "receipts" / "determinism").exists()
