#!/usr/bin/env python3
"""Apply the bounded PR #30 adapter-execution provenance repair."""
from __future__ import annotations

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"expected one anchor in {path}, found {count}: {old[:100]!r}"
        )
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def append_once(path: str, sentinel: str, addition: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    if sentinel in text:
        raise RuntimeError(f"test sentinel already exists in {path}: {sentinel}")
    target.write_text(text.rstrip() + "\n" + addition.lstrip(), encoding="utf-8")


def patch_local_runner() -> None:
    path = "scripts/run_stopdff_v5_local.py"
    replace_once(
        path,
        '''def _variant_run_candidates(out: Path, variant: str) -> list[Path]:
''',
        '''def _materialize_adapter_stage(
    *,
    out: Path,
    state: dict,
    stage: str,
    build,
    source_id: str,
    raw_id: str,
    model_id: str,
) -> tuple[dict, str]:
    """Create or reuse an adapter only with a matching durable execution record."""
    stage_path = out / stage
    stage_present = stage_path.exists() or stage_path.is_symlink()
    checkpoint_present = stage in state["adapter_executions"]
    if stage_present != checkpoint_present:
        raise ValueError(
            f"local lifecycle {stage} stage/checkpoint presence mismatch"
        )

    if stage_present:
        manifest = _load_valid_adapter_stage(
            stage_path,
            source_id=source_id,
            raw_id=raw_id,
            model_id=model_id,
        )
        execution_id = _adapter_execution_id(
            state=state,
            stage=stage,
            adapter_id=manifest["id"],
        )
        return manifest, execution_id

    _publish_stage_directory(
        out=out,
        target_name=stage,
        build=build,
    )
    manifest = _load_valid_adapter_stage(
        stage_path,
        source_id=source_id,
        raw_id=raw_id,
        model_id=model_id,
    )
    execution_id = f"local-{uuid.uuid4().hex}"
    _checkpoint_adapter_execution(
        out=out,
        state=state,
        stage=stage,
        execution_id=execution_id,
        adapter_id=manifest["id"],
    )
    return manifest, execution_id


def _variant_run_candidates(out: Path, variant: str) -> list[Path]:
''',
    )

    replace_once(
        path,
        '''    print("== adapter bundle (CPU scoring) ==")
    adapter_dir = out / "adapter_bundle"
    if adapter_dir.exists() or adapter_dir.is_symlink():
        adapter_man = _load_valid_adapter_stage(
            adapter_dir,
            source_id=source_id,
            raw_id=raw_id,
            model_id=model_id,
        )
    else:
        def build_primary_adapter(staged: Path) -> dict:
            return adapter_build.build_adapter_bundle(
                mc_dataset_path=raw_dir / "mc_dataset.json",
                val_dataset_path=raw_dir / "val_dataset.json",
                test_dataset_path=raw_dir / "test_dataset.json",
                calibration_path=raw_dir / "calibration.json",
                model_snapshot_dir=out / "model" / "snapshot",
                out_dir=staged,
                source_manifest_id=source_id,
                raw_input_bundle_id=raw_id,
                model_snapshot_id=model_id,
                producer_hashes={
                    "adapter_build.py": sha256_file(
                        _REPO / "scripts/stopdff_v5/adapter_build.py"
                    )
                },
                allow_low_mc_retention=args.allow_low_mc_retention,
            )

        adapter_man = _publish_stage_directory(
            out=out,
            target_name="adapter_bundle",
            build=build_primary_adapter,
        )
    adapter_id = adapter_man["id"]
    _load_valid_adapter_stage(
        adapter_dir,
        source_id=source_id,
        raw_id=raw_id,
        model_id=model_id,
    )
    if "adapter_bundle" not in lifecycle["adapter_executions"]:
        _checkpoint_adapter_execution(
            out=out,
            state=lifecycle,
            stage="adapter_bundle",
            execution_id=f"local-{uuid.uuid4().hex}",
            adapter_id=adapter_id,
        )
    first_build_execution_id = _adapter_execution_id(
        state=lifecycle,
        stage="adapter_bundle",
        adapter_id=adapter_id,
    )
''',
        '''    print("== adapter bundle (CPU scoring) ==")
    adapter_dir = out / "adapter_bundle"

    def build_primary_adapter(staged: Path) -> dict:
        return adapter_build.build_adapter_bundle(
            mc_dataset_path=raw_dir / "mc_dataset.json",
            val_dataset_path=raw_dir / "val_dataset.json",
            test_dataset_path=raw_dir / "test_dataset.json",
            calibration_path=raw_dir / "calibration.json",
            model_snapshot_dir=out / "model" / "snapshot",
            out_dir=staged,
            source_manifest_id=source_id,
            raw_input_bundle_id=raw_id,
            model_snapshot_id=model_id,
            producer_hashes={
                "adapter_build.py": sha256_file(
                    _REPO / "scripts/stopdff_v5/adapter_build.py"
                )
            },
            allow_low_mc_retention=args.allow_low_mc_retention,
        )

    adapter_man, first_build_execution_id = _materialize_adapter_stage(
        out=out,
        state=lifecycle,
        stage="adapter_bundle",
        build=build_primary_adapter,
        source_id=source_id,
        raw_id=raw_id,
        model_id=model_id,
    )
    adapter_id = adapter_man["id"]
''',
    )

    replace_once(
        path,
        '''        second_adapter_dir = out / "adapter_bundle_determinism"
        if second_adapter_dir.exists() or second_adapter_dir.is_symlink():
            second_adapter = _load_valid_adapter_stage(
                second_adapter_dir,
                source_id=source_id,
                raw_id=raw_id,
                model_id=model_id,
            )
        else:
            def build_second_adapter(staged: Path) -> dict:
                return adapter_build.build_adapter_bundle(
                    mc_dataset_path=raw_dir / "mc_dataset.json",
                    val_dataset_path=raw_dir / "val_dataset.json",
                    test_dataset_path=raw_dir / "test_dataset.json",
                    calibration_path=raw_dir / "calibration.json",
                    model_snapshot_dir=out / "model" / "snapshot",
                    out_dir=staged,
                    source_manifest_id=source_id,
                    raw_input_bundle_id=raw_id,
                    model_snapshot_id=model_id,
                    producer_hashes={
                        "adapter_build.py": sha256_file(
                            _REPO / "scripts/stopdff_v5/adapter_build.py"
                        )
                    },
                    allow_low_mc_retention=args.allow_low_mc_retention,
                )

            second_adapter = _publish_stage_directory(
                out=out,
                target_name="adapter_bundle_determinism",
                build=build_second_adapter,
            )
        if "adapter_bundle_determinism" not in lifecycle["adapter_executions"]:
            _checkpoint_adapter_execution(
                out=out,
                state=lifecycle,
                stage="adapter_bundle_determinism",
                execution_id=f"local-{uuid.uuid4().hex}",
                adapter_id=second_adapter["id"],
            )
        second_build_execution_id = _adapter_execution_id(
            state=lifecycle,
            stage="adapter_bundle_determinism",
            adapter_id=second_adapter["id"],
        )
''',
        '''        second_adapter_dir = out / "adapter_bundle_determinism"

        def build_second_adapter(staged: Path) -> dict:
            return adapter_build.build_adapter_bundle(
                mc_dataset_path=raw_dir / "mc_dataset.json",
                val_dataset_path=raw_dir / "val_dataset.json",
                test_dataset_path=raw_dir / "test_dataset.json",
                calibration_path=raw_dir / "calibration.json",
                model_snapshot_dir=out / "model" / "snapshot",
                out_dir=staged,
                source_manifest_id=source_id,
                raw_input_bundle_id=raw_id,
                model_snapshot_id=model_id,
                producer_hashes={
                    "adapter_build.py": sha256_file(
                        _REPO / "scripts/stopdff_v5/adapter_build.py"
                    )
                },
                allow_low_mc_retention=args.allow_low_mc_retention,
            )

        second_adapter, second_build_execution_id = _materialize_adapter_stage(
            out=out,
            state=lifecycle,
            stage="adapter_bundle_determinism",
            build=build_second_adapter,
            source_id=source_id,
            raw_id=raw_id,
            model_id=model_id,
        )
''',
    )


def patch_tests() -> None:
    append_once(
        "tests/test_pr30_control_repairs.py",
        "test_local_adapter_stage_reuse_requires_paired_checkpoint",
        r'''


@pytest.mark.parametrize(
    ("stage_present", "checkpoint_present"),
    [(True, False), (False, True)],
)
def test_local_adapter_stage_reuse_requires_paired_checkpoint(
    tmp_path,
    monkeypatch,
    stage_present,
    checkpoint_present,
):
    stage = "adapter_bundle_determinism"
    adapter_id = "a" * 64
    if stage_present:
        (tmp_path / stage).mkdir()
    state = {"adapter_executions": {}}
    if checkpoint_present:
        state["adapter_executions"][stage] = {
            "execution_id": "local-recorded",
            "adapter_id": adapter_id,
        }

    monkeypatch.setattr(
        local_runner,
        "_load_valid_adapter_stage",
        lambda *_args, **_kwargs: pytest.fail(
            "mismatched stage/checkpoint state was loaded"
        ),
    )
    monkeypatch.setattr(
        local_runner,
        "_publish_stage_directory",
        lambda **_kwargs: pytest.fail(
            "mismatched stage/checkpoint state was rebuilt"
        ),
    )
    monkeypatch.setattr(
        local_runner,
        "_checkpoint_adapter_execution",
        lambda **_kwargs: pytest.fail(
            "mismatched stage/checkpoint state minted an execution"
        ),
    )

    with pytest.raises(ValueError, match="stage/checkpoint presence mismatch"):
        local_runner._materialize_adapter_stage(
            out=tmp_path,
            state=state,
            stage=stage,
            build=lambda _path: pytest.fail("unexpected adapter build"),
            source_id="1" * 64,
            raw_id="2" * 64,
            model_id="3" * 64,
        )


def test_local_adapter_stage_reuse_preserves_checkpointed_execution(
    tmp_path,
    monkeypatch,
):
    stage = "adapter_bundle_determinism"
    stage_path = tmp_path / stage
    stage_path.mkdir()
    adapter_id = "a" * 64
    execution_id = "local-original-execution"
    state = {
        "adapter_executions": {
            stage: {
                "execution_id": execution_id,
                "adapter_id": adapter_id,
            }
        }
    }
    monkeypatch.setattr(
        local_runner,
        "_load_valid_adapter_stage",
        lambda path, **_kwargs: (
            {"id": adapter_id}
            if path == stage_path
            else pytest.fail("unexpected adapter path")
        ),
    )
    monkeypatch.setattr(
        local_runner,
        "_publish_stage_directory",
        lambda **_kwargs: pytest.fail("checkpointed adapter was rebuilt"),
    )
    monkeypatch.setattr(
        local_runner,
        "_checkpoint_adapter_execution",
        lambda **_kwargs: pytest.fail("checkpointed execution was replaced"),
    )

    manifest, observed_execution_id = local_runner._materialize_adapter_stage(
        out=tmp_path,
        state=state,
        stage=stage,
        build=lambda _path: pytest.fail("unexpected adapter build"),
        source_id="1" * 64,
        raw_id="2" * 64,
        model_id="3" * 64,
    )

    assert manifest == {"id": adapter_id}
    assert observed_execution_id == execution_id


def test_local_adapter_stage_fresh_build_checkpoints_once(
    tmp_path,
    monkeypatch,
):
    stage = "adapter_bundle"
    stage_path = tmp_path / stage
    adapter_id = "a" * 64
    state = {"adapter_executions": {}}
    calls = []

    def publish(*, out, target_name, build):
        assert out == tmp_path
        assert target_name == stage
        stage_path.mkdir()
        calls.append("publish")
        return build(stage_path)

    def checkpoint(**kwargs):
        calls.append("checkpoint")
        state["adapter_executions"][stage] = {
            "execution_id": kwargs["execution_id"],
            "adapter_id": kwargs["adapter_id"],
        }

    monkeypatch.setattr(local_runner, "_publish_stage_directory", publish)
    monkeypatch.setattr(
        local_runner,
        "_load_valid_adapter_stage",
        lambda path, **_kwargs: (
            {"id": adapter_id}
            if path == stage_path
            else pytest.fail("unexpected adapter path")
        ),
    )
    monkeypatch.setattr(
        local_runner,
        "_checkpoint_adapter_execution",
        checkpoint,
    )
    monkeypatch.setattr(
        local_runner.uuid,
        "uuid4",
        lambda: types.SimpleNamespace(hex="c" * 32),
    )

    manifest, execution_id = local_runner._materialize_adapter_stage(
        out=tmp_path,
        state=state,
        stage=stage,
        build=lambda _path: {"id": adapter_id},
        source_id="1" * 64,
        raw_id="2" * 64,
        model_id="3" * 64,
    )

    assert calls == ["publish", "checkpoint"]
    assert manifest == {"id": adapter_id}
    assert execution_id == f"local-{'c' * 32}"
    assert state["adapter_executions"][stage] == {
        "execution_id": execution_id,
        "adapter_id": adapter_id,
    }
''',
    )


def main() -> None:
    patch_local_runner()
    patch_tests()


if __name__ == "__main__":
    main()
