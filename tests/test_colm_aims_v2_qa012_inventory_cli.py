from __future__ import annotations

import copy
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from reproducibility.colm_aims_2026 import (
    phase4_driver_d7b,
    qa012,
    qa012_inventory,
    schema,
)


def _scope_roots(base: Path) -> dict[str, Path]:
    roots: dict[str, Path] = {}
    for index, name in enumerate(qa012.REQUIRED_SCOPE_PRONGS):
        root = base / f"root-{index}"
        root.mkdir(parents=True)
        (root / f"input-{index}.json").write_text(
            '{"format":"MC"}\n', encoding="utf-8"
        )
        roots[name] = root
    return roots


def _argv(roots: dict[str, Path], output: Path) -> list[str]:
    args: list[str] = []
    for name in reversed(qa012.REQUIRED_SCOPE_PRONGS):
        args.extend(("--prong", f"{name}={roots[name]}"))
    args.extend(("--out", str(output)))
    return args


def test_cli_writes_valid_create_once_non_authorizing_diagnostic(
    tmp_path, capsys
):
    roots = _scope_roots(tmp_path / "scope")
    output = tmp_path / "qa012-diagnostic.json"

    assert qa012_inventory.main(_argv(roots, output)) == qa012_inventory.EXIT_OK

    raw = output.read_bytes()
    manifest = schema.parse_json_bytes_strict(raw)
    qa012.validate_inventory_manifest(manifest)
    assert raw == schema.encode_json(manifest)
    assert manifest["result"] == "zero_hit"
    stdout = capsys.readouterr().out
    assert qa012_inventory.NON_AUTHORIZING_LABEL in stdout
    assert "does not satisfy CAMERA_READY_CLOSURE" in stdout

    assert (
        qa012_inventory.main(_argv(roots, output))
        == qa012_inventory.EXIT_USAGE_ERROR
    )
    assert output.read_bytes() == raw


def test_cli_diagnostic_is_rejected_by_authority_only_ingress(tmp_path):
    roots = _scope_roots(tmp_path / "scope")
    output = tmp_path / "qa012-diagnostic.json"
    assert qa012_inventory.main(_argv(roots, output)) == qa012_inventory.EXIT_OK

    with pytest.raises(
        schema.SchemaValidationError,
        match="canonical rev4 SHA-256",
    ):
        phase4_driver_d7b.build_qa012_block(authority_path=output)


@pytest.mark.parametrize("mutation", ["missing", "unknown", "duplicate", "malformed"])
def test_cli_requires_exactly_the_five_frozen_prongs(tmp_path, mutation):
    roots = _scope_roots(tmp_path / "scope")
    specs = [f"{name}={roots[name]}" for name in qa012.REQUIRED_SCOPE_PRONGS]
    if mutation == "missing":
        specs.pop()
    elif mutation == "unknown":
        specs.append(f"not_a_prong={tmp_path}")
    elif mutation == "duplicate":
        name = qa012.REQUIRED_SCOPE_PRONGS[0]
        specs.append(f"{name}={roots[name]}")
    else:
        specs[0] = "not-name-path"
    output = tmp_path / "diagnostic.json"
    args = [part for spec in specs for part in ("--prong", spec)]
    args.extend(("--out", str(output)))

    assert qa012_inventory.main(args) == qa012_inventory.EXIT_USAGE_ERROR
    assert not output.exists()


def test_cli_rejects_missing_or_overlapping_roots_before_scan(tmp_path):
    roots = _scope_roots(tmp_path / "scope")
    output = tmp_path / "diagnostic.json"
    roots[qa012.REQUIRED_SCOPE_PRONGS[0]] = tmp_path / "missing"
    assert (
        qa012_inventory.main(_argv(roots, output))
        == qa012_inventory.EXIT_INGRESS_ERROR
    )
    assert not output.exists()

    roots = _scope_roots(tmp_path / "second-scope")
    roots[qa012.REQUIRED_SCOPE_PRONGS[1]] = roots[
        qa012.REQUIRED_SCOPE_PRONGS[0]
    ]
    assert (
        qa012_inventory.main(_argv(roots, output))
        == qa012_inventory.EXIT_USAGE_ERROR
    )
    assert not output.exists()


def test_cli_rejects_symlink_root(tmp_path):
    roots = _scope_roots(tmp_path / "scope")
    target = roots[qa012.REQUIRED_SCOPE_PRONGS[0]]
    alias = tmp_path / "root-alias"
    try:
        alias.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable on this host")
    roots[qa012.REQUIRED_SCOPE_PRONGS[0]] = alias
    output = tmp_path / "diagnostic.json"

    assert (
        qa012_inventory.main(_argv(roots, output))
        == qa012_inventory.EXIT_INGRESS_ERROR
    )
    assert not output.exists()


def test_cli_refuses_output_inside_scanned_root(tmp_path):
    roots = _scope_roots(tmp_path / "scope")
    output = roots[qa012.REQUIRED_SCOPE_PRONGS[0]] / "diagnostic.json"

    assert (
        qa012_inventory.main(_argv(roots, output))
        == qa012_inventory.EXIT_USAGE_ERROR
    )
    assert not output.exists()


def test_cli_detects_output_parent_swap_during_create_once(tmp_path, monkeypatch):
    roots = _scope_roots(tmp_path / "scope")
    output_parent = tmp_path / "output"
    output_parent.mkdir()
    output = output_parent / "diagnostic.json"
    displaced = tmp_path / "displaced-output"
    original_create_once = qa012_inventory._DirectoryAnchor.create_once
    swap_blocked = False

    def swap_parent_then_publish(anchor, name, data, **kwargs):
        nonlocal swap_blocked
        if anchor.label == "QA-012 diagnostic output parent":
            try:
                output_parent.rename(displaced)
            except OSError:
                swap_blocked = True
            else:
                output_parent.mkdir()
        return original_create_once(anchor, name, data, **kwargs)

    monkeypatch.setattr(
        qa012_inventory._DirectoryAnchor,
        "create_once",
        swap_parent_then_publish,
    )

    result = qa012_inventory.main(_argv(roots, output))
    if swap_blocked:
        assert result == qa012_inventory.EXIT_OK
        assert output.is_file()
        assert not displaced.exists()
    else:
        assert result == qa012_inventory.EXIT_INGRESS_ERROR
        assert not output.exists()
        assert not (displaced / output.name).exists()
        assert list(displaced.iterdir()) == []


def test_canonical_authority_declared_sizes_fit_operational_limits():
    authority = qa012.load_authority_manifest()
    sizes = [entry["size"] for entry in authority["entries"]]

    assert max(sizes) <= qa012.MAX_QA_FILE_BYTES
    assert sum(sizes) <= qa012.MAX_QA_TOTAL_BYTES
    assert max(sizes) > schema.MAX_ARTIFACT_BYTES
    assert sum(sizes) > 512 * 1024 * 1024


def test_manifest_rejects_just_over_file_and_total_caps_without_large_bytes(
    tmp_path,
):
    manifest = qa012.build_inventory_manifest(_scope_roots(tmp_path / "scope"))

    oversized_file = copy.deepcopy(manifest)
    oversized_file["files"][0]["size"] = qa012.MAX_QA_FILE_BYTES + 1
    with pytest.raises(schema.SchemaValidationError, match="per-file byte limit"):
        qa012.validate_inventory_manifest(oversized_file)

    oversized_total = copy.deepcopy(manifest)
    oversized_total["files"][0]["size"] = qa012.MAX_QA_FILE_BYTES
    oversized_total["files"][1]["size"] = qa012.MAX_QA_FILE_BYTES
    oversized_total["files"][2]["size"] = 1
    with pytest.raises(schema.SchemaValidationError, match="aggregate byte limit"):
        qa012.validate_inventory_manifest(oversized_total)


def test_candidate_size_rejects_just_over_file_cap_before_read(
    tmp_path, monkeypatch
):
    root = tmp_path / "root"
    root.mkdir()
    path = root / "input.json"
    path.write_text("{}", encoding="utf-8")
    fake_info = SimpleNamespace(
        st_mode=stat.S_IFREG,
        st_size=qa012.MAX_QA_FILE_BYTES + 1,
        st_file_attributes=0,
    )
    monkeypatch.setattr(qa012.os, "stat", lambda *args, **kwargs: fake_info)

    with pytest.raises(schema.TypedIngressError, match="per-file byte limit"):
        qa012._candidate_file_size(path, root)


def test_scan_passes_explicit_qa_file_cap(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    path = root / "input.json"
    path.write_text("{}", encoding="utf-8")
    observed: dict[str, object] = {}

    def bounded_read(candidate, *, tree_root=None, max_bytes=None):
        observed.update(
            candidate=candidate,
            tree_root=tree_root,
            max_bytes=max_bytes,
        )
        return b"{}"

    monkeypatch.setattr(schema, "read_regular_file_bytes", bounded_read)
    qa012._scan_file(path, root, qa012.REQUIRED_SCOPE_PRONGS[0])

    assert observed["candidate"] == path
    assert observed["tree_root"] == root
    assert observed["max_bytes"] == qa012.MAX_QA_FILE_BYTES


def test_build_rejects_just_over_aggregate_cap_before_large_read(
    tmp_path, monkeypatch
):
    roots = _scope_roots(tmp_path / "scope")
    sizes = iter(
        [
            qa012.MAX_QA_FILE_BYTES,
            qa012.MAX_QA_FILE_BYTES,
            1,
        ]
    )
    original_size = qa012._candidate_file_size

    def virtual_size(path, root):
        try:
            return next(sizes)
        except StopIteration:
            return original_size(path, root)

    scan_calls = 0
    original_scan = qa012._scan_file

    def counting_scan(path, root, scope_prong):
        nonlocal scan_calls
        scan_calls += 1
        return original_scan(path, root, scope_prong)

    monkeypatch.setattr(qa012, "_candidate_file_size", virtual_size)
    monkeypatch.setattr(qa012, "_scan_file", counting_scan)

    with pytest.raises(schema.TypedIngressError, match="aggregate byte limit"):
        qa012.build_inventory_manifest(roots)
    assert scan_calls == 2


def test_build_rejects_manifest_wide_pointer_bytes_during_scan(
    tmp_path, monkeypatch
):
    roots = _scope_roots(tmp_path / "scope")
    pointer = "/wide"
    pointer_bytes = len(pointer.encode("utf-8"))
    monkeypatch.setattr(
        qa012, "MAX_QA_TOTAL_POINTER_BYTES", pointer_bytes * 2 - 1
    )
    scan_calls = 0

    def virtual_scan(path, root, scope_prong):
        nonlocal scan_calls
        scan_calls += 1
        return {
            "scope_prong": scope_prong,
            "path": path.relative_to(root).as_posix(),
            "size": 0,
            "content_hash": "0" * 64,
            "sha256": "0" * 64,
            "hits": [{"line": None, "pointer": pointer}],
        }

    monkeypatch.setattr(qa012, "_scan_file", virtual_scan)

    with pytest.raises(schema.TypedIngressError, match="aggregate pointer"):
        qa012.build_inventory_manifest(roots)
    assert scan_calls == 2


def test_detector_bounds_fanout_without_materializing_all_children():
    class GuardedItemsDict(dict):
        def items(self):
            yield "first", 1
            yield "second", 2
            raise AssertionError("detector eagerly consumed the full fanout")

    with pytest.raises(schema.TypedIngressError, match="traversal-node limit"):
        qa012.detect_format_qa(GuardedItemsDict(), max_nodes=2)


def test_detector_enforces_pointer_and_aggregate_pointer_byte_caps():
    with pytest.raises(schema.TypedIngressError, match="per-pointer byte limit"):
        qa012.detect_format_qa(
            {"long-key": {"format": "QA"}},
            max_pointer_bytes=len("/long-key/format".encode("utf-8")) - 1,
        )

    document = {
        "a": {"format": "QA"},
        "b": {"format": "QA"},
    }
    first_pointer_bytes = len("/a/format".encode("utf-8"))
    with pytest.raises(schema.TypedIngressError, match="aggregate pointer"):
        qa012.detect_format_qa(
            document,
            max_total_pointer_bytes=first_pointer_bytes,
        )


def test_build_rejects_same_path_content_change_after_scan(tmp_path, monkeypatch):
    roots = _scope_roots(tmp_path / "scope")
    original_scan = qa012._scan_file
    changed = False

    def mutate_after_scan(path, root, scope_prong):
        nonlocal changed
        entry = original_scan(path, root, scope_prong)
        if not changed:
            path.write_text('{"format":"QA"}\n', encoding="utf-8")
            changed = True
        return entry

    monkeypatch.setattr(qa012, "_scan_file", mutate_after_scan)

    with pytest.raises(schema.TypedIngressError, match="content changed"):
        qa012.build_inventory_manifest(roots)


def test_build_rejects_same_path_replacement_after_scan(tmp_path, monkeypatch):
    roots = _scope_roots(tmp_path / "scope")
    original_scan = qa012._scan_file
    replaced = False

    def replace_after_scan(path, root, scope_prong):
        nonlocal replaced
        entry = original_scan(path, root, scope_prong)
        if not replaced:
            replacement = path.with_name(path.name + ".replacement")
            replacement.write_bytes(path.read_bytes())
            replacement.replace(path)
            replaced = True
        return entry

    monkeypatch.setattr(qa012, "_scan_file", replace_after_scan)

    with pytest.raises(schema.TypedIngressError, match="identity changed"):
        qa012.build_inventory_manifest(roots)


def test_streaming_recheck_preserves_multiblock_dropbox_hash(tmp_path, monkeypatch):
    roots = _scope_roots(tmp_path / "scope")
    monkeypatch.setattr(qa012, "DROPBOX_CONTENT_BLOCK_BYTES", 4)

    manifest = qa012.build_inventory_manifest(roots)

    qa012.validate_inventory_manifest(manifest)


def test_cli_module_help_is_non_authorizing(capsys):
    assert qa012_inventory.main(["--help"]) == 0
    assert "non-authorizing" in capsys.readouterr().out
