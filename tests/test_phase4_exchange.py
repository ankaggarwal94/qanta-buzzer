"""Exercise the Phase 4 exchange protocol through the Python and CLI entry points."""

import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor

import pytest

from reproducibility.colm_aims_2026 import phase4_exchange as exchange
from reproducibility.colm_aims_2026 import schema

MANIFEST = "RETURN_MANIFEST.json"
FILES = {
    "nested/records/data.jsonl": b'{"score": 1}\n',
    "nested/RETURN_MANIFEST.json": b"ordinary nested file",
    "empty.bin": b"",
    "z.bin": b"\x00\xff\r\n\x1a",
}
META = {"retry": 1, "nested": {"activation_digest": "a" * 64}}


def snapshot(root):
    return {
        p.relative_to(root).as_posix(): p.read_bytes()
        for p in root.rglob("*")
        if p.is_file()
    }


@pytest.fixture
def run(tmp_path):
    root = tmp_path / "run"
    for name, data in FILES.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (root / "empty-directory").mkdir()
    return root


def publish(run, exchange_root, **kwargs):
    return exchange.publish_run_results(
        run, exchange_root, name="trial", provenance=META, **kwargs
    )


@pytest.fixture
def bundle(run, tmp_path):
    return publish(run, tmp_path / "exchange")


def refuse_pull(bundle, dest, match):
    before = snapshot(bundle)
    with pytest.raises(exchange.ExchangeError, match=match):
        exchange.pull_verified(bundle, dest)
    assert not dest.exists()
    assert not list(dest.parent.glob(f"{dest.name}.pulling-*"))
    assert snapshot(bundle) == before


@pytest.mark.parametrize("kind", ["phase4_exchange_return", "phase4_exchange_staging"])
def test_round_trip(run, tmp_path, kind):
    root = tmp_path / "exchange"
    calls = []

    def now():
        (partial,) = root.iterdir()
        assert partial.name.startswith("trial.partial-")
        assert snapshot(partial) == FILES
        calls.append("now")
        return "2026-08-24T00:00:00+00:00"

    bundle = publish(run, root, now=now, manifest_type=kind)
    assert calls == ["now"]
    assert list(root.iterdir()) == [bundle] and bundle == root / "trial"
    manifest = json.loads((bundle / MANIFEST).read_bytes())
    assert type(manifest["schema_version"]) is int and manifest["schema_version"] == 2
    assert manifest["generated_at"] == "2026-08-24T00:00:00+00:00"
    assert manifest["platform"] and manifest["node"]
    assert manifest["entries"] == [
        {"path": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        for name, data in sorted(FILES.items())
    ]
    intake = tmp_path / "intake"
    assert exchange.pull_verified(bundle, intake) == {
        "verdict": "PULLED",
        "file_count": len(FILES),
        "total_bytes": sum(map(len, FILES.values())),
        "provenance": META,
        "manifest_type": kind,
    }
    assert snapshot(intake) == FILES
    assert not (intake / "empty-directory").exists()
    assert not (intake / MANIFEST).exists()


def test_cli_publish_then_pull(run, tmp_path, capsys):
    root = tmp_path / "exchange"
    intake = tmp_path / "intake"
    metadata = tmp_path / "provenance.json"
    metadata.write_text(json.dumps(META), encoding="utf-8")
    assert (
        exchange.main(
            [
                "publish",
                str(run),
                str(root),
                "--name",
                "trial",
                "--provenance",
                str(metadata),
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == str(root / "trial")
    assert exchange.main(["pull", str(root / "trial"), str(intake)]) == 0
    assert json.loads(capsys.readouterr().out)["provenance"] == META
    assert snapshot(intake) == FILES
    assert exchange.main(["pull", str(root / "trial"), str(intake)]) == 1
    assert "already exists" in capsys.readouterr().err


@pytest.mark.parametrize(
    "layout", ["missing", "empty", "nested_exchange", "nested_run", "repo"]
)
def test_incompatible_roots(run, tmp_path, monkeypatch, layout):
    root = tmp_path / "exchange"
    match = "inside"
    if layout in ("missing", "empty"):
        run = tmp_path / "other"
        match = "missing" if layout == "missing" else "empty"
        if layout == "empty":
            run.mkdir()
    elif layout == "nested_run":
        root = run.parent
    elif layout == "nested_exchange":
        root = run / "exchange"
    else:
        monkeypatch.setattr(exchange, "_REPO_ROOT", root)
    with pytest.raises(exchange.ExchangeError, match=match):
        publish(run, root)
    assert not (root / "trial").exists()
    assert not list(root.glob("trial.partial-*"))


@pytest.mark.parametrize("name", ["", ".", "..", "a/b", "a\\b"])
def test_name_must_be_one_component(run, tmp_path, name):
    root = tmp_path / "exchange"
    with pytest.raises(exchange.ExchangeError, match="name"):
        exchange.publish_run_results(run, root, name=name, provenance={})
    assert not root.exists()


@pytest.mark.parametrize("name", [MANIFEST, MANIFEST.lower()])
def test_top_level_manifest_is_reserved(run, tmp_path, name):
    (run / name).touch()
    with pytest.raises(exchange.ExchangeError, match="reserved"):
        publish(run, tmp_path / "exchange")


@pytest.mark.parametrize("location", ["same", "nested", "repo"])
def test_invalid_intake_destination(bundle, tmp_path, monkeypatch, location):
    dest = bundle if location == "same" else bundle / "intake"
    if location == "repo":
        dest = tmp_path / "repo" / "intake"
        monkeypatch.setattr(exchange, "_REPO_ROOT", dest.parent)
    before = snapshot(bundle)
    with pytest.raises(exchange.ExchangeError, match="inside"):
        exchange.pull_verified(bundle, dest)
    assert snapshot(bundle) == before


@pytest.mark.parametrize("operation", ["publish", "pull"])
def test_occupied_destination_remains_intact(bundle, run, tmp_path, operation):
    dest = bundle if operation == "publish" else tmp_path / "intake"
    dest.mkdir(exist_ok=True)
    (dest / "sentinel").write_bytes(b"prior")
    before = snapshot(dest)
    with pytest.raises(exchange.ExchangeError, match="already exists"):
        if operation == "publish":
            publish(run, bundle.parent)
        else:
            exchange.pull_verified(bundle, dest)
    assert snapshot(dest) == before


@pytest.mark.parametrize("bad_bytes", [None, b'{"entries":', b'{"size": Infinity}'])
def test_manifest_required_and_strict(bundle, tmp_path, bad_bytes):
    if bad_bytes is None:
        (bundle / MANIFEST).unlink()
    else:
        (bundle / MANIFEST).write_bytes(bad_bytes)
    refuse_pull(bundle, tmp_path / "intake", MANIFEST)


@pytest.mark.parametrize("change", ["missing", "extra", "size", "sha256"])
def test_intake_reconciles_members_and_hashes(bundle, tmp_path, change):
    path = bundle / "z.bin"
    message = change
    if change == "missing":
        path.unlink()
    elif change == "extra":
        (bundle / "extra.txt").touch()
    else:
        data = FILES["z.bin"] + b"x" if change == "size" else b"x" * len(FILES["z.bin"])
        path.write_bytes(data)
        if change == "size":
            message = (
                f"z.bin: expected size {len(FILES['z.bin'])}, observed {len(data)}"
            )
        else:
            message = f"z.bin: expected sha256 {hashlib.sha256(FILES['z.bin']).hexdigest()}, observed {hashlib.sha256(data).hexdigest()}"
    refuse_pull(bundle, tmp_path / "intake", message)


@pytest.mark.parametrize("change", ["manifest", "additional_file"])
def test_pull_rechecks_synced_bundle_before_publishing(
    bundle, tmp_path, monkeypatch, change
):
    copy_file = exchange._copy_file_hashed

    def copy_then_sync(src, dest):
        result = copy_file(src, dest)
        if src == bundle / "z.bin":
            if change == "manifest":
                path = bundle / MANIFEST
                manifest = json.loads(path.read_bytes())
                manifest["provenance"]["retry"] += 1
                path.write_bytes(exchange.schema.encode_json(manifest))
            else:
                (bundle / "arrived.txt").write_text("just synced", encoding="utf-8")
        return result

    monkeypatch.setattr(exchange, "_copy_file_hashed", copy_then_sync)
    intake = tmp_path / "intake"
    with pytest.raises(exchange.ExchangeError, match="changed during pull"):
        exchange.pull_verified(bundle, intake)
    assert not intake.exists()
    assert not list(tmp_path.glob("intake.pulling-*"))


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("schema_version", True),
        ("schema_version", 1),
        ("entries", []),
        ("provenance", None),
        ("manifest_type", ""),
        ("total_bytes", 0),
        ("file_count", 1),
    ],
)
def test_invalid_manifest_fields(bundle, tmp_path, key, value):
    path = bundle / MANIFEST
    manifest = json.loads(path.read_bytes())
    manifest[key] = value
    path.write_text(json.dumps(manifest), encoding="utf-8")
    refuse_pull(bundle, tmp_path / "intake", key)


@pytest.mark.parametrize(
    ("name", "match"),
    [
        ("../outside", "escapes"),
        ("/absolute", "escapes"),
        ("nested\\data", "escapes"),
        (MANIFEST, "reserved"),
        ("Z.bin", "case"),
        ("z.bin", "duplicate"),
    ],
)
def test_manifest_member_names(bundle, tmp_path, name, match):
    path = bundle / MANIFEST
    manifest = json.loads(path.read_bytes())
    entry = {**manifest["entries"][-1], "path": name}
    manifest["entries"].append(entry)
    manifest["file_count"] += 1
    manifest["total_bytes"] += entry["size"]
    path.write_text(json.dumps(manifest), encoding="utf-8")
    refuse_pull(bundle, tmp_path / "intake", match)


def test_case_sensitive_source_requires_portable_names(run, tmp_path):
    (run / "Z.bin").write_bytes(b"different")
    if (run / "z.bin").read_bytes() == b"different":
        pytest.skip("requires a case-sensitive temporary filesystem")
    with pytest.raises(exchange.ExchangeError, match="case collision"):
        publish(run, tmp_path / "exchange")


def test_linked_member(run, tmp_path):
    try:
        (run / "alias").symlink_to(run / "empty.bin")
    except (OSError, NotImplementedError):
        pytest.skip("requires symlink creation")
    with pytest.raises(exchange.ExchangeError, match="symlink"):
        publish(run, tmp_path / "exchange")


def test_member_walk_uses_host_link_predicate(run, tmp_path, monkeypatch):
    monkeypatch.setattr(
        schema, "is_filesystem_link", lambda path: path.name == "empty.bin"
    )
    with pytest.raises(exchange.ExchangeError, match="empty.bin"):
        publish(run, tmp_path / "exchange")


def test_pull_verifies_copied_bytes(bundle, tmp_path, monkeypatch):
    original = exchange._copy_file_hashed

    def changed_copy(src, dest):
        size, _ = original(src, dest)
        return size, "0" * 64

    monkeypatch.setattr(exchange, "_copy_file_hashed", changed_copy)
    refuse_pull(bundle, tmp_path / "intake", "sha256")


@pytest.mark.parametrize("operation", ["publish", "pull"])
def test_rename_failure_cleans_partial(bundle, run, tmp_path, monkeypatch, operation):
    def unavailable(src, dst):
        raise OSError("destination unavailable")

    monkeypatch.setattr(os, "rename", unavailable)
    dest = tmp_path / "second" / "trial"
    with pytest.raises(exchange.ExchangeError, match="destination unavailable"):
        if operation == "publish":
            publish(run, dest.parent)
        else:
            exchange.pull_verified(bundle, dest)
    assert not dest.exists()
    assert not list(dest.parent.iterdir())


def test_concurrent_intakes_publish_once(bundle, tmp_path):
    dest = tmp_path / "intake"

    def attempt():
        try:
            return exchange.pull_verified(bundle, dest)["verdict"]
        except exchange.ExchangeError:
            return "occupied"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = [pool.submit(attempt) for _ in range(2)]
    assert sorted(f.result() for f in outcomes) == ["PULLED", "occupied"]
    assert snapshot(dest) == FILES
    assert not list(dest.parent.glob(f"{dest.name}.pulling-*"))
