"""Verify immutable ACL training checkpoint blocks and their parent links."""
from pathlib import Path

from qb_data.acl_protocol import digest
from scripts.stopdff_v5.identity import load_json_strict, sha256_file


def hash_tree(path):
    return {p.relative_to(path).as_posix(): sha256_file(p)
            for p in sorted(path.rglob("*")) if p.is_file()}


def verify_block(checkpoint, expected):
    manifest = load_json_strict(checkpoint / "files.json")
    files = hash_tree(checkpoint)
    del files["files.json"]
    if manifest != {**expected, "files": files}:
        raise ValueError("checkpoint content or parent mismatch")
    return manifest


def verify_completed_lineage(folder: Path, settings: dict, receipt: dict) -> None:
    """Require every declared block and payload before accepting the final digest."""
    parent = None
    updates, stride = settings["updates"], settings["checkpoint_updates"]
    for start in range(0, updates, stride):
        end = min(start + stride, updates)
        blocks = list(folder.glob(f"train-{end:06d}-*/files.json"))
        if len(blocks) != 1:
            raise ValueError("expected exactly one completed checkpoint for each block")
        header = {"run_id": folder.name, "first_update": start, "end": end, "parent": parent}
        parent = digest(verify_block(blocks[0].parent, header))
    if parent is None or parent != receipt["checkpoint_sha256"]:
        raise ValueError("completed checkpoint identity mismatch")
