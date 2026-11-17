from copy import deepcopy
import pytest

from qb_data.acl_menus import MenuOracle
from qb_data.acl_protocol import ArmSpec, prepare
from scripts.acl_reviewed_subset import (
    OUTPUT, REGISTRY, SELECTION, bind_review, recover_review, require_selection,
)
from scripts.stopdff_v5.identity import load_json_strict


@pytest.mark.skipif(not SELECTION.exists(), reason="private review has not been materialized")
def test_frozen_scope_and_conditional_forms():
    selection, registry = load_json_strict(SELECTION), load_json_strict(REGISTRY)
    require_selection(selection)
    oracle = MenuOracle([], aliases=registry)
    for canonical, spellings in registry["classes"].items():
        assert all(oracle.class_id(label) == canonical for label in (canonical, *spellings))
    with pytest.raises(ValueError, match="unreviewed"):
        oracle.class_id("unregistered synthetic spelling")
    selection["donors"][0]["referent"] = "changed decision"
    with pytest.raises(ValueError, match="reviewed"):
        require_selection(selection)


def test_other_archive_rejected_before_reading_members(tmp_path):
    path = tmp_path / "other.zip"
    path.write_bytes(b"other file")
    with pytest.raises(ValueError, match="archive digest"):
        recover_review(path)


def test_review_envelope_authenticates_and_uses_private_bytes(tmp_path, monkeypatch):
    import hashlib
    import json
    from qb_data.acl_protocol import digest
    from scripts import acl_reviewed_subset as subset
    aead = pytest.importorskip("cryptography.hazmat.primitives.ciphers.aead")
    if not hasattr(aead, "AESSIV"):
        pytest.skip("AESSIV requires the acl-private-review extra")

    archive = tmp_path / "fixture"
    archive.write_bytes(b"synthetic private fixture")
    registry = {"synthetic": True}
    selection = {"oracle_sha256": digest(registry)}
    left, right = json.dumps(selection), json.dumps(registry)
    monkeypatch.setattr(subset, "ARCHIVE_DIGEST", hashlib.sha256(archive.read_bytes()).hexdigest())
    monkeypatch.setattr(subset, "SELECTION_DIGEST", digest(selection))
    key = hashlib.sha512(subset.ARCHIVE_KEY_DOMAIN + archive.read_bytes()).digest()
    sealed = aead.AESSIV(key).encrypt(json.dumps([left, right]).encode(), [subset.REVIEW_DOMAIN])
    envelope = tmp_path / "fixture.siv"
    envelope.write_bytes(sealed)
    assert recover_review(archive, envelope) == (left.encode(), right.encode())
    envelope.write_bytes(bytes([sealed[0] ^ 1]) + sealed[1:])
    with pytest.raises(ValueError, match="authentication"):
        recover_review(archive, envelope)


def test_smoke_evidence_cannot_label_fixture_as_reviewed_real_data(tmp_path, monkeypatch):
    import json
    from scripts import acl_reviewed_subset as subset
    from scripts.acl_smoke_receipt import checked_review
    from scripts.acl_smoke_data import inputs
    from qb_data.acl_protocol import digest

    frozen = {"dataset_sha256": digest(inputs()), "menus_sha256": "menu"}
    assert checked_review(frozen, "synthetic held-out engineering smoke") == {}
    selection = {"materialized_dataset_sha256": "different", "questions": [], "oracle_sha256": "oracle",
                 "materialized_reservoir_sha256": "reservoir"}
    path = tmp_path/"selection.json"
    path.write_text(json.dumps(selection), encoding="utf-8")
    monkeypatch.setattr(subset, "SELECTION", path)
    monkeypatch.setattr(subset, "SELECTION_DIGEST", digest(selection))
    with pytest.raises(ValueError, match="source-bound"):
        checked_review(frozen, "reviewed ACL subset engineering smoke")


@pytest.mark.skipif(not (OUTPUT / "receipt.json").exists(),
                    reason="privately held source freeze has not been materialized")
def test_real_review_uses_full_scientific_workflow():
    import yaml
    from pathlib import Path
    from scripts.acl_reviewed_subset import class_review
    from scripts.acl_expansion import load_inputs
    from scripts.acl_smoke_receipt import checked_review
    from qb_data.acl_protocol import digest

    dataset = load_json_strict(OUTPUT / "dataset.json")
    donors = load_json_strict(OUTPUT / "reservoir.json")
    cfg = yaml.safe_load(Path("configs/acl_expansion/reviewed_geography_smoke.yaml").read_bytes())
    selection, registry = load_json_strict(SELECTION), load_json_strict(REGISTRY)
    reviewed = class_review(dataset, donors, selection, registry)
    assert load_json_strict(cfg["inputs"]["class_review"]) == reviewed
    for row in reviewed["questions"].values():
        assert len(row["classes"]) == len(donors) + 1
        assert sum(d["judgment"] == "accepted" for d in row["classes"].values()) == 1
    for seed in (1, 2, 3):
        prepared = load_inputs(cfg, "uniform_global", None, seed, None)
        assert prepared.identity["source_review"]["class_review_sha256"] == digest(reviewed)
        assert prepared.identity["source_review"]["review_sha256"] == bind_review(
            prepared, dataset, selection, registry)["review_sha256"]
        assert checked_review(prepared.identity, "reviewed ACL subset engineering smoke",
                              prepared.menus) == {"source_review": prepared.identity["source_review"]}
        changed = deepcopy(prepared.identity)
        changed["source_review"]["review_sha256"] = "0" * 64
        with pytest.raises(ValueError, match="adjudication digest"):
            checked_review(changed, "reviewed ACL subset engineering smoke", prepared.menus)


@pytest.mark.skipif(not (OUTPUT / "receipt.json").exists(),
                    reason="privately held source freeze has not been materialized")
def test_materialized_real_menu_coverage_and_tampering():
    dataset = load_json_strict(OUTPUT / "dataset.json")
    donors = load_json_strict(OUTPUT / "reservoir.json")
    selection, registry = load_json_strict(SELECTION), load_json_strict(REGISTRY)
    for menu in ("baseline", "uniform_category", "uniform_global", "mixed", "prefix_tfidf"):
        spec = ArmSpec.from_dict({"version": 1, "menu": menu, "k": 4, "seed": 77,
                                  "reward": {"arm": "baseline"}})
        p = prepare(spec, dataset, registry, reservoir=donors,
                    split_roles={"calibration": "train", "selection": "val", "test": "test"})
        assert bind_review(p, dataset, selection, registry)["reviewed_questions"] == 7
        assert sum(len(q.cumulative_prefixes) for qs in p.questions.values() for q in qs) == 70
    changed = deepcopy(dataset)
    changed["questions"][0]["answer"]["accepted"].append("unreviewed")
    with pytest.raises(ValueError, match="pinned"):
        bind_review(p, changed, selection, registry)
    qid = next(iter(p.groups))
    p.groups[qid] = "relabelled-group"
    with pytest.raises(ValueError, match="group"):
        bind_review(p, dataset, selection, registry)
    swapped = prepare(spec, dataset, registry, reservoir=donors,
                      split_roles={"test": "train", "selection": "val", "calibration": "test"})
    with pytest.raises(ValueError, match="split"):
        bind_review(swapped, dataset, selection, registry)
