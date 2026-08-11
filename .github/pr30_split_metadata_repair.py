#!/usr/bin/env python3
"""Apply the centrally adjudicated PR #30 split-metadata repair.

This temporary script is invoked once by a GitHub Actions transaction and is
removed after the resulting code commit is validated.
"""

from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    target = Path(path)
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"expected exactly one anchor in {path}, found {count}: {old[:80]!r}"
        )
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def main() -> None:
    build_helper_anchor = (
        "\n\ndef warn_on_low_retention(split_name: str, metadata: "
        "dict[str, Any], threshold: float) -> None:\n"
    )
    build_helper = '''

def build_retained_split_metadata(
    train: List[MCQuestion],
    val: List[MCQuestion],
    test: List[MCQuestion],
) -> dict[str, Any]:
    """Describe the retained MC splits consumed by StopDFF v5 staging."""
    split_questions = {
        "train": train,
        "val": val,
        "test": test,
    }
    blocks: dict[str, dict[str, Any]] = {}
    counts: list[int] = []
    for split_name, questions in split_questions.items():
        category_counts: dict[str, int] = {}
        for question in questions:
            category = str(question.category)
            category_counts[category] = category_counts.get(category, 0) + 1
        blocks[split_name] = {
            "count": len(questions),
            "categories": {
                category: category_counts[category]
                for category in sorted(category_counts)
            },
        }
        counts.append(len(questions))

    total = sum(counts)
    ratios = (
        [count / total for count in counts]
        if total
        else [0.0, 0.0, 0.0]
    )
    return {
        **blocks,
        "total_questions": total,
        "split_ratios": ratios,
    }
'''
    replace_once(
        "scripts/build_mc_dataset.py",
        build_helper_anchor,
        build_helper + build_helper_anchor,
    )

    replace_once(
        "scripts/build_mc_dataset.py",
        '''    save_json(output_dir / "test_dataset.json", test)

    build_metadata = {
''',
        '''    save_json(output_dir / "test_dataset.json", test)
    save_json(
        output_dir / "split_metadata.json",
        build_retained_split_metadata(train, val, test),
    )

    build_metadata = {
''',
    )

    producer_anchor = '''def _record_value(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = record.get(key)
        if value is not None:
            return value
    return None


'''
    producer_helpers = '''def _strict_nonnegative_int(value: Any) -> bool:
    """Return whether value is a JSON integer count, excluding booleans."""
    return (
        isinstance(value, int)
        and not isinstance(value, bool)
        and value >= 0
    )


def _record_category_distribution(
    records: list[dict[str, Any]],
) -> dict[str, int] | None:
    """Derive a deterministic category distribution from staged records."""
    counts: dict[str, int] = {}
    for record in records:
        category = _record_value(record, ("category",))
        if not isinstance(category, str):
            return None
        counts[category] = counts.get(category, 0) + 1
    return {key: counts[key] for key in sorted(counts)}


def _split_metadata_consistent(
    metadata: object,
    records_by_split: dict[str, list[dict[str, Any]]],
) -> bool:
    """Verify retained-split metadata against the staged dataset bytes."""
    split_names = ("train", "val", "test")
    expected_top = {*split_names, "total_questions", "split_ratios"}
    if not isinstance(metadata, dict) or set(metadata) != expected_top:
        return False

    counts: list[int] = []
    for split in split_names:
        block = metadata.get(split)
        if not isinstance(block, dict) or set(block) != {"count", "categories"}:
            return False
        count = block.get("count")
        categories = block.get("categories")
        if not _strict_nonnegative_int(count) or not isinstance(categories, dict):
            return False
        if any(
            not isinstance(key, str)
            or not _strict_nonnegative_int(value)
            for key, value in categories.items()
        ):
            return False
        expected_categories = _record_category_distribution(records_by_split[split])
        if expected_categories is None or categories != expected_categories:
            return False
        if count != len(records_by_split[split]) or sum(categories.values()) != count:
            return False
        counts.append(count)

    total = metadata.get("total_questions")
    if not _strict_nonnegative_int(total) or total != sum(counts):
        return False

    ratios = metadata.get("split_ratios")
    if not isinstance(ratios, list) or len(ratios) != len(split_names):
        return False
    expected_ratios = (
        [count / total for count in counts]
        if total
        else [0.0, 0.0, 0.0]
    )
    for actual, expected in zip(ratios, expected_ratios):
        if (
            isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not math.isfinite(float(actual))
            or not math.isclose(
                float(actual),
                expected,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        ):
            return False
    return True


'''
    replace_once(
        "scripts/stopdff_v5/producers.py",
        producer_anchor,
        producer_anchor + producer_helpers,
    )

    replace_once(
        "scripts/stopdff_v5/producers.py",
        '''    records_by_split = {
        split: _dataset_records(staged[f"{split}_dataset.json"])
        for split in ("train", "val", "test")
    }
    split_checks, split_ok = _split_semantics(records_by_split)
''',
        '''    records_by_split = {
        split: _dataset_records(staged[f"{split}_dataset.json"])
        for split in ("train", "val", "test")
    }
    checks["split_metadata_consistent"] = _split_metadata_consistent(
        decoded["split_metadata.json"],
        records_by_split,
    )
    split_checks, split_ok = _split_semantics(records_by_split)
''',
    )

    replace_once(
        "scripts/stopdff_v5/producers.py",
        '''    all_ok = (checks["threshold_sidecar_ok"] and checks["calibration_fit_split_is_val"]
              and split_ok and trajectory_ok
              and checks["build_metadata_retention_consistent"]
              and checks["myopic_semantics_valid"])
''',
        '''    all_ok = (checks["threshold_sidecar_ok"] and checks["calibration_fit_split_is_val"]
              and split_ok and trajectory_ok
              and checks["split_metadata_consistent"]
              and checks["build_metadata_retention_consistent"]
              and checks["myopic_semantics_valid"])
''',
    )

    replace_once(
        "tests/test_build_mc_dataset.py",
        '''    build_metadata_entry,
    make_mc_builder,
''',
        '''    build_metadata_entry,
    build_retained_split_metadata,
    make_mc_builder,
''',
    )

    build_tests = '''


def test_retained_split_metadata_uses_retained_mc_outputs() -> None:
    """The sidecar describes retained splits with deterministic categories."""
    train = [
        SimpleNamespace(category="Science"),
        SimpleNamespace(category="Arts"),
        SimpleNamespace(category="Science"),
    ]
    val = [SimpleNamespace(category="History")]
    test = [SimpleNamespace(category="Arts"), SimpleNamespace(category="History")]

    metadata = build_retained_split_metadata(train, val, test)

    assert metadata == {
        "train": {"count": 3, "categories": {"Arts": 1, "Science": 2}},
        "val": {"count": 1, "categories": {"History": 1}},
        "test": {"count": 2, "categories": {"Arts": 1, "History": 1}},
        "total_questions": 6,
        "split_ratios": [0.5, 1 / 6, 1 / 3],
    }
    assert list(metadata["train"]["categories"]) == ["Arts", "Science"]


def test_retained_split_metadata_handles_all_empty_splits() -> None:
    """An all-filtered build has a finite, deterministic zero-ratio sidecar."""
    assert build_retained_split_metadata([], [], []) == {
        "train": {"count": 0, "categories": {}},
        "val": {"count": 0, "categories": {}},
        "test": {"count": 0, "categories": {}},
        "total_questions": 0,
        "split_ratios": [0.0, 0.0, 0.0],
    }
'''
    test_build_path = Path("tests/test_build_mc_dataset.py")
    test_build_path.write_text(
        test_build_path.read_text(encoding="utf-8") + build_tests,
        encoding="utf-8",
    )

    replace_once(
        "tests/test_stopdff_v5_producers.py",
        '''def _record(qid: str, question: str, answer: str) -> dict:
    return {
        "qid": qid,
        "question": question,
        "answer_primary": answer,
    }
''',
        '''def _record(
    qid: str,
    question: str,
    answer: str,
    category: str = "General",
) -> dict:
    return {
        "qid": qid,
        "question": question,
        "answer_primary": answer,
        "category": category,
    }
''',
    )

    replace_once(
        "tests/test_stopdff_v5_producers.py",
        '''    _write_json(sources / "split_metadata.json", {"claimed_disjoint": True})
''',
        '''    _write_json(
        sources / "split_metadata.json",
        {
            "train": {"count": 1, "categories": {"General": 1}},
            "val": {"count": 1, "categories": {"General": 1}},
            "test": {"count": 1, "categories": {"General": 1}},
            "total_questions": 3,
            "split_ratios": [1 / 3, 1 / 3, 1 / 3],
        },
    )
''',
    )

    producer_tests = '''


@pytest.mark.parametrize(
    "mutation",
    ["count", "category", "total", "ratio", "shape"],
)
def test_stage_raw_inputs_rejects_stale_split_metadata(tmp_path, mutation):
    """The retained-split sidecar must agree with the staged dataset bytes."""
    source_paths = _source_paths(tmp_path)
    metadata_path = source_paths["split_metadata.json"]
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if mutation == "count":
        payload["train"]["count"] = 2
    elif mutation == "category":
        payload["val"]["categories"] = {"Stale": 1}
    elif mutation == "total":
        payload["total_questions"] = 4
    elif mutation == "ratio":
        payload["split_ratios"] = [0.5, 0.25, 0.25]
    else:
        payload["unexpected"] = True
    _write_json(metadata_path, payload)

    with pytest.raises(ValueError, match="raw-input semantic checks failed"):
        producers.stage_raw_inputs(source_paths, tmp_path / "staged")
'''
    test_producer_path = Path("tests/test_stopdff_v5_producers.py")
    test_producer_path.write_text(
        test_producer_path.read_text(encoding="utf-8") + producer_tests,
        encoding="utf-8",
    )

    replace_once(
        "AGENTS.md",
        '''`build_mc_dataset.py` writes `train_dataset.json`, `val_dataset.json`, and
`test_dataset.json` as the canonical downstream inputs. `mc_dataset.json`
remains as a combined legacy/debug artifact. By default, `run_baselines.py`
''',
        '''`build_mc_dataset.py` writes `train_dataset.json`, `val_dataset.json`,
`test_dataset.json`, and retained-split `split_metadata.json` as canonical
downstream inputs. `mc_dataset.json` remains as a combined legacy/debug
artifact. By default, `run_baselines.py`
''',
    )

    replace_once(
        "docs/stopdff_v5/REPRODUCTION.md",
        '''`split_metadata.json` live under
`data/processed/`; `calibration.json` and `stopdff.json` under `paper_exports/`;
`threshold_manifest.json{,.sha256}` at the repo root). These large inputs are reproducible
from the belief-feature pipeline (`scripts/build_mc_dataset.py`, `scripts/fresh_split.py`,
`scripts/compute_prefix_calibration.py`, `scripts/compute_stopdff.py`) and are synced via
''',
        '''`split_metadata.json` live under
`data/processed/`; `calibration.json` and `stopdff.json` under `paper_exports/`;
`threshold_manifest.json{,.sha256}` at the repo root). `split_metadata.json` is emitted by
`scripts/build_mc_dataset.py` from the retained MC train/validation/test bytes and is
validated against those exact datasets during v5 staging. These large inputs are reproducible
from the belief-feature pipeline (`scripts/build_mc_dataset.py`,
`scripts/compute_prefix_calibration.py`, `scripts/compute_stopdff.py`) and are synced via
''',
    )


if __name__ == "__main__":
    main()
