"""Wire compatibility and workspace bounds for StopDFF adapter and cell output."""
from __future__ import annotations

import gzip
import io
import itertools
import json
import random
import statistics
import tracemalloc

import numpy as np
import pytest

from scripts.stopdff_v5 import bootstrap, rowio


def test_row_encoders_match_the_original_one_shot_gzip(tmp_path):
    rng = random.Random(11)
    rows = [
        {"食": {"b": "café", "a": []}, "float": 0.125, "data": rng.randbytes(851).hex()}
        for _ in range(150)
    ]
    for sample in ([], rows):
        payload = "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in sample
        ).encode("utf-8")
        original = io.BytesIO()
        with gzip.GzipFile(fileobj=original, mode="wb", mtime=0, compresslevel=6) as archive:
            archive.write(payload)
        dest = tmp_path / "fit_rows.jsonl.gz"
        rowio.write_jsonl_gz(dest, iter(sample))
        assert dest.read_bytes() == original.getvalue() == rowio.dumps_rows(iter(sample))
        assert rowio.read_jsonl_gz(dest) == sample


def test_row_write_rolls_back_late_serialization_failure(tmp_path):
    dest = tmp_path / "fit_rows.jsonl.gz"
    dest.write_bytes(b"previous artifact")

    def rows():
        for i in range(37):
            yield {"example": i}
        yield {"example": float("nan")}

    with pytest.raises(ValueError):
        rowio.write_jsonl_gz(dest, rows())
    assert dest.read_bytes() == b"previous artifact"
    assert list(tmp_path.iterdir()) == [dest]


@pytest.mark.parametrize("write_to_disk", [False, True])
def test_row_encoding_memory_is_bounded_by_row_size(tmp_path, write_to_disk):
    # Disk output must bound even the compressed bytes held in memory.
    # In-memory output includes those bytes, so use highly compressible rows there.
    rng = random.Random(23)
    rows = (
        {"data": rng.randbytes(8192).hex() if write_to_disk else "a" * 16384}
        for _ in range(640)
    )
    tracemalloc.start()
    try:
        if write_to_disk:
            rowio.write_jsonl_gz(tmp_path / "rows.gz", rows)
        else:
            rowio.dumps_rows(rows)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 4 * 1024 * 1024, f"row encoding allocated {peak} bytes"


@pytest.mark.parametrize(("n_items", "replicates"), [(1, 1), (7, 19), (1024, 257)])
def test_batched_paired_stats_match_eager_paired_reductions(n_items, replicates):
    shifts = {f"item-{i}": (i % 7) - 3 for i in range(n_items)}
    plan = bootstrap.build_bootstrap_plan(shifts, replicates=replicates, seed=3)
    original_indices = plan.resample_indices.copy()
    ordered = np.array([shifts[item] for item in plan.item_ids], dtype=np.float64)
    actual = bootstrap.cell_bootstrap_stats(shifts, plan)
    for name, values in (("signed", ordered), ("absolute", np.abs(ordered))):
        sampled = values[original_indices]
        for reducer in (np.mean, np.median):
            metric = f"{name}_index_{reducer.__name__}"
            expected = reducer(sampled, axis=1)
            assert actual["point"][metric] == reducer(values)
            assert actual["ci"][metric] == np.percentile(expected, [2.5, 97.5]).tolist()
            if metric == "absolute_index_median":
                assert actual["abs_median_replicates"].tobytes() == expected.tobytes()
    assert plan.resample_indices.tobytes() == original_indices.tobytes()


def test_bootstrap_workspace_is_bounded_independently_of_resample_matrix():
    shifts = {str(i): (i % 9) - 4 for i in range(1024)}
    plan = bootstrap.build_bootstrap_plan(shifts, replicates=1001)
    # Exclude the already-built shared resampling plan from per-cell allocations.
    tracemalloc.start()
    try:
        bootstrap.cell_bootstrap_stats(shifts, plan)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert peak < 4 * 1024 * 1024, f"paired cell reductions allocated {peak} bytes"


def test_all_small_resamples_match_scalar_statistics(monkeypatch):
    """Exhaust every shift vector and draw for 1–3 items, including a partial last batch."""
    monkeypatch.setattr(bootstrap, "_RESAMPLE_BATCH_BYTES", 48)
    for count in range(1, 4):
        draws = list(itertools.product(range(count), repeat=count))
        indices = np.array(draws, dtype=np.int64)
        for shifts in itertools.product((-1, 0, 1), repeat=count):
            rows = [[shifts[item] for item in draw] for draw in draws]
            means, medians = bootstrap._replicate_statistics(np.array(shifts, dtype=float), indices)
            np.testing.assert_array_equal(means, [statistics.mean(row) for row in rows])
            np.testing.assert_array_equal(medians, [statistics.median(row) for row in rows])
