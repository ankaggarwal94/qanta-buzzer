#!/usr/bin/env python3
"""Apply the bounded successor repair for PR #30's live review findings."""
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


def patch_modal_runner() -> None:
    path = "scripts/modal_stopdff_v5_runner.py"
    replace_once(
        path,
        "def _canonical_adapter_subdir(value: object) -> str:\n",
        '''_MODAL_RUN_ID_RE = re.compile(r"(smoke|final)_modal_([0-9a-f]{12})")


def _canonical_modal_run_id(
    value: object,
    *,
    variant: str | None = None,
    run_spec_id: str | None = None,
) -> str:
    """Return the exact one-component run slot bound to a run-spec ID."""
    if not isinstance(value, str) or not value or "\\0" in value:
        raise ValueError("run_id must be a nonempty canonical string")
    parsed = PurePosixPath(value)
    if (
        parsed.is_absolute()
        or ".." in parsed.parts
        or len(parsed.parts) != 1
        or str(parsed) != value
        or _MODAL_RUN_ID_RE.fullmatch(value) is None
    ):
        raise ValueError(f"unsafe or noncanonical run_id: {value!r}")
    if (variant is None) != (run_spec_id is None):
        raise ValueError("run_id binding requires both variant and run_spec_id")
    if variant is not None:
        if (
            variant not in {"smoke", "final"}
            or not isinstance(run_spec_id, str)
            or re.fullmatch(r"[0-9a-f]{64}", run_spec_id) is None
        ):
            raise ValueError("run_id binding inputs are noncanonical")
        expected = f"{variant}_modal_{run_spec_id[:12]}"
        if value != expected:
            raise ValueError("run_id is not bound to run_spec_id")
    return value


def _canonical_adapter_subdir(value: object) -> str:
''',
    )

    replace_once(
        path,
        "def _validated_cached_fvi(\n",
        '''def _canonical_fvi_study_path(root: Path) -> Path:
    """Return the FVI manifest only from its canonical cache slot."""
    root = Path(root)
    manifest_path = root / "fvi_study.json"
    if (
        root.is_symlink()
        or not root.is_dir()
        or manifest_path.is_symlink()
        or not manifest_path.is_file()
    ):
        raise ValueError("FVI cache is incomplete or noncanonical")
    return manifest_path


def _validated_cached_fvi(
''',
    )

    replace_once(
        path,
        '''    manifest_path = out / "fvi_study.json"
    execution_path = out / "fvi_study_execution.json"
    if (
        out.is_symlink()
        or not out.is_dir()
        or manifest_path.is_symlink()
        or not manifest_path.is_file()
        or execution_path.is_symlink()
        or not execution_path.is_file()
    ):
        raise FileExistsError("FVI destination is incomplete or noncanonical")
''',
        '''    try:
        manifest_path = _canonical_fvi_study_path(out)
    except ValueError as exc:
        raise FileExistsError(
            "FVI destination is incomplete or noncanonical"
        ) from exc
    execution_path = out / "fvi_study_execution.json"
    if execution_path.is_symlink() or not execution_path.is_file():
        raise FileExistsError("FVI destination is incomplete or noncanonical")
''',
    )

    replace_once(
        path,
        '''    if binding["bootstrap_plan_id"] != bootstrap_plan_id:
        raise ValueError("bootstrap argument does not match verified run spec")

    # Reject contradictory duplicated wrapper fields instead of trusting them.
''',
        '''    if binding["bootstrap_plan_id"] != bootstrap_plan_id:
        raise ValueError("bootstrap argument does not match verified run spec")
    run_id = _canonical_modal_run_id(
        spec.get("run_id"),
        variant=binding["variant"],
        run_spec_id=binding["run_spec_id"],
    )

    # Reject contradictory duplicated wrapper fields instead of trusting them.
''',
    )

    replace_once(
        path,
        '''    fvi_id = spec_ids["fvi_study_id"]
    fvi_manifest = checker.load_json(
        Path(_p("fvi", fvi_id)) / "fvi_study.json"
    )
''',
        '''    fvi_id = spec_ids["fvi_study_id"]
    fvi_manifest = checker.load_json(
        _canonical_fvi_study_path(Path(_p("fvi", fvi_id)))
    )
''',
    )

    replace_once(
        path,
        '''    run_root = Path(_p("runs", spec["run_id"]))
    if binding["run_spec_id"][:12] not in str(spec["run_id"]):
        raise ValueError("run_id is not bound to run_spec_id")
''',
        '''    run_root = Path(_p("runs", run_id))
''',
    )

    replace_once(
        path,
        '''    result = {"run_id": spec["run_id"], "requested": agg["requested"], "completed": agg["completed"],
''',
        '''    result = {"run_id": run_id, "requested": agg["requested"], "completed": agg["completed"],
''',
    )

    replace_once(
        path,
        '''    vol.reload()
    run_root = Path(_p("runs", run_id))
    agg = checker.load_json(run_root / "aggregate.json")
''',
        '''    vol.reload()
    run_id = _canonical_modal_run_id(run_id)
    run_root = Path(_p("runs", run_id))
    agg = checker.load_json(run_root / "aggregate.json")
''',
    )

    replace_once(
        path,
        '''    spec_manifest = checker.load_json(run_root / "run_spec.json")
    spec_ids = spec_manifest["identity"]["identity"]
''',
        '''    spec_manifest = checker.load_json(run_root / "run_spec.json")
    _canonical_modal_run_id(
        run_id,
        variant=spec_manifest["identity"]["profile_variant"],
        run_spec_id=spec_manifest["id"],
    )
    spec_ids = spec_manifest["identity"]["identity"]
''',
    )

    replace_once(
        path,
        '''    fvi_path = Path(_p(
        "fvi",
        spec_ids["fvi_study_id"],
        "fvi_study.json",
    ))
''',
        '''    fvi_path = _canonical_fvi_study_path(Path(_p(
        "fvi",
        spec_ids["fvi_study_id"],
    )))
''',
    )

    replace_once(
        path,
        '''@app.function(volumes={MNT: vol}, timeout=DAY, max_containers=1)
def validate(run_id: str, adapter_id: str, require_final: bool, require_package: bool) -> dict:
    from pathlib import Path
    from scripts.stopdff_v5 import checker
    vol.reload()
    res = checker.validate_run(Path(_p("runs", run_id)), backend="modal",
                              adapter_bundle=Path(_p("adapters", f"canonical_{adapter_id}")),
                              require_final_profile=require_final, require_package=require_package)
    return {"passed": res.passed, "errors": res.errors[:50], "recomputed": res.recomputed}
''',
        '''@app.function(volumes={MNT: vol}, timeout=DAY, max_containers=1)
def validate(run_id: str, adapter_id: str, require_final: bool, require_package: bool) -> dict:
    from pathlib import Path
    from scripts.stopdff_v5 import checker
    vol.reload()
    run_id = _canonical_modal_run_id(run_id)
    run_root = Path(_p("runs", run_id))
    spec_path = run_root / "run_spec.json"
    if not spec_path.is_symlink() and spec_path.is_file():
        try:
            spec_manifest = checker.load_json(spec_path)
        except (OSError, UnicodeError, TypeError, ValueError):
            pass
        else:
            identity = (
                spec_manifest.get("identity")
                if isinstance(spec_manifest, dict)
                else None
            )
            if (
                isinstance(identity, dict)
                and identity.get("profile_variant") in {"smoke", "final"}
                and isinstance(spec_manifest.get("id"), str)
                and re.fullmatch(r"[0-9a-f]{64}", spec_manifest["id"])
            ):
                _canonical_modal_run_id(
                    run_id,
                    variant=identity["profile_variant"],
                    run_spec_id=spec_manifest["id"],
                )
    res = checker.validate_run(
        run_root,
        backend="modal",
        adapter_bundle=Path(_p("adapters", f"canonical_{adapter_id}")),
        require_final_profile=require_final,
        require_package=require_package,
    )
    return {
        "passed": res.passed,
        "errors": res.errors[:50],
        "recomputed": res.recomputed,
    }
''',
    )


def patch_adapter_builder() -> None:
    path = "scripts/stopdff_v5/adapter_build.py"
    replace_once(
        path,
        '''    _validate_scoring_question(question)
    qid = str(question["qid"])
    full_q = question["question"]
''',
        '''    _validate_scoring_question(question)
    qid = _record_qid(question)
    full_q = question["question"]
''',
    )

    replace_once(
        path,
        "def _score_questions_rows(\n",
        '''def _select_retained_questions(
    questions: list[dict[str, Any]],
    val_qids: set[str],
    test_qids: set[str],
) -> list[tuple[dict[str, Any], str]]:
    """Select validated MC rows using the repository's accepted qid aliases."""
    retained: list[tuple[dict[str, Any], str]] = []
    for question in sorted(questions, key=_record_qid):
        qid = _record_qid(question)
        if not qid:
            raise ValueError("MC dataset record lacks qid")
        if qid in val_qids:
            retained.append((question, "val"))
        if qid in test_qids:
            retained.append((question, "test"))
    return retained


def _score_questions_rows(
''',
    )

    replace_once(
        path,
        '''    retained_questions: list[tuple[dict[str, Any], str]] = []
    for q in sorted(questions, key=lambda x: str(x["qid"])):
        qid = str(q["qid"])
        if qid in val_qids:
            retained_questions.append((q, "val"))
        if qid in test_qids:
            retained_questions.append((q, "test"))

    scored_rows = _score_questions_rows(retained_questions, model)
''',
        '''    retained_questions = _select_retained_questions(
        questions,
        val_qids,
        test_qids,
    )

    scored_rows = _score_questions_rows(retained_questions, model)
''',
    )


def patch_checker() -> None:
    path = "scripts/stopdff_v5/checker.py"
    replace_once(
        path,
        '''    try:
        spec = load_json(spec_path)
''',
        '''    spec_path = Path(spec_path)
    path_issue = _canonical_path_issue(
        spec_path,
        expect_directory=False,
    )
    if path_issue == "missing":
        return CheckResult(passed=False, errors=["run spec is missing"])
    if path_issue is not None:
        return CheckResult(
            passed=False,
            errors=["run spec path must be a non-symlink regular file"],
        )
    try:
        spec = load_json(spec_path)
''',
    )


def patch_tests() -> None:
    append_once(
        "tests/test_pr30_control_repairs.py",
        "test_modal_run_id_contract_rejects_nested_and_mismatched_values",
        r'''


def test_modal_run_id_contract_rejects_nested_and_mismatched_values(
    monkeypatch,
):
    runner = _load_modal_runner(monkeypatch)
    spec_id = "a" * 64
    valid = f"smoke_modal_{spec_id[:12]}"
    assert runner._canonical_modal_run_id(
        valid,
        variant="smoke",
        run_spec_id=spec_id,
    ) == valid

    invalid = (
        f"nested/{valid}",
        f"{valid}/nested",
        f"final_modal_{spec_id[:12]}",
        "smoke_modal_not-hex",
        "smoke_modal_aaaaaaaaaaaa/..",
    )
    for value in invalid:
        with pytest.raises(ValueError):
            runner._canonical_modal_run_id(
                value,
                variant="smoke",
                run_spec_id=spec_id,
            )


@pytest.mark.parametrize("entrypoint", ["package", "validate"])
def test_modal_run_entrypoints_reject_nested_ids_before_path_use(
    monkeypatch,
    entrypoint,
):
    runner = _load_modal_runner(monkeypatch)

    def forbidden_path(*_parts):
        raise AssertionError("noncanonical run_id reached volume path derivation")

    monkeypatch.setattr(runner, "_p", forbidden_path)
    with pytest.raises(ValueError, match="run_id"):
        if entrypoint == "package":
            runner.package("nested/final_modal_aaaaaaaaaaaa")
        else:
            runner.validate(
                "nested/final_modal_aaaaaaaaaaaa",
                "b" * 64,
                True,
                True,
            )


@pytest.mark.parametrize(
    "mutation",
    ["root_symlink", "manifest_symlink", "manifest_directory"],
)
def test_fvi_manifest_path_rejects_noncanonical_cache_before_read(
    tmp_path,
    monkeypatch,
    mutation,
):
    runner = _load_modal_runner(monkeypatch)
    root = tmp_path / "fvi" / ("f" * 64)
    root.mkdir(parents=True)
    manifest = root / "fvi_study.json"
    manifest.write_text("{}", encoding="utf-8")
    assert runner._canonical_fvi_study_path(root) == manifest

    if mutation == "root_symlink":
        external = tmp_path / "external-fvi"
        root.rename(external)
        root.symlink_to(external, target_is_directory=True)
    elif mutation == "manifest_symlink":
        external = tmp_path / "external-fvi.json"
        manifest.rename(external)
        manifest.symlink_to(external)
    else:
        manifest.unlink()
        manifest.mkdir()

    with pytest.raises(ValueError, match="FVI cache"):
        runner._canonical_fvi_study_path(root)
''',
    )

    append_once(
        "tests/test_stopdff_v5_adapter_build.py",
        "test_adapter_retained_selection_accepts_qid_aliases",
        r'''


def test_adapter_retained_selection_accepts_qid_aliases():
    questions = [
        {"question_id": "val-b"},
        {"id": "test-a"},
        {"qid": "ignored"},
    ]
    retained = adapter_build._select_retained_questions(
        questions,
        {"val-b"},
        {"test-a"},
    )
    assert [
        (adapter_build._record_qid(question), split)
        for question, split in retained
    ] == [("test-a", "test"), ("val-b", "val")]


@pytest.mark.parametrize("qid_field", ["question_id", "id"])
def test_adapter_scoring_accepts_qid_aliases(qid_field):
    class FakeModel:
        def encode(
            self,
            values,
            *,
            batch_size,
            convert_to_numpy,
            show_progress_bar,
        ):
            assert batch_size == adapter_build._ENCODE_BATCH_SIZE
            assert convert_to_numpy is True
            assert show_progress_bar is False
            vectors = {
                "correct": [1.0, 0.0],
                "wrong": [0.0, 1.0],
                "prefix": [1.0, 0.0],
            }
            return np.asarray([vectors[value] for value in values], dtype=float)

    question = {
        qid_field: "alias-qid",
        "question": "prefix",
        "cumulative_prefixes": ["prefix"],
        "options": ["correct", "wrong"],
        "gold_index": 0,
        "answer_primary": "correct",
        "category": "Test",
    }
    rows = adapter_build._score_question_rows(
        question,
        FakeModel(),
        "val",
    )
    assert rows
    assert {row["item_id"] for row in rows} == {"alias-qid"}
''',
    )

    append_once(
        "tests/test_stopdff_v5_checker.py",
        "test_validate_spec_rejects_symlink_before_decode",
        r'''


def test_validate_spec_rejects_symlink_before_decode(tmp_path, monkeypatch):
    built = selftest.build_valid_package(tmp_path)
    canonical = built["run_root"] / "run_spec.json"
    external = tmp_path / "external-run-spec.json"
    external.write_bytes(canonical.read_bytes())
    selected = tmp_path / "selected-run-spec.json"
    selected.symlink_to(external)

    monkeypatch.setattr(
        checker,
        "load_json",
        lambda _path: pytest.fail("symlinked run spec was decoded"),
    )
    result = checker.validate_spec(
        selected,
        require_final_profile=False,
    )
    assert not result.passed
    assert result.errors == [
        "run spec path must be a non-symlink regular file"
    ]


def test_validate_spec_preserves_missing_path_diagnostic(tmp_path):
    result = checker.validate_spec(
        tmp_path / "missing-run-spec.json",
        require_final_profile=False,
    )
    assert not result.passed
    assert result.errors == ["run spec is missing"]
''',
    )


def main() -> None:
    patch_modal_runner()
    patch_adapter_builder()
    patch_checker()
    patch_tests()


if __name__ == "__main__":
    main()
