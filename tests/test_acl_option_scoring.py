"""CPU correctness guards; no model-stack import, network, or GPU allocation."""
from copy import deepcopy
import json
import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import acl_option_scoring as scoring
from scripts.jane_gpu_backend import build_choice_control_prompt


class CharacterTokenizer:
    """A transparent tokenizer makes boundary bugs independently observable."""
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        return "<user>" + messages[0]["content"] + "<assistant>"

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [ord(c) for c in text]}

    def decode(self, ids, **kwargs):
        return "".join(map(chr, ids))


def job(index=0):
    options = [{"id": label, "text": text} for label, text in zip("ABCD", ["Mercury", "Venus", "Earth", "Mars"])]
    prompt = build_choice_control_prompt(options)
    return {"job_id": f"j{index}", "qid": f"q{index}", "group_id": f"g{index}",
            "split": "calibration", "format": "mc", "condition": "independent",
            "menu_id": f"m{index}", "options": options, "prompt": prompt,
            "prompt_sha256": scoring.sha(prompt.encode())}


def test_softmax_is_stable_offset_invariant_and_normalized():
    first = scoring.option_statistics([1000.0, 999.0, 998.0, 997.0])
    second = scoring.option_statistics([0.0, -1.0, -2.0, -3.0])
    assert first["conditional_option_probabilities"] == second["conditional_option_probabilities"]
    assert math.isclose(sum(first["conditional_option_probabilities"].values()), 1.0)
    assert first["top_option_id"] == "A"
    assert first["logit_margin"] == 1.0
    assert first["probability_margin"] > 0
    assert first["conditional_option_probabilities"]["A"] == pytest.approx(1 / sum(math.exp(-i) for i in range(4)))


@pytest.mark.parametrize("values", [[1, 2, 3], [1, 2, 3, float("nan")],
                                      [float("inf"), 2, 3, 4], [True, 2, 3, 4]])
def test_nonfinite_or_invalid_logit_vectors_are_rejected(values):
    with pytest.raises(ValueError, match="finite numeric"):
        scoring.option_statistics(values)


def test_ties_remain_visible_and_abcd_tie_break_is_deterministic():
    stats = scoring.option_statistics([1.0, 3.0, 3.0, 0.0])
    assert stats["top_option_id"] == "B"
    assert stats["tied_top_option_ids"] == ["B", "C"]
    assert stats["logit_margin"] == stats["probability_margin"] == 0


def test_context_preserves_original_prompt_and_proves_four_token_extensions():
    tokenizer, value = CharacterTokenizer(), job()
    context = scoring.prepare_context(tokenizer, value)
    original = tokenizer.apply_chat_template([{"content": value["prompt"]}], tokenize=False, add_generation_prompt=True)
    scored = original + scoring.ASSISTANT_PREFIX
    assert context["input_token_ids"] == [ord(c) for c in original]
    assert context["scored_input_token_ids"] == [ord(c) for c in scored]
    assert context["scored_context_sha256"] == scoring.sha(scored.encode())
    assert context["option_token_ids"] == {c: ord(c) for c in "ABCD"}
    assert "correct" not in context


def test_retokenization_at_option_boundary_is_rejected():
    class MergingTokenizer(CharacterTokenizer):
        def __call__(self, text, **kwargs):
            output = super().__call__(text, **kwargs)
            if text.endswith(scoring.ASSISTANT_PREFIX + "A"):
                output["input_ids"][-2:] = [999]
            return output
    with pytest.raises(ValueError, match="single-token extension"):
        scoring.prepare_context(MergingTokenizer(), job())


def test_token_aliases_cannot_silently_score_same_token_twice():
    class AliasedTokenizer(CharacterTokenizer):
        def __call__(self, text, **kwargs):
            output = super().__call__(text, **kwargs)
            if text.endswith(scoring.ASSISTANT_PREFIX + "B"):
                output["input_ids"][-1] = ord("A")
            return output
    with pytest.raises(ValueError, match="decode"):
        scoring.prepare_context(AliasedTokenizer(), job())


def test_benchmark_gate_includes_safety_factor_and_shutdown_reserve():
    estimate = scoring.budget_projection(256, 10000, 10.0, 600.0)
    assert estimate["projected_remaining_seconds"] == pytest.approx(380.625)
    assert estimate["proceed"] is True
    assert scoring.budget_projection(256, 10000, 10.0, 580.0)["proceed"] is False
    assert scoring.budget_projection(256, 10000, 10.0, -1.0)["proceed"] is False
    with pytest.raises(ValueError):
        scoring.budget_projection(256, 10000, 0.0, 600)


def test_coverage_accepts_only_exact_ordered_prefix_and_complete_final():
    jobs = [job(0), job(1), job(2)]
    rows = [{"job_index": i, "job_id": x["job_id"], "prompt_sha256": x["prompt_sha256"]} for i, x in enumerate(jobs)]
    scoring.validate_score_coverage(jobs, rows)
    scoring.validate_score_coverage(jobs, rows[:2], require_complete=False)
    for invalid in [rows[:2], rows[::-1], [rows[0], rows[0], rows[2]]]:
        with pytest.raises(ValueError):
            scoring.validate_score_coverage(jobs, invalid)
    corrupt = deepcopy(rows)
    corrupt[1]["prompt_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        scoring.validate_score_coverage(jobs, corrupt)


def test_fp32_sanity_gate_allows_numerical_ties_but_rejects_material_drift():
    result = scoring.validate_fp32_agreement([[4, 3, 2, 1]], [[4.0005, 3, 2, 1]])
    assert result["numeric_gate_passed"] is True
    assert result["maximum_absolute_logit_difference"] == pytest.approx(0.0005)
    # An exact tie can become a tiny reversed preference without a logic error.
    near_tie = scoring.validate_fp32_agreement([[4, 4, 2, 1]], [[3.9999, 4, 2, 1]])
    assert near_tie["changed_top_option_rows"] == [0]
    assert near_tie["exact_top_equality_required"] is False
    with pytest.raises(ValueError, match="predeclared numeric tolerance"):
        scoring.validate_fp32_agreement([[4, 3, 2, 1]], [[4.01, 3, 2, 1]])
    with pytest.raises(ValueError, match="finite numeric"):
        scoring.validate_fp32_agreement([[4, 3, 2, 1]], [[float("nan"), 3, 2, 1]])


def test_bf16_sensitivity_is_reported_without_disappearing_into_a_loosened_gate():
    report = scoring.comparison_diagnostics([[1, 2, 3, 4]], [[1, 2, 4, 3]])
    assert report["changed_top_option_rows"] == [0]
    assert report["maximum_absolute_logit_difference"] == 1
    assert report["maximum_option_probability_difference"] > 0
    common_offset = scoring.comparison_diagnostics([[10, 9, 8, 7]], [[9, 8, 7, 6]])
    assert common_offset["maximum_absolute_logit_difference"] == 1
    assert common_offset["maximum_centered_logit_difference"] == 0
    assert common_offset["maximum_option_probability_difference"] == 0


class FakeTensor:
    """Minimal tensor API for testing per-tensor dtype restoration independently."""
    def __init__(self, values, dtype):
        self.values, self.dtype = list(values), dtype
        self.shape = (len(values),)

    @property
    def data(self):
        return self

    @data.setter
    def data(self, tensor):
        self.values, self.dtype, self.shape = tensor.values, tensor.dtype, tensor.shape

    def detach(self):
        return self

    def reshape(self, _):
        return self

    def __getitem__(self, item):
        return FakeTensor(self.values[item], self.dtype)

    def cpu(self):
        return self

    def tolist(self):
        return self.values

    def to(self, *, dtype):
        return FakeTensor(self.values, dtype)


class FakeModel:
    def __init__(self):
        self.weight = FakeTensor([1.25, 2.5, 4.0], "bfloat16")
        self.rope = FakeTensor([1.0, 0.00001234567], "float32")
        self.integer = FakeTensor([3], "int64")

    def named_parameters(self):
        return [("weight", self.weight)]

    def named_buffers(self):
        return [("rope.inv_freq", self.rope), ("integer_counter", self.integer)]


def test_per_tensor_restoration_preserves_fp32_rope_and_integer_buffers():
    model = FakeModel()
    before = scoring.snapshot_tensor_state(model)
    model.weight.data = model.weight.to(dtype="float32")
    result = scoring.restore_tensor_state(model, before)
    assert model.weight.dtype == "bfloat16"
    assert model.rope.dtype == "float32"
    assert model.integer.dtype == "int64"
    assert result["all_dtypes_restored"] is True
    assert result["all_sampled_values_restored_exactly"] is True
    assert result["original_dtypes"]["buffer:rope.inv_freq"] == "float32"


def test_dtype_restoration_fails_if_fp32_diagnostic_altered_values_or_schema():
    model = FakeModel()
    before = scoring.snapshot_tensor_state(model)
    model.weight.data = model.weight.to(dtype="float32")
    model.weight.values[0] = 99.0
    with pytest.raises(ValueError, match="sampled values"):
        scoring.restore_tensor_state(model, before)
    model = FakeModel()
    before = scoring.snapshot_tensor_state(model)
    before.pop("buffer:rope.inv_freq")
    with pytest.raises(ValueError, match="names changed"):
        scoring.restore_tensor_state(model, before)


def test_public_package_hash_and_validator_are_both_enforced(tmp_path, monkeypatch):
    path = tmp_path / "public.json"
    package = {"schema_version": "jane-choice-controls-v1", "evidence_scope": "scientific", "jobs": [job()]}
    raw = scoring.canonical(package)
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="hash"):
        scoring.load_jobs(path)
    monkeypatch.setattr(scoring, "PUBLIC_SHA256", scoring.sha(raw))
    monkeypatch.setattr(scoring, "EXPECTED_JOBS", 1)
    assert scoring.load_jobs(path) == package["jobs"]
    package["jobs"][0]["correct_answer"] = "A"
    raw = scoring.canonical(package)
    path.write_bytes(raw)
    monkeypatch.setattr(scoring, "PUBLIC_SHA256", scoring.sha(raw))
    with pytest.raises(ValueError, match="unexpected choices-only public keys"):
        scoring.load_jobs(path)


def test_existing_results_are_never_overwritten(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(FileExistsError):
        scoring.run_scoring("qwen3b", tmp_path / "missing", tmp_path, output)


def test_input_failure_is_durably_receipted_without_importing_gpu_stack(tmp_path):
    output = tmp_path / "new"
    receipt = scoring.run_scoring("qwen3b", tmp_path / "missing", tmp_path, output)
    assert receipt["status"] == "failed"
    assert receipt["completed_rows"] == 0
    assert receipt["scores_sha256"] is None
    assert json.loads((output / "receipt.json").read_text()) == receipt


@pytest.mark.parametrize("kwargs", [{"max_seconds": 1700}, {"max_seconds": float("nan")},
                                    {"batch_size": True}, {"batch_size": 0}])
def test_unbounded_or_invalid_execution_is_rejected_before_output(tmp_path, kwargs):
    with pytest.raises(ValueError):
        scoring.run_scoring("qwen3b", tmp_path / "missing", tmp_path, tmp_path / "new", **kwargs)
    assert not (tmp_path / "new").exists()
