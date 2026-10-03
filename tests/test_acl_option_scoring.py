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


def test_batch_single_comparison_rejects_changed_rank_or_excessive_drift():
    result = scoring.validate_batch_agreement([[4, 3, 2, 1]], [[4.0625, 3, 2, 1]])
    assert result["maximum_absolute_difference"] == 0.0625
    with pytest.raises(ValueError, match="top-option"):
        scoring.validate_batch_agreement([[1, 2, 3, 4]], [[1, 2, 4, 3]])
    with pytest.raises(ValueError, match="tolerance"):
        scoring.validate_batch_agreement([[4, 3, 2, 1]], [[5, 4, 3, 2]])
    with pytest.raises(ValueError, match="top-option"):
        scoring.validate_batch_agreement([[4, 4, 2, 1]], [[4, 3.99, 2, 1]])


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
