"""CPU guards for paired prompt identity, fixed shapes, and numerical evidence."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from contextlib import nullcontext

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import acl_paired_prompt_scoring as scoring


def job(index=0, condition="independent_pool"):
    options = [{"id": label, "text": name} for label, name in zip("ABCD", ["Mercury", "Venus", "Earth", "Mars"])]
    prompt = scoring.build_choice_control_prompt(options)
    return {"job_id": f"j{index}", "qid": f"q{index}", "group_id": f"g{index}",
        "split": "test", "format": "mc", "condition": condition, "menu_id": f"m{index}",
        "options": options, "prompt": prompt, "prompt_sha256": scoring.base.sha(prompt.encode())}


def context(tokens):
    return {"scored_input_token_ids": tokens, "option_token_ids": dict(zip("ABCD", (12, 13, 14, 15)))}


def test_forced_prompt_changes_only_exact_abstention_tail_and_preserves_menu():
    original = job()
    snapshot = deepcopy(original)
    forced = scoring.forced_prompt(original)
    prefix = original["prompt"][:-len(scoring.ORIGINAL_TAIL)]
    assert forced == prefix + scoring.FORCED_TAIL
    assert '"abstain"' not in forced and "null" not in forced
    assert '"answer" as status' in forced
    assert original == snapshot
    for option in original["options"]:
        assert f'{option["id"]}. {option["text"]}' in forced


@pytest.mark.parametrize("mutation", ["extra", "hash", "options"])
def test_forced_prompt_rejects_noncanonical_or_unbound_original(mutation):
    value = job()
    if mutation == "extra":
        value["prompt"] += " Answer B."
        value["prompt_sha256"] = scoring.base.sha(value["prompt"].encode())
    elif mutation == "hash":
        value["prompt_sha256"] = "0" * 64
    else:
        value["options"][0]["text"] = "Neptune"
    with pytest.raises(ValueError, match="canonical"):
        scoring.forced_prompt(value)


def test_exact_templates_reconstruct_both_actual_prompts():
    manifest = scoring.prompt_manifest()
    value = job()
    menu = "\n".join(f'{x["id"]}. {x["text"]}' for x in value["options"])
    for condition, expected in (("original", value["prompt"]), ("forced", scoring.forced_prompt(value))):
        template = manifest["templates"][condition]
        assert template.replace("{menu}", menu) == expected
        assert manifest["template_sha256"][condition] == scoring.base.sha(template.encode())
    assert manifest["assistant_prefix"] == '{"answer":"'


def test_pairs_preserve_input_order_and_bind_original_and_actual_hashes():
    inputs = [job(0), job(1)]
    result = scoring.derived_jobs(inputs)
    assert [(x["job_index"], x["prompt_condition"]) for x in result] == [(0,"original"),(0,"forced"),(1,"original"),(1,"forced")]
    assert [x["score_index"] for x in result] == list(range(4))
    assert len({x["score_id"] for x in result}) == 4
    assert result[1]["original_prompt_sha256"] == inputs[0]["prompt_sha256"]
    assert result[1]["prompt_sha256"] == scoring.base.sha(result[1]["prompt"].encode())
    assert result[1]["prompt_sha256"] != result[0]["prompt_sha256"]
    assert all(x["options"] == inputs[x["job_index"]]["options"] for x in result)


def test_pair_coverage_rejects_missing_swapped_duplicate_and_cross_prompt_hashes():
    expected = scoring.derived_jobs([job(0), job(1)])
    scoring.validate_coverage(expected, expected, complete=True)
    scoring.validate_coverage(expected, expected[:2], complete=False)
    for invalid in [expected[:-1], expected[::-1], [expected[0], expected[0], *expected[2:]]]:
        with pytest.raises(ValueError):
            scoring.validate_coverage(expected, invalid, complete=True)
    corrupted = deepcopy(expected)
    corrupted[1]["prompt_sha256"] = corrupted[0]["prompt_sha256"]
    with pytest.raises(ValueError, match="hash"):
        scoring.validate_coverage(expected, corrupted, complete=True)


def test_diagnostic_selection_covers_short_and_long_both_conditions_without_gold():
    jobs = [job(i, "independent_pool" if i < 5 else "same_category_pool") for i in range(10)]
    contexts = [context(list(range(80 + i // 2))) for i in range(20)]
    result = scoring.select_diagnostic_menus(jobs, contexts)
    assert result["menu_indices"] == [0, 1, 3, 4, 5, 6, 8, 9]
    assert result["score_indices"] == [2 * i + j for i in result["menu_indices"] for j in (0, 1)]
    assert result["selection_uses_gold"] is False
    assert result == scoring.select_diagnostic_menus(jobs, contexts)


def test_fixed_layout_masks_left_padding_keeps_positions_and_duplicates_only_fillers():
    contexts = [context([8, 9]), context([5, 6, 7])]
    original = deepcopy(contexts)
    layout = scoring.batch_layout(contexts, pad_token_id=0, padded_width=4, batch_size=4)
    assert layout["input_ids"] == [[0,0,8,9], [0,5,6,7], [0,5,6,7], [0,5,6,7]]
    assert layout["attention_mask"] == [[0,0,1,1], [0,1,1,1], [0,1,1,1], [0,1,1,1]]
    assert layout["position_ids"] == [[0,0,0,1], [0,0,1,2], [0,0,1,2], [0,0,1,2]]
    assert layout["real_rows"] == layout["filler_rows"] == 2
    assert contexts == original
    with pytest.raises(ValueError, match="exceeds"):
        scoring.batch_layout(contexts, pad_token_id=0, padded_width=2, batch_size=4)


class Tensor:
    def __init__(self, values, dtype="torch.float32"):
        self.values = np.asarray(values)
        self.dtype = dtype

    @property
    def shape(self):
        return self.values.shape

    def __getitem__(self, index):
        return Tensor(self.values[index], self.dtype)

    def gather(self, dim, indices):
        return Tensor(np.take_along_axis(self.values, indices.values, axis=dim), self.dtype)

    def float(self):
        return Tensor(self.values.astype(float), "torch.float32")

    def cpu(self):
        return self

    def tolist(self):
        return self.values.tolist()

    def detach(self):
        return self

    def reshape(self, width):
        return Tensor(self.values.reshape(width), self.dtype)

    def is_floating_point(self):
        return self.dtype in {"torch.bfloat16", "torch.float32"}


class Torch:
    long = "torch.int64"
    cuda = SimpleNamespace(synchronize=lambda: None)

    @staticmethod
    def tensor(values, *, device, dtype):
        assert device == "cuda:0"
        return Tensor(values, dtype)

    @staticmethod
    def inference_mode():
        return nullcontext()


class RecordingModel:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        assert kwargs["use_cache"] is False
        assert kwargs["logits_to_keep"] == 1
        assert kwargs["return_dict"] is True
        ids, mask, positions = [kwargs[x].values for x in ("input_ids", "attention_mask", "position_ids")]
        self.calls.append((ids.copy(), mask.copy(), positions.copy()))
        logits = np.zeros((len(ids), 1, 20))
        for index, (tokens, active, pos) in enumerate(zip(ids, mask, positions)):
            assert active[-1] == 1
            logits[index, 0, 12:16] = [sum(tokens * active), sum(active), sum(pos), active[-1]]
        return SimpleNamespace(logits=Tensor(logits))


def test_forward_extracts_four_scores_discards_fillers_and_keeps_exact_shape():
    model = RecordingModel()
    contexts = [context([8, 9]), context([5, 6, 7])]
    values = scoring.forward(Torch(), model, SimpleNamespace(pad_token_id=0), contexts,
                             padded_width=5, batch_size=32)
    assert values == [[17.0, 2.0, 1.0, 1.0], [18.0, 3.0, 3.0, 1.0]]
    assert model.calls[0][0].shape == (32,5)
    singleton = scoring.forward(Torch(), model, SimpleNamespace(pad_token_id=0), [contexts[0]])
    assert singleton == [values[0]]
    assert model.calls[1][0].shape == (1,2)


def test_numeric_gates_reject_material_drift_replay_difference_and_bad_permutation():
    batched = [[1.0,2.0,3.0,4.0], [4.0,3.0,2.0,1.0]]
    single = [[1.0001,2.0,3.0,4.0], [4.0,3.0,2.0,1.0]]
    report = scoring.validate_diagnostics(batched, single, batched, batched[::-1], [1,0])
    assert report["all_gates_passed"] and report["exact_replay_passed"]
    drift = deepcopy(single)
    drift[0][0] += .01
    with pytest.raises(ValueError, match="tolerance"):
        scoring.validate_diagnostics(batched, drift, batched, batched[::-1], [1,0])
    with pytest.raises(ValueError, match="replay"):
        scoring.validate_diagnostics(batched, single, single, batched[::-1], [1,0])
    with pytest.raises(ValueError, match="permutation"):
        scoring.validate_diagnostics(batched, single, batched, batched[::-1], [0,0])


def test_fp32_promotion_preserves_values_and_rejects_dtype_fallback():
    weight = Tensor([1.25, 2.5], "torch.bfloat16")
    rope = Tensor([1.0, .0001234], "torch.float32")
    model = SimpleNamespace(named_parameters=lambda: [("weight", weight)], named_buffers=lambda: [("rope", rope)])
    before = scoring.base.snapshot_tensor_state(model)
    with pytest.raises(ValueError, match="FP32"):
        scoring.validate_promotion(model, before)
    weight.dtype = "torch.float32"
    report = scoring.validate_promotion(model, before)
    assert report["sampled_values_preserved_exactly"]
    assert report["original"]["parameter:weight"]["dtype"] == "torch.bfloat16"
    assert report["promoted_dtypes"]["buffer:rope"] == "torch.float32"
    weight.values[0] = 4.0
    with pytest.raises(ValueError, match="values changed"):
        scoring.validate_promotion(model, before)


def test_failure_leaves_receipt_and_never_overwrites_output(tmp_path):
    output = tmp_path / "new"
    receipt = scoring.run_scoring("qwen3b", tmp_path / "missing", tmp_path, output)
    assert receipt["status"] == "failed" and receipt["completed_rows"] == 0
    assert json.loads((output / "receipt.json").read_text()) == receipt
    with pytest.raises(FileExistsError):
        scoring.run_scoring("qwen3b", tmp_path / "missing", tmp_path, output)


@pytest.mark.parametrize("kwargs", [{"batch_size": 16}, {"batch_size": True},
                                    {"max_seconds": 1401}, {"max_seconds": float("inf")}])
def test_invalid_settings_rejected_before_allocating_output(tmp_path, kwargs):
    with pytest.raises(ValueError):
        scoring.run_scoring("qwen3b", tmp_path / "missing", tmp_path, tmp_path / "new", **kwargs)
    assert not (tmp_path / "new").exists()


def paired_contexts():
    # Unequal common-prefix lengths and unequal suffix lengths exercise both
    # the prefix's left pad and the masked gap before a shorter suffix.
    return [context([2,3,4,7,8]), context([2,3,4,9]),
            context([5,6,1,2]), context([5,6,8,9,3])]


def test_cache_plan_uses_exact_lcp_and_preserves_both_suffixes():
    contexts = paired_contexts()
    plan = scoring.cache_plan(contexts)
    assert plan["n_pairs"] == 2
    assert plan["prefix_width"] == 3 and plan["suffix_width"] == 3
    assert plan["full_width"] == 5
    assert plan["total_shared_prefix_tokens"] == 5
    assert plan["total_full_input_tokens"] == 18
    assert plan["total_suffix_tokens"] == 8
    assert plan["padded_cached_token_positions_per_pair"] == 9
    with pytest.raises(ValueError, match="complete"):
        scoring.cache_plan(contexts[:-1])
    with pytest.raises(ValueError, match="nonempty"):
        scoring.split_context_pair(contexts[0], contexts[0])


def test_cached_layout_separates_physical_cache_positions_from_logical_positions():
    contexts = paired_contexts()
    layout = scoring.cached_batch_layout(contexts, pad_token_id=0,
                                         prefix_width=4, suffix_width=4, batch_size=6)
    assert layout["prefix_input_ids"] == [[0,2,3,4], [0,0,5,6], [0,0,5,6]]
    assert layout["prefix_cache_position"] == [0,1,2,3]
    assert layout["suffix_cache_position"] == [4,5,6,7]
    assert layout["suffix_input_ids"][1] == [0,0,0,9]
    assert layout["full_attention_mask"][1] == [0,1,1,1,0,0,0,1]
    assert layout["suffix_position_ids"][1] == [0,0,0,3]
    assert layout["suffix_input_ids"][-2:] == layout["suffix_input_ids"][2:4]
    for index in range(layout["real_rows"]):
        physical = layout["prefix_input_ids"][index//2] + layout["suffix_input_ids"][index]
        mask = layout["full_attention_mask"][index]
        reconstructed = [token for token, keep in zip(physical, mask) if keep]
        assert reconstructed == contexts[index]["scored_input_token_ids"]
        positions = layout["prefix_position_ids"][index//2] + layout["suffix_position_ids"][index]
        assert [pos for pos, keep in zip(positions, mask) if keep] == list(range(len(reconstructed)))
    with pytest.raises(ValueError, match="complete pairs"):
        scoring.cached_batch_layout(contexts[:-1], pad_token_id=0,prefix_width=4,suffix_width=4,batch_size=6)


class ToyCache:
    def __init__(self, tokens, positions):
        self.tokens, self.positions = tokens.copy(), positions.copy()

    def get_seq_length(self):
        return self.tokens.shape[1]

    def batch_repeat_interleave(self, repeats):
        self.tokens = np.repeat(self.tokens, repeats, axis=0)
        self.positions = np.repeat(self.positions, repeats, axis=0)


class CausalToyModel:
    """Independent causal state model checks branch order, gaps, and state reuse."""
    def __init__(self):
        self.created_cache_ids = []
        self.calls = []

    def __call__(self, **kwargs):
        tokens = kwargs["input_ids"].values
        positions = kwargs["position_ids"].values
        active = kwargs["attention_mask"].values
        cache = kwargs.get("past_key_values")
        old_width = cache.get_seq_length() if cache is not None else 0
        if kwargs["use_cache"]:
            physical_positions = kwargs["cache_position"].values
            assert list(physical_positions) == list(range(old_width,old_width+tokens.shape[1]))
            if cache is None:
                cache = ToyCache(tokens, positions)
                self.created_cache_ids.append(cache)
            else:
                assert cache.tokens.shape[0] == tokens.shape[0]
                cache.tokens = np.concatenate([cache.tokens,tokens],axis=1)
                cache.positions = np.concatenate([cache.positions,positions],axis=1)
            total_tokens, total_positions = cache.tokens, cache.positions
        else:
            total_tokens, total_positions = tokens, positions
        assert active.shape == total_tokens.shape
        self.calls.append((tokens.shape,active.shape,kwargs["use_cache"]))
        logits = np.zeros((len(tokens),1,20))
        for index in range(len(tokens)):
            valid_positions = total_positions[index][active[index].astype(bool)]
            assert list(valid_positions) == list(range(len(valid_positions)))
            logits[index,0,12:16] = [sum(total_tokens[index]*active[index]),sum(active[index]),
                                    sum(total_positions[index]*active[index]),active[index][-1]]
        return SimpleNamespace(logits=Tensor(logits),past_key_values=cache)


def test_cached_forward_matches_native_and_never_shares_cache_between_calls():
    contexts, model = paired_contexts(), CausalToyModel()
    tokenizer = SimpleNamespace(pad_token_id=0)
    native = scoring.forward(Torch(),model,tokenizer,contexts,padded_width=6,batch_size=6)
    cached = scoring.cached_forward(Torch(),model,tokenizer,contexts,
                                    prefix_width=4,suffix_width=4,batch_size=6)
    replay = scoring.cached_forward(Torch(),model,tokenizer,contexts,
                                    prefix_width=4,suffix_width=4,batch_size=6)
    assert cached == native == replay
    assert len(cached) == 4
    assert model.calls[1] == ((3,4),(3,4),True)
    assert model.calls[2] == ((6,4),(6,8),True)
    assert len(model.created_cache_ids) == 2
    assert model.created_cache_ids[0] is not model.created_cache_ids[1]
    permutation = scoring.reverse_pair_permutation(len(contexts))
    reordered = scoring.cached_forward(Torch(),model,tokenizer,[contexts[i] for i in permutation],
                                        prefix_width=4,suffix_width=4,batch_size=6)
    gates = scoring.validate_cached_diagnostics(cached,native,replay,reordered,permutation,native)
    assert gates["exact_cached_replay_passed"]
    assert permutation == [2,3,0,1]
    with pytest.raises(ValueError,match="complete"):
        scoring.validate_cached_diagnostics(cached,native,replay,reordered,[3,2,1,0],native)


def test_cached_projection_uses_worse_of_mean_margin_and_slow_batch_projection():
    ordinary = scoring.cached_budget_projection(256,20000,2.0,250.0,[1.0,1.0],128)
    assert ordinary["safety_factor"] == 1.2
    assert ordinary["projected_remaining_seconds"] == pytest.approx((19744/256)*2*1.2)
    assert ordinary["proceed"] is True
    slow = scoring.cached_budget_projection(256,20000,3.0,350.0,[1.0,2.0],128)
    assert slow["p95_projected_remaining_seconds"] == 310.0
    assert slow["projected_remaining_seconds"] == 310.0
    assert slow["proceed"] is True
    assert scoring.cached_budget_projection(256,20000,3.0,320.0,[1.0,2.0],128)["proceed"] is False


def test_cache_mode_requires_explicit_flag_and_matching_batch_size(tmp_path):
    with pytest.raises(ValueError, match="mode-specific"):
        scoring.run_scoring("qwen3b",tmp_path/"missing",tmp_path,tmp_path/"no",batch_size=128)
    with pytest.raises(ValueError, match="mode-specific"):
        scoring.run_scoring("qwen3b",tmp_path/"missing",tmp_path,tmp_path/"no",shared_prefix_cache=True)
    receipt = scoring.run_scoring("qwen3b",tmp_path/"missing",tmp_path,tmp_path/"yes",
                                  shared_prefix_cache=True,batch_size=128)
    assert receipt["status"] == "failed"  # Input absence, after mode validation.


def test_cached_direct_single_gate_prevents_transitive_double_tolerance():
    single = [[1.0,2.0,3.0,4.0], [4.0,3.0,2.0,1.0]]
    native = deepcopy(single)
    native[0][0] += .0009
    cached = deepcopy(single)
    cached[0][0] += .0018
    scoring.base.validate_fp32_agreement(cached,native)
    scoring.base.validate_fp32_agreement(native,single)
    with pytest.raises(ValueError,match="tolerance"):
        scoring.validate_cached_diagnostics(cached,native,cached,cached,[0,1],single)
