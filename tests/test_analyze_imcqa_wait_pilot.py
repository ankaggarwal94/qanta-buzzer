"""Synthetic counterexamples for the pilot analyzer; not experiment evidence."""
import copy
import hashlib
import json

import numpy as np
import pytest

from scripts import analyze_imcqa_wait_pilot as a


def fixture():
    questions = []
    jobs = []
    for split in a.SPLITS:
        qid = f"q-{split}"
        menus = [{"condition": condition, "menu_id": "frozen",
                  "gold_option_id": "A", "options": [{"id": label, "text": f"candidate-{label}"} for label in a.OPTIONS]}
                 for condition in a.MENUS]
        prefixes = [{"prefix_id": prefix, "fraction": (index + 1) / 5, "text": "clue " * (index + 1)}
                    for index, prefix in enumerate(a.PREFIX_IDS)]
        questions.append({"qid": qid, "group_id": f"g-{qid}", "split": split, "menus": menus, "prefixes": prefixes})
        for menu in menus:
            for index, prefix in enumerate(prefixes):
                for variant in a.VARIANTS:
                    mapping = a.canonical_mapping(int(variant == "rotation"))
                    options = [{"id": label, "text": f"candidate-{mapping[label]}"} for label in a.OPTIONS]
                    prompt = json.dumps(options)
                    jobs.append({"job_id": f"{qid}-{menu['condition']}-{index}-{variant}",
                                 "qid": qid, "group_id": f"g-{qid}", "split": split,
                                 "condition": menu["condition"], "menu_id": "frozen",
                                 "prefix_id": prefix["prefix_id"], "fraction": prefix["fraction"],
                                 "round": index + 1, "variant": variant,
                                 "rotation": int(variant == "rotation"), "canonical_option_ids": mapping,
                                 "options": options, "question_prefix": a.OMISSION_MARKER if variant == "questionless" else prefix["text"],
                                 "prompt": prompt, "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()})
    return {"questions": questions}, jobs


def scored(job, action="A"):
    labels = a.OPTIONS if job["variant"] == "forced" else a.ACTIONS
    logits = {label: float(label == action) for label in labels}
    return {**{key: job[key] for key in ("job_id", "qid", "group_id", "split", "condition", "menu_id", "prefix_id", "round", "variant", "prompt_sha256", "fraction", "rotation")},
            "raw_option_logits": logits, "conditional_option_probabilities": a.softmax(logits, labels),
            "top_option_id": action, "tied_top_option_ids": [action]}


def trajectory(actions="EEEEE", variant="wait", qid="q"):
    return [{"qid": qid, "group_id": f"g-{qid}", "split": "selection", "condition": "independent_pool", "variant": variant,
             "round": index + 1, "top_option_id": action, "canonical_choice": action if action != "E" else None,
             "actual_fraction": (index + 1) / 5, "correct": action == "A", "scored_input_token_ids": [1] * (index + 1)}
            for index, action in enumerate(actions)]


def test_complete_fixture_and_rotated_gold_join():
    dataset, jobs = fixture()
    joined = a.validate_jobs(jobs, dataset, n_per_split=1, n_diagnostic_per_split=1)
    rotated = next(job for job in joined if job["variant"] == "rotation")
    assert rotated["gold_option_id"] == "B"
    row = a.validate_rows([scored(rotated, "B")], [rotated])[0]
    assert row["correct"] is True
    assert row["canonical_choice"] == "A"


def test_wrong_rotated_gold_label_fails_instead_of_trusting_generator():
    dataset, jobs = fixture()
    job = next(job for job in jobs if job["variant"] == "rotation")
    job["gold_option_id"] = "A"
    with pytest.raises(ValueError, match="gold label"):
        a.join_job_gold(job, a.frozen_index(dataset))


def test_reverse_rotation_or_option_text_corruption_fails():
    dataset, jobs = fixture()
    job = next(job for job in jobs if job["variant"] == "rotation")
    job["options"] = [{"id": label, "text": f"candidate-{source}"} for label, source in zip("ABCD", "BCDA")]
    with pytest.raises(ValueError, match="option text/rotation"):
        a.join_job_gold(job, a.frozen_index(dataset))


def test_duplicate_scores_and_missing_factorial_cell_fail():
    dataset, jobs = fixture()
    joined = a.validate_jobs(jobs, dataset, n_per_split=1, n_diagnostic_per_split=1)
    with pytest.raises(ValueError, match="duplicate"):
        a.validate_rows([scored(joined[0]), scored(joined[0])], joined[:2])
    with pytest.raises(ValueError, match="incomplete factorial"):
        a.validate_jobs(jobs[:-1], dataset, n_per_split=1, n_diagnostic_per_split=1)


def test_qid_cross_split_and_test_inclusion_fail():
    dataset, jobs = fixture()
    duplicate = copy.deepcopy(dataset["questions"][0])
    duplicate["split"] = "selection"
    with pytest.raises(ValueError, match="duplicate frozen qid"):
        a.frozen_index({"questions": dataset["questions"] + [duplicate]})
    jobs[0]["split"] = "test"
    with pytest.raises(ValueError, match="split"):
        a.validate_jobs(jobs, dataset, n_per_split=1, n_diagnostic_per_split=1)


def test_terminal_wait_means_pass_zero_not_a_sixth_round():
    record = a.policy_record(trajectory())
    assert record["reward"] == 0
    assert record["round"] is None
    assert record["observed_round"] == 5
    assert record["terminal_pass"] is True
    assert record["counterfactual_uncached_input_tokens"] == 15
    summary = a.summarize_policy([record], np.zeros((8, 1), dtype=int))
    assert summary["risk"] is None
    assert summary["bootstrap"]["risk"]["ci95"] is None
    assert summary["bootstrap"]["risk"]["defined_resamples"] == 0


def test_wrong_commit_terminates_before_later_correct_answer():
    record = a.policy_record(trajectory("EBAAA"))
    assert record["round"] == 2
    assert record["wrong"] is True
    assert record["reward"] == -1
    assert record["counterfactual_uncached_input_tokens"] == 3


def test_final_correct_reward_and_fixed_round_reads_only_one_prompt():
    record = a.policy_record(trajectory("EEEEA"))
    assert record["reward"] == .2
    final = a.policy_record(trajectory("AAAAA", variant="forced"), fixed_round=5)
    assert final["reward"] == .2
    assert final["counterfactual_uncached_input_tokens"] == 5


def test_argmax_and_softmax_are_recomputed_not_trusted():
    row = {"raw_option_logits": {label: float(label == "E") for label in a.ACTIONS},
           "conditional_option_probabilities": {label: .2 for label in a.ACTIONS},
           "top_option_id": "E", "tied_top_option_ids": ["E"]}
    with pytest.raises(ValueError, match="softmax"):
        a.validate_score(row, a.ACTIONS)
    row["conditional_option_probabilities"] = a.softmax(row["raw_option_logits"], a.ACTIONS)
    row["top_option_id"] = "A"
    with pytest.raises(ValueError, match="argmax"):
        a.validate_score(row, a.ACTIONS)


def test_semantic_rotation_is_invariant_under_correct_relabelling():
    p = {"A": .5, "B": .1, "C": .1, "D": .1, "E": .2}
    original = {"qid": "q", "split": "selection", "condition": "independent_pool", "round": 1,
                "conditional_option_probabilities": p, "canonical_choice": "A"}
    rotated = {**original, "conditional_option_probabilities": {"A": .1, "B": .5, "C": .1, "D": .1, "E": .2},
               "canonical_option_ids": a.canonical_mapping(1)}
    result = a.rotation_effect(original, rotated)
    assert result["semantic_action_changed"] is False
    assert result["semantic_total_variation"] == 0


def test_forced_comparison_conditions_on_not_wait_stably():
    common = {"qid": "q", "split": "selection", "condition": "independent_pool", "round": 1, "gold_option_id": "A"}
    wait_logits = {"A": 0., "B": -1., "C": -2., "D": -3., "E": 1000.}
    wait = {**common, "raw_option_logits": wait_logits, "conditional_option_probabilities": a.softmax(wait_logits, a.ACTIONS)}
    forced = {**common, "raw_option_logits": {label: wait_logits[label] for label in a.OPTIONS}}
    effect = a.prompt_effect(wait, forced)
    assert effect["conditional_total_variation"] == 0
    assert effect["primary_conditional_gold_probability"] > .6


def test_paired_bootstrap_preserves_menus_and_requires_matching_questions():
    left = [a.policy_record(trajectory("AAAAA", qid=qid)) for qid in ("q", "r")]
    right = copy.deepcopy(left)
    indices = np.array([[0, 1], [0, 0], [1, 1]])
    result = a.paired_difference(left, right, indices)
    assert result["differences"]["mean_reward"]["ci95"] == [0, 0]
    right[0]["qid"] = "other"
    with pytest.raises(ValueError, match="paired"):
        a.paired_difference(left, right, indices)


def test_numeric_gate_rejects_small_argmax_flip_even_within_logit_tolerance():
    config = {"raw_logit_atol": .001, "raw_logit_rtol": .00001, "max_action_probability_absolute_difference": .001}
    left = [{"logits": [1., 1.00001, 0., 0., 0.]}]
    right = [{"logits": [1.00001, 1., 0., 0., 0.]}]
    with pytest.raises(ValueError, match="action agreement"):
        a.compare_numeric(left, right, [{"allowed_actions": "ABCDE"}], config)


def test_end_to_end_synthetic_report_keeps_splits_policies_and_paired_controls():
    dataset, jobs = fixture()
    joined = a.validate_jobs(jobs, dataset, n_per_split=1, n_diagnostic_per_split=1)
    rows = []
    for job in joined:
        action = (job["gold_option_id"] if job["variant"] == "forced" or
                  (job["split"] == "calibration" and job["round"] >= 3) else "E")
        row = a.validate_rows([scored(job, action)], [job])[0]
        labels = a.OPTIONS if job["variant"] == "forced" else a.ACTIONS
        logits = {**row["raw_option_logits"]}
        logits.setdefault("E", -1.)
        row.update(allowed_actions="".join(labels), raw_action_logits=logits,
                   action_probabilities=row["conditional_option_probabilities"], chosen_action=action,
                   legal_action_vocabulary_mass=.9, option_token_ids={label: index for index, label in enumerate(a.ACTIONS)},
                   unconstrained_top_token_id=a.ACTIONS.index(action), scored_input_token_ids=[1, 2])
        rows.append(row)
    report = a.analyze_outputs({"qwen3b": rows, "qwen7b": copy.deepcopy(rows)}, samples=5)
    primary = [row for row in report["policy_summaries"] if row["model"] == "qwen3b" and row["variant"] == "wait"]
    assert len(primary) == 4
    for summary in primary:
        if summary["split"] == "selection":
            assert summary["n_terminal_pass"] == 1
            assert summary["risk"] is None
        else:
            assert summary["mean_reward"] == .6
            assert summary["mean_round_when_committed"] == 3
    comparisons = [row for row in report["paired_comparisons"] if row["comparison"].startswith("paired_models")]
    assert len(comparisons) == 8
    assert all(row["differences"]["mean_reward"]["mean"] == 0 for row in comparisons)


def provenance_fixture():
    cached = {"model": "fixture", "revision": "pinned", "model_files_sha256": {"model.safetensors": "a" * 64}}
    metadata = {**cached, "versions": {**a.PINNED_VERSIONS}}
    promotion = {"schema_version": "acl-paired-dtype-promotion-v1", "all_checks_passed": True,
                 "sampled_values_preserved_exactly": True, "all_floating_tensors_fp32": True,
                 "sample_rule": "first two and last two flattened values of every named tensor",
                 "original": {"parameter:a": {"dtype": "torch.bfloat16", "shape": [3], "sample": [1., 2., 2., 3.]},
                              "buffer:b": {"dtype": "torch.float32", "shape": [1], "sample": [1., 1.]}},
                 "promoted_dtypes": {"parameter:a": "torch.float32", "buffer:b": "torch.float32"}}
    return metadata, promotion, cached


@pytest.mark.parametrize("corruption", ("version", "cache", "dtype", "sample", "assertion"))
def test_provenance_rejects_corrupted_stack_cache_or_dtype_evidence(corruption):
    metadata, promotion, cached = provenance_fixture()
    assert a.validate_provenance(metadata, promotion, cached)["pinned_stack_passed"]
    if corruption == "version":
        metadata["versions"]["torch"] = "0.0.0"
    elif corruption == "cache":
        metadata["model_files_sha256"] = {"model.safetensors": "b" * 64}
    elif corruption == "dtype":
        promotion["promoted_dtypes"]["parameter:a"] = "torch.bfloat16"
    elif corruption == "sample":
        promotion["original"]["parameter:a"]["sample"] = [float("nan")] * 4
    else:
        promotion["sampled_values_preserved_exactly"] = False
    with pytest.raises(ValueError):
        a.validate_provenance(metadata, promotion, cached)


def test_source_hash_validation_detects_changed_scorer_or_config(tmp_path):
    for name in a.SOURCE_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    metadata = {"source_files_sha256": {name: a.sha256(tmp_path / name) for name in a.SOURCE_FILES}}
    assert a.validate_source_metadata(metadata, tmp_path) == metadata["source_files_sha256"]
    (tmp_path / "configs/imcqa_wait_pilot.json").write_text("changed")
    with pytest.raises(ValueError, match="source hashes"):
        a.validate_source_metadata(metadata, tmp_path)


@pytest.mark.parametrize("field,value", (("reward", .7), ("allowed_actions", "ABCD")))
def test_primary_public_job_must_keep_reward_and_wait_action(field, value):
    dataset, jobs = fixture()
    jobs[0][field] = value
    with pytest.raises(ValueError, match="reward|action set"):
        a.join_job_gold(jobs[0], a.frozen_index(dataset))
