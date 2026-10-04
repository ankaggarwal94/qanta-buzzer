"""Action-label controls, numerical decision guards, and deterministic pairing."""
from copy import deepcopy
import math

import pytest

from scripts import imcqa_protocol_scoring as scorer
from scripts import imcqa_protocol_design as design


def job(mapping=None, allowed="ABCDE", **extra):
    return {"option_source_ids": mapping or dict(zip("ABCD", "ABCD")),
            "allowed_actions": allowed, **extra}


def test_swapped_wait_is_excluded_from_conditional_answers_and_identity_restored():
    mapping = design.mapping(0, "A")
    result = scorer.action_statistics([100, 1, 2, 3, 4], "ABCDE", mapping)
    assert result["chosen_action"] == "A"
    assert set(result["conditional_answer_probabilities"]) == set("BCDE")
    assert result["canonical_answer_probabilities"]["A"] == result["conditional_answer_probabilities"]["E"]
    assert max(result["canonical_answer_probabilities"], key=result["canonical_answer_probabilities"].get) == "A"
    assert math.fsum(result["conditional_answer_probabilities"].values()) == pytest.approx(1)


def test_forced_normalization_ignores_illegal_label_even_when_it_dominates():
    result = scorer.action_statistics([1, 2, 3, 4, 100], "ABCD", design.mapping(0))
    assert result["chosen_action"] == "D"
    assert "E" not in result["action_probabilities"]


def test_display_label_tie_rule_is_explicit_after_relabeling():
    result = scorer.action_statistics([0, 1, 1, 1, 1], "ABCDE", design.mapping(0, "A"))
    assert result["chosen_action"] == "B"
    assert result["tied_top_actions"] == list("BCDE")


@pytest.mark.parametrize("logits,labels,mapping", [
    ([0, 0, 0, 0], "ABCDE", design.mapping(0)),
    ([0, 0, 0, 0, float("nan")], "ABCDE", design.mapping(0)),
    ([0, 0, 0, 0, True], "ABCDE", design.mapping(0)),
    ([0]*5, "ABCDA", design.mapping(0)),
    ([0]*5, "DCBA", design.mapping(0)),
    ([0]*5, "ABCD", design.mapping(0, "A")),
    ([0]*5, "ABCDE", dict(zip("ABCD", "AAAA"))),
])
def test_statistics_fail_closed(logits, labels, mapping):
    with pytest.raises(ValueError):
        scorer.action_statistics(logits, labels, mapping)


def test_numerical_guard_checks_candidate_argmax_even_when_wait_wins():
    before = [{"logits": [.00001, 0, -1, -2, 10]}]
    after = [{"logits": [0, .00001, -1, -2, 10]}]
    with pytest.raises(ValueError, match="candidate argmax"):
        scorer.numeric_agreement(before, after, [job()])


def test_numerical_guard_checks_swapped_wait_action_and_all_five_logits():
    before = [{"logits": [1, 0, 0, 0, 0]}]
    same = scorer.numeric_agreement(before, before, [job(design.mapping(0, "A"))])
    assert same["candidate_argmax_changes"] == same["argmax_changes"] == 0
    with pytest.raises(ValueError, match="logits"):
        scorer.numeric_agreement(before, [{"logits": [1.1, 0, 0, 0, 0]}], [job(design.mapping(0, "A"))])


def context(ids):
    return {"scored_input_token_ids": ids, "option_token_ids": dict(zip("ABCDE", range(11, 16)))}


def test_pairing_is_complete_deterministic_and_long_pairs_first():
    contexts = [context([1, 3, 5]), context([1, 3, 7]), context([1, 2, 4, 5]), context([1, 2, 4, 7])]
    order = scorer.paired_order(contexts)
    assert order == [2, 3, 0, 1]
    with pytest.raises(ValueError):
        scorer.paired_order(contexts[:3])
    with pytest.raises(ValueError, match="different suffixes"):
        scorer.paired_order([contexts[0], deepcopy(contexts[0])])


def test_overlap_sentinels_cover_reused_strata_rounds_menus_and_extreme_lengths():
    reused = []
    for menu in scorer.CONDITIONS:
        for round_number in range(1, 6):
            for arm, rotation in (("forced", 0), ("wait", 0), ("wait", 1)):
                item = job(score_id=f"{menu}-{round_number}-{arm}-{rotation}", condition=menu,
                           round=round_number, arm=arm, rotation=rotation, block="factorial")
                reused.append((item, context([1]*(10+len(reused))), {}))
    selected = scorer.reuse_sentinels(reused)
    assert 10 <= len(selected) <= 16
    assert {value[0]["round"] for value in selected} == set(range(1, 6))
    assert {value[0]["condition"] for value in selected} == set(scorer.CONDITIONS)
    assert {(value[0]["arm"], value[0]["rotation"]) for value in selected} == {("forced", 0), ("wait", 0), ("wait", 1)}
    assert reused[0] in selected and reused[-1] in selected
