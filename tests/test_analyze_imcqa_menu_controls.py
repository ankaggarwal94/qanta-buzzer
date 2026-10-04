import pytest
from scripts.analyze_imcqa_menu_controls import accuracy_interval, summarize


def test_zero_error_interval_is_not_degenerate():
    low, high = accuracy_interval(7, 7)
    assert 0 < low < 1 and high == 1


def test_duplicate_questions_rejected():
    with pytest.raises(ValueError, match="one nonempty"):
        summarize([{"qid":"a"}, {"qid":"a"}])


def test_gold_is_not_inferred_from_predicted_position():
    r=summarize([{"qid":"a", "gold":"D", "top":"A", "correct":False,
                  "gold_probability":.1,"max_probability":.7}])
    assert r["n_correct"] == 0 and r["gold_position_counts"] == {"D":1}
    assert r["uniform_random_choice_accuracy"] == .25
