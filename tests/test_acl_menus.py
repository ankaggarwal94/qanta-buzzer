import itertools
import math

import pytest

from qb_data.acl_menus import MenuOracle, ReservoirRow, select_indices


REGISTRY = {
    "version": "synthetic-v1",
    "reviewer": "test fixture",
    "classes": {"Gold": ["Old Gold"], "Red": [], "Blue": [], "Green": [],
                "Yellow": [], "Purple": [], "Teal": [], "Leak": []},
}
ROWS = [ReservoirRow(answer, profile, category, str(i)) for i, (answer, profile, category)
        in enumerate([("Red", "scarlet apple", "fruit"), ("Blue", "blueberry", "fruit"),
                      ("Green", "green apple", "fruit"), ("Yellow", "lemon", "fruit"),
                      ("Purple", "plum", "color"), ("Teal", "teal sea", "color")])]
ARGS = dict(gold="Old Gold", group_id="target", category="fruit", prefix="apple", k=3)


def test_shrinking_pool_bijection():
    outcomes = [select_indices(4, 3, digits) for digits in itertools.product(
        range(4), range(3), range(2))]
    assert set(outcomes) == set(itertools.permutations(range(4), 3))
    assert len(outcomes) == len(set(outcomes))
    for invalid in ((4, 0, 0), (0, 3, 0), (0, 0, -1), (0, 0), (False, 0, 0)):
        with pytest.raises(ValueError):
            select_indices(4, 3, invalid)


@pytest.mark.parametrize("strategy", ["uniform_global", "uniform_category", "tfidf_easy",
                                      "tfidf_hard", "mixed", "prefix_tfidf"])
def test_group_isolation_order_invariance_and_provenance(strategy):
    target_row = ReservoirRow("Leak", "apple", "fruit", "target")
    menu = MenuOracle([target_row, *ROWS], aliases=REGISTRY).sample(strategy=strategy, **ARGS)
    clean = MenuOracle(reversed(ROWS), aliases=REGISTRY).sample(strategy=strategy, **ARGS)
    assert menu == clean
    assert menu.options[menu.gold_index] == "Gold"
    assert menu.class_ids[menu.gold_index] == "Gold"
    assert "Leak" not in menu.class_ids
    assert len(set(menu.class_ids)) == 3
    assert menu.log_probability <= 0
    if strategy == "uniform_category":
        assert math.exp(menu.log_probability) == pytest.approx(1 / (4 * 3 * 3))


def test_closed_world_and_split_and_short_pool():
    with pytest.raises(ValueError, match="train"):
        MenuOracle([ReservoirRow("Red", "x", "fruit", "other", split="val")], aliases=REGISTRY)
    oracle = MenuOracle(ROWS, aliases=REGISTRY)
    with pytest.raises(ValueError, match="unreviewed"):
        oracle.sample(**{**ARGS, "gold": "unreviewed"})
    with pytest.raises(ValueError, match="insufficient"):
        oracle.sample(**{**ARGS, "k": 10})
    with pytest.raises(ValueError):
        oracle.sample(**ARGS, strategy="baseline", explicit=["Gold", "Old Gold", "Blue"])


def test_exact_legacy_order_and_training_source_ids_for_unseen_gold():
    oracle = MenuOracle(ROWS, aliases=REGISTRY)
    options = ["Red", "Old Gold", "Blue"]
    menu = oracle.sample(strategy="baseline", explicit=options,
                         explicit_source_ids=[["0"], [], ["1"]], **ARGS)
    assert menu.options == tuple(options)
    assert menu.gold_index == 1
    assert menu.source_ids == (("0",), (), ("1",))
    assert menu.log_probability == 0
    with pytest.raises(ValueError, match="source"):
        oracle.sample(strategy="baseline", explicit=options,
                      explicit_source_ids=[["1"], [], ["0"]], **ARGS)


def test_review_registry_is_closed_and_punctuation_is_not_silently_dropped():
    classes = {"C": [], "C++": [], "C#": [], "Beyoncé": ["Beyonce"]}
    oracle = MenuOracle([], aliases={**REGISTRY, "classes": classes})
    assert oracle.class_id("C++") != oracle.class_id("C")
    assert oracle.class_id("beyonce\u0301") == "Beyoncé"
    with pytest.raises(ValueError, match="ambiguous"):
        MenuOracle([], aliases={**REGISTRY, "classes": {"First": ["shared"], "Second": ["SHARED"]}})
    with pytest.raises(ValueError, match="reviewer"):
        MenuOracle([], aliases={**REGISTRY, "reviewer": ""})


def test_exact_rank_bands_prefix_support_and_ordered_laws():
    oracle = MenuOracle(ROWS, aliases=REGISTRY)
    for seed in range(12):
        hard = oracle.sample(strategy="tfidf_hard", seed=seed, **ARGS)
        easy = oracle.sample(strategy="tfidf_easy", seed=seed, **ARGS)
        top = oracle.sample(strategy="prefix_tfidf", seed=seed, **ARGS)
        assert set(hard.class_ids) <= {"Red", "Green", "Blue", "Gold"}
        assert set(easy.class_ids) <= {"Purple", "Yellow", "Teal", "Gold"}
        assert set(top.class_ids) == {"Red", "Green", "Gold"}
        assert math.exp(top.log_probability) == pytest.approx(1 / 6)
        assert math.exp(hard.log_probability) == pytest.approx(1 / 18)
        assert math.exp(easy.log_probability) == pytest.approx(1 / 18)


def test_mixed_prefers_novel_categories_on_alternating_extremes():
    oracle = MenuOracle(ROWS, aliases=REGISTRY)
    result = oracle.sample(strategy="mixed", **ARGS)
    assert set(result.class_ids) == {"Gold", "Green", "Teal"}
    assert math.exp(result.log_probability) == pytest.approx(1 / 6)


def test_sbert_receives_only_observed_query_and_eligible_class_profiles():
    calls = []
    class ScoreSpy:
        def score(self, prefix, profiles):
            calls.append((prefix, profiles))
            return [float("teal" in p) for p in profiles]

    oracle = MenuOracle([*ROWS, ReservoirRow("Gold", "FUTURE", "fruit", "target")],
                        aliases=REGISTRY, encoder=ScoreSpy())
    for strategy in ("sbert_hard", "sbert_easy", "prefix_sbert"):
        menu = oracle.sample(strategy=strategy, **ARGS)
        if strategy != "sbert_easy":
            if strategy == "prefix_sbert":
                assert "Teal" in menu.class_ids
        else:
            assert "Teal" not in menu.class_ids
        prefix, profiles = calls[-1]
        assert prefix == "apple"
        assert "FUTURE" not in profiles
        assert all(label not in profiles for label in ("Gold", "Old Gold"))
    assert len(calls) == 3


def test_nonfinite_scores_and_missing_sbert_fail():
    class InvalidScorer:
        def score(self, prefix, profiles):
            return [float("nan")] * len(profiles)
    with pytest.raises(ValueError, match="invalid retrieval"):
        MenuOracle(ROWS, aliases=REGISTRY, encoder=InvalidScorer()).sample(
            strategy="sbert_easy", **ARGS)
    with pytest.raises(ValueError, match="configured SBERT"):
        MenuOracle(ROWS, aliases=REGISTRY).sample(strategy="sbert_easy", **ARGS)


def test_registry_hash_bound_to_realized_menu():
    original = MenuOracle(ROWS, aliases=REGISTRY).sample(**ARGS)
    changed = MenuOracle(ROWS, aliases={**REGISTRY, "version": "v2"}).sample(**ARGS)
    assert changed.options == original.options
    assert changed.equivalence_digest != original.equivalence_digest
    assert changed.digest != original.digest


def test_optional_decoder_reaches_real_generation_and_disjoint_reference_donors():
    class AuditedDecoder:
        def __init__(self):
            self.calls = []

        def draw(self, n, digits):
            self.calls.append((n, digits))
            return select_indices(n, len(digits), digits)

        def insert(self, xs, gold, pos):
            return [*xs[:pos], gold, *xs[pos:]]

    decoder = AuditedDecoder()
    refs = [ReservoirRow(r.answer, r.profile, r.category, r.group_id, "reference") for r in ROWS]
    for mode in ("uniform_global", "tfidf_hard", "mixed", "prefix_tfidf"):
        observed = MenuOracle(refs, aliases=REGISTRY, decoder=decoder).sample(strategy=mode, **ARGS)
        expected = MenuOracle(ROWS, aliases=REGISTRY).sample(strategy=mode, **ARGS)
        assert observed == expected
    assert len(decoder.calls) == 4
