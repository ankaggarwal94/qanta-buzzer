"""Regressions for content-bound reference profile and ranking caches."""

from __future__ import annotations

from qb_data.answer_profiles import AnswerProfileBuilder
from qb_data.data_loader import TossupQuestion
from qb_data.mc_builder import MCBuilder


def _question(qid: str, answer: str, text: str) -> TossupQuestion:
    tokens = text.split()
    return TossupQuestion(
        qid=qid,
        question=text,
        tokens=tokens,
        answer_primary=answer,
        clean_answers=[answer],
        run_indices=list(range(len(tokens))),
        human_buzz_positions=None,
        category="History",
        cumulative_prefixes=[
            " ".join(tokens[: index + 1]) for index in range(len(tokens))
        ],
    )


def test_refit_changed_text_and_answer_rebuilds_profiles() -> None:
    builder = AnswerProfileBuilder()
    builder.fit(
        [
            _question("r0", "Alpha", "old alpha text"),
            _question("r1", "Beta", "old beta text"),
        ]
    )
    assert builder.profile_for_answer("Alpha") == "old alpha text"

    builder.fit(
        [
            _question("r0", "Alpha", "new alpha text"),
            _question("r1", "Gamma", "new gamma text"),
        ]
    )

    assert builder.profile_for_answer("Alpha") == "new alpha text"
    assert builder.profile_for_answer("Gamma") == "new gamma text"
    assert builder.profile_for_answer("Beta") == "Beta"


def test_refit_reordering_changes_truncated_profile_order() -> None:
    first = _question("r0", "Alpha", "one two")
    second = _question("r1", "Alpha", "three four")
    builder = AnswerProfileBuilder(max_tokens_per_profile=2)

    builder.fit([first, second])
    assert builder.profile_for_answer("Alpha") == "one two"
    builder.fit([second, first])

    assert builder.profile_for_answer("Alpha") == "three four"


def test_refit_duplicate_addition_and_removal_changes_profile() -> None:
    first = _question("r0", "Alpha", "alpha first")
    builder = AnswerProfileBuilder()

    builder.fit([first])
    assert builder.profile_for_answer("Alpha") == "alpha first"
    builder.fit([first, first])
    assert builder.profile_for_answer("Alpha") == "alpha first alpha first"
    builder.fit([first])

    assert builder.profile_for_answer("Alpha") == "alpha first"


class RecordingMCBuilder(MCBuilder):
    def __init__(self) -> None:
        super().__init__(
            K=2,
            strategy="tfidf_profile",
            random_seed=13,
        )
        self.profile_calls: list[dict[str, str]] = []

    def _compute_rankings(
        self,
        answers: list[str],
        answer_profiles: dict[str, str],
        answer_to_category: dict[str, str],
    ) -> dict[str, list[str]]:
        del answer_to_category
        self.profile_calls.append(dict(answer_profiles))
        return {
            answer: [other for other in answers if other != answer]
            for answer in answers
        }


def test_mc_builder_cache_key_preserves_reference_order() -> None:

    alpha_first = _question("r0", "Alpha", "alpha first")
    alpha_second = _question("r1", "Alpha", "alpha second")
    beta = _question("r2", "Beta", "beta reference")
    target = [_question("t0", "Alpha", "unrelated target clue")]
    original = [alpha_first, alpha_second, beta]
    reordered = [alpha_second, alpha_first, beta]
    ordered_builder = RecordingMCBuilder()

    ordered_builder.build(
        target,
        AnswerProfileBuilder().fit(original),
        reference_questions=original,
    )
    assert ordered_builder.profile_calls[-1]["Alpha"] == (
        "alpha first alpha second"
    )

    # An identical ordered corpus remains cacheable.
    ordered_builder.build(
        target,
        AnswerProfileBuilder().fit(list(original)),
        reference_questions=list(original),
    )
    assert len(ordered_builder.profile_calls) == 1

    ordered_builder.build(
        target,
        AnswerProfileBuilder().fit(reordered),
        reference_questions=reordered,
    )
    assert ordered_builder.profile_calls[-1]["Alpha"] == (
        "alpha second alpha first"
    )


def test_mc_builder_cache_key_preserves_reference_multiplicity() -> None:
    alpha_first = _question("r0", "Alpha", "alpha first")
    alpha_second = _question("r1", "Alpha", "alpha second")
    beta = _question("r2", "Beta", "beta reference")
    target = [_question("t0", "Alpha", "unrelated target clue")]
    original = [alpha_first, alpha_second, beta]
    duplicated = [alpha_first, alpha_second, alpha_first, beta]
    multiplicity_builder = RecordingMCBuilder()
    multiplicity_builder.build(
        target,
        AnswerProfileBuilder().fit(original),
        reference_questions=original,
    )
    multiplicity_builder.build(
        target,
        AnswerProfileBuilder().fit(duplicated),
        reference_questions=duplicated,
    )
    assert len(multiplicity_builder.profile_calls) == 2
    assert multiplicity_builder.profile_calls[-1]["Alpha"] == (
        "alpha first alpha second alpha first"
    )
