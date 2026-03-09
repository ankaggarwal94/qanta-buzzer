"""Shared pytest fixtures for test suites.

Provides reusable test data for environment, likelihood, features,
factory, and agent test suites. All fixtures create minimal but complete
data structures that satisfy the interfaces expected by the codebase modules.

Fixtures
--------
sample_mc_question
    A single MCQuestion with 4 options (gold_index=0), 6 clue steps,
    and pre-computed cumulative prefixes. Suitable for environment and
    feature extraction tests.

sample_config
    A minimal config dict matching the YAML structure expected by
    ``make_env_from_config`` and ``build_likelihood_from_config``.
    Uses "simple" reward mode for predictable test outcomes.

sample_corpus
    A list of 10 short text strings about US presidents and historical
    events. Suitable for fitting TF-IDF vectorizers in tests.

sample_tfidf_env
    A TossupMCEnv with TF-IDF likelihood and 3 sample MCQuestions.
    Fast to construct, suitable for agent and PPO tests.
"""

from __future__ import annotations

import pytest

from qb_data.mc_builder import MCQuestion


@pytest.fixture
def sample_mc_question() -> MCQuestion:
    """Return a minimal MCQuestion for testing.

    The question is about the first US president with 4 answer options.
    Gold answer is "George Washington" at index 0. Six clue steps are
    defined via run_indices with pre-computed cumulative prefixes.

    Returns
    -------
    MCQuestion
        A complete MCQuestion suitable for environment testing.
    """
    tokens = [
        "Who", "was", "the", "first", "president",
        "of", "the", "United", "States", "?",
    ]
    run_indices = [0, 2, 4, 6, 8, 9]
    cumulative_prefixes = [
        "Who",
        "Who was the",
        "Who was the first president",
        "Who was the first president of the",
        "Who was the first president of the United States",
        "Who was the first president of the United States ?",
    ]
    return MCQuestion(
        qid="test_q1",
        question="Who was the first president of the United States?",
        tokens=tokens,
        answer_primary="George Washington",
        clean_answers=["George Washington", "Washington"],
        run_indices=run_indices,
        human_buzz_positions=[],
        category="History",
        cumulative_prefixes=cumulative_prefixes,
        options=[
            "George Washington",
            "Thomas Jefferson",
            "John Adams",
            "Benjamin Franklin",
        ],
        gold_index=0,
        option_profiles=[
            "George Washington first president commander revolutionary war continental army",
            "Thomas Jefferson third president declaration independence Virginia",
            "John Adams second president Massachusetts diplomat",
            "Benjamin Franklin inventor diplomat Philadelphia printing press",
        ],
        option_answer_primary=[
            "George Washington",
            "Thomas Jefferson",
            "John Adams",
            "Benjamin Franklin",
        ],
        distractor_strategy="test",
    )


@pytest.fixture
def sample_config() -> dict:
    """Return a minimal config dict for factory tests.

    Matches the YAML structure expected by ``make_env_from_config`` and
    ``build_likelihood_from_config``. Uses "simple" reward mode and
    "from_scratch" belief mode for predictable test outcomes.

    Returns
    -------
    dict
        Config dict with data, environment, and likelihood sections.
    """
    return {
        "data": {"K": 4},
        "environment": {
            "reward": "simple",
            "wait_penalty": 0.0,
            "buzz_correct": 1.0,
            "buzz_incorrect": -1.0,
            "belief_mode": "from_scratch",
        },
        "likelihood": {
            "model": "sbert",
            "beta": 5.0,
        },
    }


@pytest.fixture
def sample_corpus() -> list[str]:
    """Return a list of 10 short text strings for TF-IDF fitting.

    Topics cover US presidents and major historical events, providing
    sufficient vocabulary variety for TF-IDF vectorizer tests.

    Returns
    -------
    list[str]
        Ten text strings suitable for corpus fitting.
    """
    return [
        "George Washington was the first president of the United States",
        "Thomas Jefferson wrote the Declaration of Independence",
        "John Adams served as the second president after Washington",
        "Benjamin Franklin was an inventor and diplomat in Philadelphia",
        "Abraham Lincoln freed the slaves during the Civil War",
        "Alexander Hamilton established the national banking system",
        "James Madison authored the Bill of Rights and Constitution",
        "Andrew Jackson was a military hero and populist president",
        "The American Revolution established independence from Britain",
        "The Constitution created a federal system of government",
    ]


@pytest.fixture(scope="module")
def sample_t5_model():
    """Return a T5Likelihood model for testing.

    Uses t5-small (60M params) for fast test execution. Scoped to module
    level so the model is loaded once per test file, not per test function.

    Returns
    -------
    T5Likelihood
        A T5 likelihood model suitable for testing semantic scoring.

    Notes
    -----
    This fixture may take 5-10 seconds on first run to download the model
    from HuggingFace. Subsequent runs use cached weights.
    """
    from models.likelihoods import T5Likelihood

    return T5Likelihood(model_name="t5-small")


@pytest.fixture
def sample_tfidf_env(sample_mc_question: MCQuestion) -> "TossupMCEnv":
    """Return a TossupMCEnv with TF-IDF likelihood and 3 sample questions.

    Creates a lightweight environment suitable for PPOBuzzer and agent
    tests. Uses TF-IDF likelihood for fast execution (< 1ms per score).
    Three copies of the sample question are used to provide enough data
    for environment sampling.

    Returns
    -------
    TossupMCEnv
        A configured environment with simple reward mode.
    """
    from models.likelihoods import TfIdfLikelihood
    from qb_env.tossup_env import TossupMCEnv

    corpus = sample_mc_question.option_profiles[:]
    model = TfIdfLikelihood(corpus_texts=corpus)

    # Use 3 copies for variety in sampling
    questions = [sample_mc_question] * 3
    return TossupMCEnv(
        questions=questions,
        likelihood_model=model,
        K=4,
        reward_mode="simple",
        wait_penalty=0.0,
        buzz_correct=1.0,
        buzz_incorrect=-1.0,
        belief_mode="from_scratch",
        beta=5.0,
    )
