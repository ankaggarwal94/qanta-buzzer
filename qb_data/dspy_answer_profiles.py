"""Optional DSPy-based answer profile augmentation.

Generates richer answer profiles using an LM when the ``dspy`` extra is
installed and enabled.  The extractive ``AnswerProfileBuilder`` remains
the default and fallback — this module only augments, never replaces.

This module requires the ``dspy`` optional extra.
"""

from __future__ import annotations

from typing import Any


def build_dspy_profiles(
    answers: list[str],
    existing_profiles: dict[str, str],
    dspy_config: dict[str, Any],
    max_answers: int = 100,
) -> dict[str, str]:
    """Generate LM-augmented answer profiles via DSPy.

    Leave-one-out discipline depends on the *caller* providing
    ``existing_profiles`` that already exclude the current question
    (as ``AnswerProfileBuilder.profile_for_answer(answer, exclude_qid)``
    does).  This function itself does not receive per-question exclusion
    context — it augments whatever profiles it is given.

    Parameters
    ----------
    answers : list[str]
        Answer strings to generate profiles for.
    existing_profiles : dict[str, str]
        Extractive profiles from ``AnswerProfileBuilder``.
    dspy_config : dict
        DSPy configuration section from YAML.
    max_answers : int
        Cap on number of answers to augment.

    Returns
    -------
    dict[str, str]
        Mapping from answer to augmented profile text.  Falls back to
        the extractive profile when augmentation fails.
    """
    try:
        import dspy
    except ImportError as exc:
        raise ImportError(
            "DSPy answer profile augmentation requires the dspy package. "
            "Install with: pip install -e '.[dspy]'"
        ) from exc

    lm_name = dspy_config.get("model", "openai/gpt-4o-mini")
    lm = dspy.LM(lm_name)
    dspy.configure(lm=lm)

    class AnswerProfileSignature(dspy.Signature):
        """Generate a rich factual profile for a quiz bowl answer."""
        answer: str = dspy.InputField(desc="the answer entity")
        existing_profile: str = dspy.InputField(desc="extractive profile from question corpus")
        augmented_profile: str = dspy.OutputField(desc="enriched factual profile suitable for quiz bowl scoring")

    generator = dspy.Predict(AnswerProfileSignature)

    import logging

    logger = logging.getLogger(__name__)

    result: dict[str, str] = {}
    n_augmented = 0
    n_fallback = 0
    for answer in answers[:max_answers]:
        existing = existing_profiles.get(answer, "")
        try:
            pred = generator(answer=answer, existing_profile=existing)
            result[answer] = pred.augmented_profile
            n_augmented += 1
        except Exception as exc:
            logger.warning("DSPy augmentation failed for %r: %s", answer, exc)
            result[answer] = existing
            n_fallback += 1

    for answer in answers[max_answers:]:
        result[answer] = existing_profiles.get(answer, "")

    if n_fallback:
        logger.info(
            "DSPy profiles: %d augmented, %d fell back to extractive",
            n_augmented, n_fallback,
        )

    return result
