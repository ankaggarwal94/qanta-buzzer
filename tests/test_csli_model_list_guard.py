"""Empty --models guard for scripts.compute_csli (Iter1 IN-06).

The prior implementation accepted ``--models ""`` and survived all
the way to ``per_question_csli /= len(model_list)`` where it raised
``ZeroDivisionError`` on a numpy array -- failing far enough
downstream that the operator could not tell their CLI typo was the
root cause.

Iter1 IN-06 adds an early validation at the top of main(): empty
model_list (from ``--models ""`` OR ``--models ","`` OR any whitespace-
only variant) prints a clear error and returns exit code 2. The
documented script exit codes are:

  0 - success
  1 - runtime error
  2 - argument error

so exit 2 matches the "argparse-rejected input" convention even
though the validation lives below argparse (split-and-filter cannot
be expressed as a pure argparse type).

The test invokes ``compute_csli.main([...])`` directly and asserts
the return code is 2 for several empty-equivalent inputs.
"""

from __future__ import annotations

import pytest

from scripts.compute_csli import main as compute_csli_main


# ---------------------------------------------------------------------------
# Iter1 IN-06 guard
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "models_arg",
    [
        "",
        ",",
        " ",
        " , , ",
        ",,,",
    ],
)
def test_compute_csli_rejects_empty_models_list(
    models_arg: str,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``--models <empty-equivalent>`` exits with code 2 and clear error.

    The argument-parsing layer cannot reject these directly because
    they ARE valid strings -- the emptiness only emerges after the
    split-on-comma + whitespace-strip filter. The guard at the top
    of main() catches them before any side effects.

    Tests the specific empty-equivalent forms a tired operator might
    produce: empty string, comma-only, whitespace-only, and
    whitespace-around-commas.
    """
    rc = compute_csli_main(["--models", models_arg, "--dry-run"])
    assert rc == 2, (
        f"Iter1 IN-06 regression: --models={models_arg!r} returned "
        f"exit code {rc}, expected 2 (argument error per the script's "
        "documented exit-code convention)."
    )
    captured = capsys.readouterr()
    assert "must specify at least one model" in captured.err, (
        "Iter1 IN-06: error message did not surface the actionable "
        "phrase 'must specify at least one model'. The operator "
        "should not have to guess that their --models typo was the "
        "root cause. Captured stderr:\n" + captured.err
    )


def test_compute_csli_accepts_single_model(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Happy path: a single comma-separated model still works.

    Pins the contract that the guard does not falsely reject
    legitimate single-model invocations. Uses --dry-run so the test
    does not load any actual data / models.
    """
    rc = compute_csli_main(["--models", "tfidf", "--dry-run"])
    assert rc == 0, (
        f"Iter1 IN-06 regression: --models=tfidf --dry-run returned "
        f"exit code {rc}, expected 0. The guard is over-aggressive."
    )
    # capsys consumed to keep test fixture happy; substring check is
    # incidental here.
    capsys.readouterr()


def test_compute_csli_accepts_multiple_models(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Happy path: comma-separated multi-model list survives the filter.

    Confirms whitespace-tolerant parsing (the filter strips spaces
    around commas). Uses --dry-run for speed.
    """
    rc = compute_csli_main(
        ["--models", "tfidf, sbert , t5-small", "--dry-run"]
    )
    assert rc == 0
    capsys.readouterr()
