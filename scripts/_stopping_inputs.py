"""Read belief and prefix-SFT experiment inputs before legacy numeric coercion."""

from __future__ import annotations

import json
from pathlib import Path

from qb_data.mc_builder import MCQuestion
from scripts._common import iter_split_questions, mc_question_from_dict


def parse_stopping_questions(raw: bytes, path: Path) -> list[MCQuestion]:
    """Deserialize one snapshot of the project's list or wrapped question format."""
    questions = []
    for row in iter_split_questions(json.loads(raw), source_path=path):
        gold = row["gold_index"]
        if isinstance(gold, bool) or (isinstance(gold, float) and not gold.is_integer()):
            raise ValueError(f"{row['qid']}: gold_index must be an integer slot")
        questions.append(mc_question_from_dict(row))
    return questions
