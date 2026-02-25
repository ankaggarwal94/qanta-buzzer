"""
Data structures and loaders for quiz bowl questions.
"""

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Any, Dict

from qb_data.text_utils import normalize_answer


@dataclass
class TossupQuestion:
    """
    A quiz bowl tossup question with incremental clues.

    Attributes
    ----------
    qid : str
        Unique question identifier
    question : str
        Full question text (all clues concatenated)
    tokens : List[str]
        Tokenized question split on whitespace
    answer_primary : str
        Primary answer text
    clean_answers : List[str]
        List of acceptable answer variants
    run_indices : List[int]
        Token indices where clues end (for incremental reveal)
    human_buzz_positions : Optional[List[Tuple[int, int]]]
        Human buzzer positions as (position, count) tuples
    category : str
        Question category (e.g., "History", "Literature")
    cumulative_prefixes : List[str]
        Precomputed text prefixes at each run_index
    """
    qid: str
    question: str
    tokens: List[str]
    answer_primary: str
    clean_answers: List[str]
    run_indices: List[int]
    human_buzz_positions: Optional[List[Tuple[int, int]]]
    category: str
    cumulative_prefixes: List[str]


def _parse_clues_to_tokens(clues: List[str]) -> Tuple[List[str], List[int]]:
    """
    Convert list of clues to tokens and run indices.

    Parameters
    ----------
    clues : List[str]
        List of clue strings

    Returns
    -------
    Tuple[List[str], List[int]]
        Tokens (words) and indices where each clue ends
    """
    tokens = []
    run_indices = []

    for clue in clues:
        clue_tokens = clue.split()
        tokens.extend(clue_tokens)
        if clue_tokens:  # Only add index if clue has tokens
            run_indices.append(len(tokens) - 1)

    return tokens, run_indices


def _generate_qid(text: str) -> str:
    """
    Generate a unique question ID from question text.

    Parameters
    ----------
    text : str
        Question text to hash

    Returns
    -------
    str
        Unique identifier based on text hash
    """
    hash_obj = hashlib.md5(text.encode('utf-8'))
    return f"qid-{hash_obj.hexdigest()[:12]}"


class QANTADatasetLoader:
    """
    Loader for QANTA-format quiz bowl CSV files.

    The QANTA format has questions with clues separated by ||| delimiters.
    """

    @classmethod
    def load_from_csv(cls, filepath: str) -> List[TossupQuestion]:
        """
        Load questions from a QANTA-format CSV file.

        Parameters
        ----------
        filepath : str
            Path to the CSV file

        Returns
        -------
        List[TossupQuestion]
            List of parsed questions

        Raises
        ------
        FileNotFoundError
            If the CSV file doesn't exist
        ValueError
            If required columns are missing or data is malformed
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"CSV file not found: {filepath}")

        questions = []

        with open(filepath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)

            # Validate required columns
            required_columns = {'question', 'answer'}
            actual_columns = set(reader.fieldnames or [])

            # Handle alternate column names
            if 'Text' in actual_columns and 'question' not in actual_columns:
                # QANTA format uses 'Text' instead of 'question'
                text_col = 'Text'
            elif 'question' in actual_columns:
                text_col = 'question'
            else:
                raise ValueError(f"Missing required column 'question' or 'Text'. Found columns: {actual_columns}")

            if 'Answer' in actual_columns and 'answer' not in actual_columns:
                answer_col = 'Answer'
            elif 'answer' in actual_columns:
                answer_col = 'answer'
            else:
                raise ValueError(f"Missing required column 'answer' or 'Answer'. Found columns: {actual_columns}")

            # Check for optional columns
            category_col = None
            if 'Category' in actual_columns:
                category_col = 'Category'
            elif 'category' in actual_columns:
                category_col = 'category'

            qid_col = None
            if 'Question ID' in actual_columns:
                qid_col = 'Question ID'
            elif 'qid' in actual_columns:
                qid_col = 'qid'
            elif 'question_id' in actual_columns:
                qid_col = 'question_id'

            # Parse each row
            for row_idx, row in enumerate(reader):
                try:
                    # Get question text and parse clues
                    question_text = row[text_col]
                    if not question_text or not question_text.strip():
                        continue  # Skip empty questions

                    # Split on ||| delimiter
                    if '|||' in question_text:
                        clues = [clue.strip() for clue in question_text.split('|||')]
                        clues = [c for c in clues if c]  # Remove empty clues
                    else:
                        # Treat entire text as single clue if no delimiter
                        clues = [question_text.strip()]

                    if not clues:
                        continue  # Skip if no valid clues

                    # Get answer
                    answer = row[answer_col].strip()
                    if not answer:
                        continue  # Skip questions without answers

                    # Get category (optional)
                    category = ""
                    if category_col:
                        category = row.get(category_col, "").strip()

                    # Get or generate question ID
                    if qid_col and row.get(qid_col):
                        qid = row[qid_col].strip()
                    else:
                        qid = _generate_qid(question_text)

                    # Parse clues into tokens and run indices
                    tokens, run_indices = _parse_clues_to_tokens(clues)

                    # Build cumulative prefixes
                    cumulative_prefixes = []
                    for idx in run_indices:
                        prefix = " ".join(tokens[:idx + 1])
                        cumulative_prefixes.append(prefix)

                    # Create clean answers list
                    clean_answers = [normalize_answer(answer)]

                    # Full question is all clues joined
                    full_question = " ".join(clues)

                    # Create TossupQuestion
                    question = TossupQuestion(
                        qid=qid,
                        question=full_question,
                        tokens=tokens,
                        answer_primary=answer,
                        clean_answers=clean_answers,
                        run_indices=run_indices,
                        human_buzz_positions=None,  # Not available in basic CSV
                        category=category,
                        cumulative_prefixes=cumulative_prefixes
                    )

                    questions.append(question)

                except Exception as e:
                    print(f"Warning: Failed to parse row {row_idx + 1}: {e}")
                    continue

        if not questions:
            raise ValueError(f"No valid questions found in {filepath}")

        return questions