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


def _coerce_human_buzz_positions(value: Any) -> Optional[List[Tuple[int, int]]]:
    """Coerce various metadata formats into ``(position, count)`` tuples."""
    if value is None:
        return None

    if isinstance(value, list):
        result: List[Tuple[int, int]] = []
        for item in value:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                try:
                    result.append((int(item[0]), int(item[1])))
                except (TypeError, ValueError):
                    continue
            elif isinstance(item, dict):
                pos = item.get("position")
                count = item.get("count", 1)
                if pos is None:
                    continue
                try:
                    result.append((int(pos), int(count)))
                except (TypeError, ValueError):
                    continue
        return result or None

    return None


def _coerce_run_indices(run_indices: Any, token_count: int) -> List[int]:
    """Validate and coerce run indices into a sorted unique list."""
    clean: List[int] = []
    for idx in run_indices or []:
        try:
            clean.append(int(idx))
        except (TypeError, ValueError):
            continue

    if not clean:
        if token_count <= 0:
            raise ValueError("question must contain at least one token")
        clean = list(range(token_count))

    clean = sorted(set(clean))
    if clean[0] < 0 or clean[-1] > token_count - 1:
        raise ValueError(
            f"run_indices out of bounds: min={clean[0]} max={clean[-1]} token_count={token_count}"
        )
    return clean


def parse_row(row: Dict[str, Any]) -> TossupQuestion:
    """Parse a qb-rl/HuggingFace-style row into ``TossupQuestion``."""
    question = str(row["question"])
    tokens = question.split()
    metadata = row.get("metadata", {}) or {}
    answer_primary = str(
        row.get("answer_primary") or (row.get("clean_answers") or [""])[0]
    ).strip()
    clean_answers = [str(x) for x in (row.get("clean_answers") or [])]
    if not clean_answers and answer_primary:
        clean_answers = [answer_primary]

    run_indices = _coerce_run_indices(
        row.get("run_indices") or [],
        token_count=len(tokens),
    )

    normalized_question = " ".join(question.split())
    normalized_tokens = " ".join(tokens)
    if normalized_tokens != normalized_question:
        raise ValueError("tokenization roundtrip mismatch")
    if max(run_indices) > len(tokens) - 1:
        raise ValueError("run_indices out of bounds")

    cumulative_prefixes = [" ".join(tokens[: idx + 1]) for idx in run_indices]
    category = str(metadata.get("category") or row.get("category") or "")
    human_buzz_positions = _coerce_human_buzz_positions(
        metadata.get("human_buzz_positions") or row.get("human_buzz_positions")
    )

    qid_raw = row.get("qid") or row.get("question_id") or row.get("id")
    if qid_raw is None:
        qid_raw = _generate_qid(question)

    return TossupQuestion(
        qid=str(qid_raw),
        question=question,
        tokens=tokens,
        answer_primary=answer_primary,
        clean_answers=clean_answers,
        run_indices=run_indices,
        human_buzz_positions=human_buzz_positions,
        category=category,
        cumulative_prefixes=cumulative_prefixes,
    )


def load_tossup_questions(
    dataset: str,
    dataset_config: Optional[str] = None,
    split: str = "eval",
    limit: Optional[int] = None,
) -> List[TossupQuestion]:
    """Load tossup questions from Hugging Face datasets using qb-rl semantics."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "datasets is required for Hugging Face loading. Install it with: pip install datasets"
        ) from exc

    if dataset_config:
        ds = load_dataset(dataset, dataset_config, split=split)
    else:
        ds = load_dataset(dataset, split=split)

    if limit is not None:
        ds = ds.select(range(min(int(limit), len(ds))))

    return [parse_row(dict(row)) for row in ds]


def load_tossup_questions_from_config(
    config: Dict[str, Any],
    smoke: bool = False,
) -> List[TossupQuestion]:
    """Load tossups from config, supporting qb-rl and qanta-buzzer keys."""
    from qb_data.config import resolve_data_loading_options

    data_opts = resolve_data_loading_options(config, smoke=smoke)
    csv_path = data_opts.get("csv_path")
    dataset = data_opts.get("dataset")
    dataset_config = data_opts.get("dataset_config")
    split = data_opts.get("split", "eval")
    limit = data_opts.get("max_questions")

    if csv_path and Path(csv_path).exists():
        questions = QANTADatasetLoader.load_from_csv(str(csv_path))
    elif dataset:
        questions = load_tossup_questions(
            dataset=str(dataset),
            dataset_config=str(dataset_config) if dataset_config else None,
            split=str(split),
            limit=int(limit) if limit is not None else None,
        )
    elif csv_path and data_opts.get("use_huggingface"):
        from qb_data.huggingface_loader import try_huggingface_fallback

        questions = try_huggingface_fallback(str(csv_path))
        if questions is None:
            raise FileNotFoundError(
                f"Could not load questions from missing CSV path {csv_path} via Hugging Face fallback"
            )
    else:
        raise FileNotFoundError(
            "No valid data source configured. Provide data.csv_path or "
            "data.dataset/data.dataset_config for qb-rl compatibility."
        )

    if limit is not None:
        questions = questions[: int(limit)]

    return questions


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
