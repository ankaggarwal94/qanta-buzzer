"""
HuggingFace dataset loader for quiz bowl data.

This module provides fallback loading from HuggingFace Hub when local CSV files
are not available.
"""

from typing import List, Optional, Dict, Any

from qb_data.data_loader import TossupQuestion
from qb_data.text_utils import tokenize_text, normalize_answer


def load_from_huggingface(
    dataset_name: str,
    config_name: Optional[str] = None,
    split: str = "eval"
) -> List[TossupQuestion]:
    """
    Load quiz bowl dataset from HuggingFace Hub.

    Parameters
    ----------
    dataset_name : str
        Name of the HuggingFace dataset (e.g., "qanta-challenge/acf-co24-tossups")
    config_name : Optional[str]
        Configuration name for the dataset (e.g., "questions", "tossup")
    split : str
        Dataset split to load (default: "eval")

    Returns
    -------
    List[TossupQuestion]
        List of parsed questions

    Raises
    ------
    ImportError
        If datasets library is not installed
    ValueError
        If dataset not found or required fields missing
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Warning: datasets library not installed. Falling back to CSV loader.")
        print("Install with: pip install datasets")
        raise ImportError("HuggingFace datasets library not available. Please use CSV fallback.")

    # Known dataset configurations from qb-rl
    known_configs = {
        "qanta-challenge/acf-co24-tossups": "questions",
        "qanta-challenge/qanta25-playground": "tossup"
    }

    # Use known config if not provided
    if config_name is None and dataset_name in known_configs:
        config_name = known_configs[dataset_name]
        print(f"Using known config '{config_name}' for {dataset_name}")

    # Try to load dataset
    try:
        print(f"Loading {dataset_name} from HuggingFace Hub...")
        if config_name:
            dataset = load_dataset(dataset_name, config_name, split=split)
        else:
            dataset = load_dataset(dataset_name, split=split)
        print(f"Successfully loaded {len(dataset)} questions")
    except Exception as e:
        error_msg = f"Failed to load dataset {dataset_name}: {e}"
        print(f"Error: {error_msg}")
        print("Falling back to local CSV loader...")
        raise ValueError(error_msg)

    # Parse dataset rows into TossupQuestion format
    questions = []
    for idx, row in enumerate(dataset):
        try:
            question = parse_huggingface_row(row, idx)
            questions.append(question)
        except KeyError as e:
            print(f"Warning: Skipping row {idx} due to missing field: {e}")
            continue
        except Exception as e:
            print(f"Warning: Failed to parse row {idx}: {e}")
            continue

    if not questions:
        raise ValueError(f"No valid questions parsed from {dataset_name}")

    print(f"Parsed {len(questions)} questions from HuggingFace dataset")
    return questions


def parse_huggingface_row(row: Dict[str, Any], idx: int = 0) -> TossupQuestion:
    """
    Parse a HuggingFace dataset row into TossupQuestion format.

    Parameters
    ----------
    row : Dict[str, Any]
        Single row from HuggingFace dataset
    idx : int
        Row index for generating IDs

    Returns
    -------
    TossupQuestion
        Parsed question object

    Raises
    ------
    KeyError
        If required fields are missing
    """
    # Field mapping for different dataset formats
    # Primary fields
    question_fields = ["question", "text", "question_text", "tossup_text"]
    answer_fields = ["answer_primary", "answer", "clean_answer", "clean_answers", "page"]
    category_fields = ["category", "topic", "subject"]

    # Extract question text
    question_text = None
    for field in question_fields:
        if field in row:
            question_text = row[field]
            break

    if not question_text:
        raise KeyError(f"No question field found. Available fields: {list(row.keys())}")

    # Extract answer
    answer_text = None
    for field in answer_fields:
        if field in row:
            value = row[field]
            # Handle list of answers
            if isinstance(value, list) and value:
                answer_text = value[0]
            elif isinstance(value, str):
                answer_text = value
            break

    if not answer_text:
        raise KeyError(f"No answer field found. Available fields: {list(row.keys())}")

    # Extract category (with default)
    category = "General"
    for field in category_fields:
        if field in row and row[field]:
            category = str(row[field])
            break

    # Generate ID if not present
    qid = row.get("qid") or row.get("id") or row.get("qanta_id") or f"hf_{idx:06d}"

    # Handle clues that may be separated by ||| or in a list
    if "|||" in question_text:
        # QANTA format with ||| separators
        clues = question_text.split("|||")
        question_text = " ".join(clues)
    elif isinstance(question_text, list):
        # List of clues
        clues = question_text
        question_text = " ".join(clues)
    else:
        # Single text, split by sentences as approximation
        import re
        sentences = re.split(r'(?<=[.!?])\s+', question_text)
        clues = sentences if len(sentences) > 1 else [question_text]

    # Tokenize text
    tokens = tokenize_text(question_text)

    # Build run indices (boundaries between clues)
    run_indices = []
    current_pos = 0
    for clue in clues:
        clue_tokens = tokenize_text(clue)
        current_pos += len(clue_tokens)
        if current_pos > 0:
            run_indices.append(current_pos - 1)  # Index is 0-based

    # Build cumulative prefixes
    cumulative_prefixes = []
    for idx in run_indices:
        prefix = " ".join(tokens[:idx + 1])
        cumulative_prefixes.append(prefix)

    # Normalize answer for matching
    clean_answers = [normalize_answer(answer_text)]

    return TossupQuestion(
        qid=qid,
        question=question_text,
        tokens=tokens,
        answer_primary=answer_text,  # Keep original answer as primary
        clean_answers=clean_answers,  # Normalized version for matching
        run_indices=run_indices,
        human_buzz_positions=None,  # Not available from HuggingFace
        category=category,
        cumulative_prefixes=cumulative_prefixes
    )


def try_huggingface_fallback(csv_path: str) -> Optional[List[TossupQuestion]]:
    """
    Attempt to load from HuggingFace if CSV is missing.

    Parameters
    ----------
    csv_path : str
        Path to missing CSV file

    Returns
    -------
    Optional[List[TossupQuestion]]
        Questions if HuggingFace load succeeds, None otherwise
    """
    print(f"CSV file {csv_path} not found. Attempting HuggingFace fallback...")

    # Try known datasets in order
    fallback_datasets = [
        ("qanta-challenge/acf-co24-tossups", "questions"),
        ("qanta-challenge/qanta25-playground", "tossup")
    ]

    for dataset_name, config_name in fallback_datasets:
        try:
            questions = load_from_huggingface(dataset_name, config_name)
            if questions:
                print(f"Successfully loaded {len(questions)} questions from {dataset_name}")
                return questions
        except Exception as e:
            print(f"Failed to load {dataset_name}: {e}")
            continue

    print("All HuggingFace fallback attempts failed")
    return None
