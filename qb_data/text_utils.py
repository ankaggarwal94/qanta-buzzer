"""
Text utilities for quiz bowl answer normalization and tokenization.
"""

import re
import string
from typing import Optional, List


def tokenize_text(text: str) -> List[str]:
    """
    Tokenize text by splitting on whitespace.

    Parameters
    ----------
    text : str
        Text to tokenize

    Returns
    -------
    List[str]
        List of tokens (words)
    """
    if not text:
        return []
    return text.split()


def normalize_answer(answer: str) -> str:
    """
    Normalize an answer string for comparison.

    Removes articles (a, an, the) from the beginning, converts to lowercase,
    strips punctuation and extra whitespace, and handles edge cases.

    Parameters
    ----------
    answer : str
        The answer string to normalize

    Returns
    -------
    str
        The normalized answer string

    Examples
    --------
    >>> normalize_answer("The Great Gatsby")
    'great gatsby'
    >>> normalize_answer("A Tale of Two Cities!")
    'tale of two cities'
    >>> normalize_answer("   An    Example   ")
    'example'
    >>> normalize_answer("")
    ''
    """
    if not answer:
        return ""

    # Convert to lowercase
    normalized = answer.lower()

    # Remove leading/trailing whitespace
    normalized = normalized.strip()

    # Remove leading articles (a, an, the)
    # Use \b word boundary to ensure we match complete words
    normalized = re.sub(r'^(a|an|the)\b\s*', '', normalized)

    # Remove punctuation
    # Keep alphanumeric characters and spaces
    normalized = re.sub(r'[^\w\s]', '', normalized)

    # Normalize whitespace (collapse multiple spaces to single space)
    normalized = re.sub(r'\s+', ' ', normalized)

    # Final strip in case punctuation removal left spaces
    normalized = normalized.strip()

    return normalized