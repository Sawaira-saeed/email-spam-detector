"""
Text preprocessing for spam detection.

Steps:
    1. Lowercasing
    2. URL replacement
    3. Email address replacement
    4. Phone number replacement
    5. Currency symbol replacement
    6. Punctuation removal
    7. Digit removal (optional)
    8. Stopword removal (optional)
    9. Whitespace normalization
"""

from __future__ import annotations

import re
import string

import nltk
from nltk.corpus import stopwords

_NLTK_DOWNLOADED = False


def _ensure_nltk_data() -> None:
    """Download required NLTK data (stopwords) if not already present."""
    global _NLTK_DOWNLOADED
    if _NLTK_DOWNLOADED:
        return
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    _NLTK_DOWNLOADED = True


def preprocess_text(
    text: str,
    remove_stopwords: bool = True,
    remove_digits: bool = True,
) -> str:
    """
    Clean and normalize a single SMS/email message.

    Args:
        text: Raw message string.
        remove_stopwords: Whether to remove English stopwords.
        remove_digits: Whether to remove standalone digits.

    Returns:
        Cleaned text string.
    """
    if text is None:
        return ""

    text = str(text).lower()

    text = re.sub(r"https?://\S+|www\.\S+", " urltoken ", text)
    text = re.sub(r"\S+@\S+\.\S+", " emailtoken ", text)
    text = re.sub(r"\b\d{5,}\b", " phonetoken ", text)
    text = re.sub(r"£|\$|€", " currencytoken ", text)
    text = re.sub(r"\b\d+\b", " numbertoken ", text) if remove_digits else text

    text = text.translate(str.maketrans("", "", string.punctuation))

    text = re.sub(r"\s+", " ", text).strip()

    if remove_stopwords:
        _ensure_nltk_data()
        try:
            stop_words = set(stopwords.words("english"))
            tokens = [w for w in text.split() if w not in stop_words]
            text = " ".join(tokens)
        except Exception:
            pass

    return text


def validate_message(text: str, min_length: int = 1, max_length: int = 5000) -> tuple[bool, str]:
    """
    Validate an email/SMS message before classification.

    Returns:
        (is_valid, error_message)
    """
    if text is None or not str(text).strip():
        return False, "Message is empty. Please enter a message to classify."

    text = str(text).strip()
    if len(text) < min_length:
        return False, f"Message is too short (minimum {min_length} character)."

    if len(text) > max_length:
        return False, f"Message is too long (maximum {max_length} characters)."

    return True, ""