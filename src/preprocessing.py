"""Text preprocessing for spam detection."""

from __future__ import annotations

import re
import string

import nltk
from nltk.corpus import stopwords

_NLTK_DOWNLOADED = False


def _ensure_nltk_data() -> None:

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

    if text is None or not str(text).strip():
        return False, "Message is empty. Please enter a message to classify."

    text = str(text).strip()
    if len(text) < min_length:
        return False, f"Message is too short (minimum {min_length} character)."

    if len(text) > max_length:
        return False, f"Message is too long (maximum {max_length} characters)."

    return True, ""