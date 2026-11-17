"""
Test suite for the Email Spam Detector.

Run with: pytest tests/ -v
"""

import os
import sys
import io
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.preprocessing import preprocess_text, validate_message
from src.data_loader import load_dataset
from src.model_trainer import train_and_evaluate, predict_message, get_top_features


# ─── Preprocessing Tests ───────────────────────────────────────────────

class TestPreprocessing:
    def test_lowercase(self):
        result = preprocess_text("HELLO WORLD")
        assert result == result.lower()

    def test_url_replacement(self):
        result = preprocess_text("Visit http://example.com now")
        assert "urltoken" in result
        assert "http" not in result

    def test_email_replacement(self):
        result = preprocess_text("Contact me at test@example.com")
        assert "emailtoken" in result

    def test_punctuation_removal(self):
        result = preprocess_text("Hello!!! How are you???")
        assert "!" not in result
        assert "?" not in result

    def test_empty_input(self):
        assert preprocess_text("") == ""
        assert preprocess_text(None) == ""

    def test_validate_valid(self):
        is_valid, msg = validate_message("Hello there")
        assert is_valid is True

    def test_validate_empty(self):
        is_valid, msg = validate_message("")
        assert is_valid is False

    def test_validate_none(self):
        is_valid, msg = validate_message(None)
        assert is_valid is False

    def test_validate_too_long(self):
        is_valid, msg = validate_message("a" * 10000, max_length=5000)
        assert is_valid is False


# ─── Dataset Tests ─────────────────────────────────────────────────────

class TestDataset:
    def test_load_dataset(self):
        df = load_dataset()
        assert not df.empty
        assert "label" in df.columns
        assert "message" in df.columns
        assert set(df["label"].unique()).issubset({"ham", "spam"})

    def test_dataset_has_both_classes(self):
        df = load_dataset()
        assert "spam" in df["label"].values
        assert "ham" in df["label"].values


# ─── Model Tests ───────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def trained_model():
    df = load_dataset()
    result = train_and_evaluate(df)
    return result


class TestModel:
    def test_training_succeeds(self, trained_model):
        assert trained_model.best_model is not None
        assert trained_model.vectorizer is not None
        assert trained_model.best_model_name in (
            "Multinomial Naive Bayes", "Logistic Regression"
        )

    def test_metrics_present(self, trained_model):
        m = trained_model.best_metrics
        assert 0.0 <= m.accuracy <= 1.0
        assert 0.0 <= m.precision <= 1.0
        assert 0.0 <= m.recall <= 1.0
        assert 0.0 <= m.f1_score <= 1.0
        assert len(m.confusion_matrix) == 2
        assert len(m.confusion_matrix[0]) == 2

    def test_both_models_evaluated(self, trained_model):
        assert len(trained_model.all_metrics) == 2

    def test_predict_spam(self, trained_model):
        result = predict_message(
            "CONGRATULATIONS! You have won a $1000 prize. Call now to claim!",
            trained_model.best_model,
            trained_model.vectorizer,
        )
        assert result["label"] in ("SPAM", "HAM")
        assert 0.0 <= result["confidence"] <= 1.0
        assert 0.0 <= result["spam_probability"] <= 1.0

    def test_predict_ham(self, trained_model):
        result = predict_message(
            "Hey, are we still on for dinner tonight? Let me know.",
            trained_model.best_model,
            trained_model.vectorizer,
        )
        assert result["label"] in ("SPAM", "HAM")
        assert 0.0 <= result["confidence"] <= 1.0

    def test_predict_empty(self, trained_model):
        result = predict_message(
            "", trained_model.best_model, trained_model.vectorizer
        )
        assert result["label"] == "HAM"
        assert result["confidence"] == 0.0

    def test_top_features(self, trained_model):
        features = get_top_features(
            trained_model.best_model, trained_model.vectorizer
        )
        assert "spam" in features
        assert "ham" in features


# ─── CSV Handler Tests ─────────────────────────────────────────────────

class TestCSVHandler:
    def test_batch_processing(self, trained_model):
        from utils.csv_handler import process_csv_batch
        csv = "message\nWin a free prize now\nHey let's meet tomorrow\n"
        df = process_csv_batch(
            io.StringIO(csv),
            trained_model.best_model,
            trained_model.vectorizer,
        )
        assert len(df) == 2
        assert "prediction" in df.columns

    def test_batch_empty_raises(self, trained_model):
        from utils.csv_handler import process_csv_batch
        with pytest.raises(ValueError):
            process_csv_batch(
                io.StringIO("message\n"),
                trained_model.best_model,
                trained_model.vectorizer,
            )