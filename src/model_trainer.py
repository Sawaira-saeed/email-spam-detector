"""Model training, evaluation, and persistence for spam detection."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from .preprocessing import preprocess_text

logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_FILE = MODELS_DIR / "spam_model.joblib"
VECTORIZER_FILE = MODELS_DIR / "tfidf_vectorizer.joblib"
METRICS_FILE = MODELS_DIR / "metrics.joblib"

RANDOM_SEED = 42


@dataclass
class ModelMetrics:


    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: list[list[int]]
    classification_report: str


@dataclass
class TrainingResult:


    best_model_name: str
    best_metrics: ModelMetrics
    all_metrics: dict[str, ModelMetrics] = field(default_factory=dict)
    vectorizer: Optional[TfidfVectorizer] = None
    best_model: Optional[object] = None
    feature_names: list[str] = field(default_factory=list)


def train_and_evaluate(df: pd.DataFrame) -> TrainingResult:

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    df = df.dropna(subset=["label", "message"])
    df = df.drop_duplicates(subset=["message"])
    df["message"] = df["message"].astype(str)
    df["cleaned"] = df["message"].apply(preprocess_text)
    df = df[df["cleaned"].str.len() > 0]

    label_map = {"ham": 0, "spam": 1}
    df["label_num"] = df["label"].str.lower().map(label_map)
    df = df.dropna(subset=["label_num"])
    df["label_num"] = df["label_num"].astype(int)

    X = df["cleaned"].values
    y = df["label_num"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    models = {
        "Multinomial Naive Bayes": MultinomialNB(alpha=1.0),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=RANDOM_SEED, solver="liblinear"
        ),
    }

    all_metrics: dict[str, ModelMetrics] = {}
    trained_models: dict[str, object] = {}

    for name, model in models.items():
        logger.info("Training %s...", name)
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)

        metrics = ModelMetrics(
            model_name=name,
            accuracy=float(accuracy_score(y_test, y_pred)),
            precision=float(precision_score(y_test, y_pred, zero_division=0)),
            recall=float(recall_score(y_test, y_pred, zero_division=0)),
            f1_score=float(f1_score(y_test, y_pred, zero_division=0)),
            confusion_matrix=confusion_matrix(y_test, y_pred).tolist(),
            classification_report=classification_report(
                y_test, y_pred, target_names=["Ham", "Spam"], zero_division=0
            ),
        )
        all_metrics[name] = metrics
        trained_models[name] = model
        logger.info(
            "%s — Accuracy: %.4f, F1: %.4f",
            name, metrics.accuracy, metrics.f1_score,
        )

    best_name = max(all_metrics, key=lambda k: all_metrics[k].f1_score)
    best_model = trained_models[best_name]
    best_metrics = all_metrics[best_name]

    joblib.dump(best_model, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)
    joblib.dump(best_metrics, METRICS_FILE)

    feature_names = vectorizer.get_feature_names_out().tolist()

    logger.info("Best model: %s (F1: %.4f)", best_name, best_metrics.f1_score)

    return TrainingResult(
        best_model_name=best_name,
        best_metrics=best_metrics,
        all_metrics=all_metrics,
        vectorizer=vectorizer,
        best_model=best_model,
        feature_names=feature_names,
    )


def load_saved_model() -> tuple[object, TfidfVectorizer, ModelMetrics]:

    if not MODEL_FILE.exists() or not VECTORIZER_FILE.exists():
        raise FileNotFoundError(
            "Trained model not found. Please train the model first."
        )

    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)
    metrics = joblib.load(METRICS_FILE) if METRICS_FILE.exists() else None

    return model, vectorizer, metrics


def predict_message(
    text: str,
    model: object,
    vectorizer: TfidfVectorizer,
) -> dict:

    cleaned = preprocess_text(text)
    if not cleaned:
        return {
            "label": "HAM",
            "confidence": 0.0,
            "spam_probability": 0.0,
            "ham_probability": 1.0,
        }

    vec = vectorizer.transform([cleaned])
    prediction = int(model.predict(vec)[0])

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec)[0]
        spam_prob = float(probs[1])
        ham_prob = float(probs[0])
    else:
        spam_prob = float(prediction)
        ham_prob = 1.0 - spam_prob

    label = "SPAM" if prediction == 1 else "HAM"
    confidence = max(spam_prob, ham_prob)

    return {
        "label": label,
        "confidence": confidence,
        "spam_probability": spam_prob,
        "ham_probability": ham_prob,
    }


def get_top_features(
    model: object,
    vectorizer: TfidfVectorizer,
    top_n: int = 15,
) -> dict[str, list[tuple[str, float]]]:

    feature_names = vectorizer.get_feature_names_out()

    if hasattr(model, "coef_"):
        coefs = model.coef_[0]
        top_spam = sorted(
            zip(feature_names, coefs), key=lambda x: x[1], reverse=True
        )[:top_n]
        top_ham = sorted(
            zip(feature_names, coefs), key=lambda x: x[1]
        )[:top_n]
        return {
            "spam": [(w, float(v)) for w, v in top_spam],
            "ham": [(w, float(v)) for w, v in top_ham],
        }
    elif hasattr(model, "feature_log_prob_"):
        diff = model.feature_log_prob_[1] - model.feature_log_prob_[0]
        top_spam = sorted(
            zip(feature_names, diff), key=lambda x: x[1], reverse=True
        )[:top_n]
        top_ham = sorted(
            zip(feature_names, diff), key=lambda x: x[1]
        )[:top_n]
        return {
            "spam": [(w, float(v)) for w, v in top_spam],
            "ham": [(w, float(v)) for w, v in top_ham],
        }

    return {"spam": [], "ham": []}