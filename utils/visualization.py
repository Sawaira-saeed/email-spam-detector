"""
Visualization utilities for spam detector using Matplotlib and Seaborn.
"""

from __future__ import annotations

import io

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.figure import Figure


def plot_confusion_matrix(cm: list[list[int]]) -> Figure:
    """
    Render a confusion matrix heatmap.

    Args:
        cm: 2x2 confusion matrix [[TN, FP], [FN, TP]].

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Ham", "Spam"],
        yticklabels=["Ham", "Spam"],
        ax=ax,
        cbar=False,
    )
    ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax.set_ylabel("Actual", fontsize=12, fontweight="bold")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    return fig


def plot_model_comparison(metrics_dict: dict) -> Figure:
    """
    Compare accuracy, precision, recall, and F1 across models.

    Args:
        metrics_dict: Dict of model_name -> ModelMetrics.

    Returns:
        Matplotlib Figure.
    """
    names = list(metrics_dict.keys())
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#667eea", "#764ba2"]

    for i, name in enumerate(names):
        values = [getattr(metrics_dict[name], m) for m in metrics]
        bars = ax.bar(
            x + i * width - width / 2,
            values,
            width,
            label=name,
            color=colors[i % len(colors)],
        )
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
            )

    ax.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax.set_title("Model Comparison", fontsize=14, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def plot_prediction_probability(spam_prob: float, ham_prob: float) -> Figure:
    """
    Horizontal bar chart showing spam vs ham probability.

    Args:
        spam_prob: Probability of spam (0–1).
        ham_prob: Probability of ham (0–1).

    Returns:
        Matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 2.5))
    labels = ["HAM (Not Spam)", "SPAM"]
    values = [ham_prob * 100, spam_prob * 100]
    colors = ["#22c55e", "#ef4444"]

    bars = ax.barh(labels, values, color=colors, height=0.5)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_xlim(0, 110)
    ax.set_xlabel("Probability (%)", fontsize=11)
    ax.set_title("Classification Probability", fontsize=13, fontweight="bold", pad=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig


def plot_top_features(features: dict[str, list[tuple[str, float]]]) -> Figure:
    """
    Bar chart of top spam-indicating words.

    Args:
        features: Dict with 'spam' and 'ham' lists of (word, weight).

    Returns:
        Matplotlib Figure.
    """
    spam_words = features.get("spam", [])[:10]
    if not spam_words:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No feature data available", ha="center", va="center")
        return fig

    words = [w for w, _ in reversed(spam_words)]
    weights = [v for _, v in reversed(spam_words)]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(words, weights, color="#ef4444", height=0.6)
    ax.set_xlabel("Feature Weight (log probability ratio)", fontsize=11)
    ax.set_title("Top Spam-Indicating Words", fontsize=13, fontweight="bold", pad=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    return fig