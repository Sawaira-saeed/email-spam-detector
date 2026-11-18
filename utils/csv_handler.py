"""CSV batch processing for spam detector."""

from __future__ import annotations

import io
from typing import Optional

import pandas as pd

from src.preprocessing import preprocess_text
from src.model_trainer import predict_message


def process_csv_batch(
    file_obj,
    model,
    vectorizer,
    text_column: Optional[str] = None,
) -> pd.DataFrame:

    try:
        df = pd.read_csv(file_obj)
    except Exception as exc:
        raise ValueError(f"Failed to read CSV: {exc}") from exc

    if df.empty:
        raise ValueError("CSV file is empty.")

    if text_column is not None:
        if text_column not in df.columns:
            raise ValueError(
                f"Column '{text_column}' not found. Available: {list(df.columns)}"
            )
        col = text_column
    else:
        object_cols = df.select_dtypes(include=["object"]).columns.tolist()
        if not object_cols:
            raise ValueError("No text column found in CSV.")
        col = object_cols[0]

    messages = df[col].fillna("").astype(str).tolist()

    results = []
    for msg in messages:
        result = predict_message(msg, model, vectorizer)
        results.append(result)

    output_df = pd.DataFrame(
        {
            "message": messages,
            "prediction": [r["label"] for r in results],
            "spam_probability": [round(r["spam_probability"] * 100, 2) for r in results],
            "confidence": [round(r["confidence"] * 100, 2) for r in results],
        }
    )

    return output_df


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:

    buffer = io.StringIO()
    df.to_csv(buffer, index=False)
    return buffer.getvalue().encode("utf-8")