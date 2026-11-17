"""
Email Spam Detector — Streamlit Application
============================================

An NLP + Machine Learning application that classifies SMS/email messages
as SPAM or HAM (not spam) using TF-IDF features and classical ML models.

Author: AI-NLP-Portfolio
"""

from __future__ import annotations

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_loader import load_dataset
from src.model_trainer import (
    train_and_evaluate,
    load_saved_model,
    predict_message,
    get_top_features,
    TrainingResult,
    ModelMetrics,
)
from src.preprocessing import validate_message, preprocess_text
from utils.visualization import (
    plot_confusion_matrix,
    plot_model_comparison,
    plot_prediction_probability,
    plot_top_features,
)
from utils.csv_handler import process_csv_batch, dataframe_to_csv_bytes

st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    .stApp { font-family: 'Inter', sans-serif; }

    .main-header {
        text-align: center;
        padding: 2rem 1rem 1.5rem 1rem;
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(245, 87, 108, 0.25);
    }
    .main-header h1 { color: white; font-size: 2.4rem; font-weight: 700; margin-bottom: 0.4rem; letter-spacing: -0.5px; }
    .main-header p { color: #fef2f2; font-size: 0.95rem; margin: 0; }
    .result-card {
        padding: 2rem 1.5rem; border-radius: 16px; text-align: center;
        margin: 1rem 0; box-shadow: 0 4px 16px rgba(0,0,0,0.1);
    }
    .result-spam {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border: 2px solid #ef4444;
    }
    .result-ham {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        border: 2px solid #22c55e;
    }
    .result-label { font-size: 2.2rem; font-weight: 700; margin: 0; letter-spacing: -0.5px; }
    .result-confidence { font-size: 1.3rem; margin-top: 0.6rem; font-weight: 500; color: #475569; }
    .info-card {
        background: #f8fafc; padding: 1.2rem 1.5rem; border-radius: 10px;
        border-left: 4px solid #f5576c; margin: 0.5rem 0; font-size: 0.9rem; color: #475569;
    }
    .metric-box {
        background: white; padding: 1.2rem; border-radius: 12px;
        text-align: center; border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04); margin: 0.3rem 0;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #f5576c; }
    .metric-label { font-size: 0.8rem; color: #94a3b8; margin-top: 0.3rem; text-transform: uppercase; letter-spacing: 0.5px; }
    .footer { text-align: center; padding: 2rem 0 1rem 0; color: #94a3b8; font-size: 0.85rem; border-top: 1px solid #e2e8f0; margin-top: 2rem; }
    .footer a { color: #f5576c; text-decoration: none; }
    .stButton > button { border-radius: 10px; font-weight: 500; transition: all 0.2s; }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    section[data-testid="stSidebar"] { background: #fafbfc; }
    .stRadio > label { font-weight: 600; color: #334155; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

EXAMPLE_MESSAGES = [
    ("Spam: Prize Winner", "CONGRATULATIONS! You have been selected as the winner of a $1,000 Walmart gift card. To claim your prize, reply with CLAIM to 555-1234 or visit http://bit.ly/claim-now."),
    ("Spam: Free Offer", "FREE RINGTONE! Text SOUNDS to 87077 to get a free polyphonic ringtone. Limited time offer. T&C apply. 18+ only."),
    ("Ham: Normal Message", "Hey, are we still on for dinner tonight? Let me know what time works for you. I was thinking around 7pm at that Italian place."),
    ("Ham: Work Email", "Hi team, I've attached the quarterly report for your review. Please send your feedback by Friday afternoon. Thanks!"),
]


@st.cache_resource
def get_trained_model() -> tuple[object, object, ModelMetrics, TrainingResult]:
    """Load dataset, train models, and cache the result."""
    df = load_dataset()
    result = train_and_evaluate(df)
    return result.best_model, result.vectorizer, result.best_metrics, result


def render_header():
    st.markdown(
        """
        <div class="main-header">
            <h1>📧 Email Spam Detector</h1>
            <p>TF-IDF · Naive Bayes · Logistic Regression · Scikit-learn</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    with st.sidebar:
        st.markdown("### 📋 Navigation")
        page = st.radio(
            "Go to",
            ["Classify Message", "Batch Detection", "Model Evaluation", "About"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown("### 📊 Model Status")
        try:
            with st.spinner("Training models..."):
                model, vectorizer, metrics, training_result = get_trained_model()
            st.success("✅ Model ready")
            st.caption(f"Best: `{metrics.model_name}`")
            st.caption(f"F1-Score: `{metrics.f1_score:.4f}`")
        except Exception as exc:
            st.error(f"❌ Model error: {exc}")
            model = vectorizer = metrics = training_result = None

        return page, model, vectorizer, metrics, training_result


def render_result_card(label: str, confidence: float):
    css = "result-spam" if label == "SPAM" else "result-ham"
    emoji = "🚫" if label == "SPAM" else "✅"
    st.markdown(
        f"""
        <div class="result-card {css}">
            <p class="result-label">{emoji} {label}</p>
            <p class="result-confidence">Confidence: {confidence * 100:.1f}%</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_classify(model, vectorizer):
    st.markdown("## 🔍 Classify a Message")
    st.markdown("Enter an SMS or email message to check if it's spam.")

    col_ex = st.columns(4)
    for i, (name, msg) in enumerate(EXAMPLE_MESSAGES):
        with col_ex[i]:
            if st.button(name, key=f"ex_{i}", use_container_width=True):
                st.session_state["msg_input"] = msg

    default_msg = st.session_state.get("msg_input", "")
    user_input = st.text_area(
        "Enter message:",
        value=default_msg,
        height=150,
        placeholder="Type or paste a message here...",
        key="msg_text",
    )

    col1, col2, _ = st.columns([1, 1, 4])
    with col1:
        predict_btn = st.button("🚀 Classify", type="primary", use_container_width=True)
    with col2:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)

    if clear_btn:
        st.session_state["msg_input"] = ""
        st.rerun()

    if predict_btn:
        is_valid, err = validate_message(user_input)
        if not is_valid:
            st.warning(err)
            return
        if model is None:
            st.error("Model not loaded.")
            return

        result = predict_message(user_input, model, vectorizer)

        st.markdown("---")
        col_res, col_viz = st.columns([1, 2])
        with col_res:
            render_result_card(result["label"], result["confidence"])

        with col_viz:
            st.pyplot(
                plot_prediction_probability(
                    result["spam_probability"], result["ham_probability"]
                )
            )

        with st.expander("📝 Preprocessed Text"):
            st.code(preprocess_text(user_input), language="text")


def page_batch(model, vectorizer):
    st.markdown("## 📂 Batch Detection")
    st.markdown("Upload a CSV file with messages to classify in bulk.")

    col_up, col_col = st.columns([3, 1])
    with col_up:
        uploaded = st.file_uploader("Choose CSV", type=["csv"])
    with col_col:
        col_name = st.text_input("Column name (optional)", placeholder="auto-detect")

    if uploaded is not None:
        if model is None:
            st.error("Model not loaded.")
            return
        try:
            with st.spinner("Processing..."):
                result_df = process_csv_batch(
                    uploaded, model, vectorizer,
                    text_column=col_name if col_name.strip() else None,
                )
        except ValueError as e:
            st.error(f"Error: {e}")
            return

        st.success(f"✅ Processed {len(result_df)} messages.")

        spam_count = (result_df["prediction"] == "SPAM").sum()
        ham_count = (result_df["prediction"] == "HAM").sum()

        col_s, col_h = st.columns(2)
        with col_s:
            st.metric("🚫 Spam Detected", int(spam_count))
        with col_h:
            st.metric("✅ Ham (Legitimate)", int(ham_count))

        st.markdown("### Results")
        st.dataframe(result_df, use_container_width=True, height=400)

        st.download_button(
            "⬇️ Download Results",
            data=dataframe_to_csv_bytes(result_df),
            file_name="spam_results.csv",
            mime="text/csv",
        )


def page_evaluation(metrics, training_result):
    st.markdown("## 📊 Model Evaluation Dashboard")

    if metrics is None or training_result is None:
        st.error("Model not available.")
        return

    st.markdown(f"### Best Model: {metrics.model_name}")

    col1, col2, col3, col4 = st.columns(4)
    for col, label, value in [
        (col1, "Accuracy", metrics.accuracy),
        (col2, "Precision", metrics.precision),
        (col3, "Recall", metrics.recall),
        (col4, "F1-Score", metrics.f1_score),
    ]:
        with col:
            st.markdown(
                f"<div class='metric-box'>"
                f"<div class='metric-value'>{value * 100:.2f}%</div>"
                f"<div class='metric-label'>{label}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("---")
    col_cm, col_cmp = st.columns(2)
    with col_cm:
        st.pyplot(plot_confusion_matrix(metrics.confusion_matrix))
    with col_cmp:
        st.pyplot(plot_model_comparison(training_result.all_metrics))

    st.markdown("### Classification Report")
    st.code(metrics.classification_report, language="text")

    st.markdown("---")
    st.markdown("### 🔎 Feature Importance (Explainability)")
    st.info(
        "The following words are the strongest indicators according to the model. "
        "These are **model-based statistical indicators**, not definitive causal explanations."
    )

    try:
        features = get_top_features(training_result.best_model, training_result.vectorizer)
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            st.pyplot(plot_top_features(features))
        with col_f2:
            st.markdown("**Top Spam-Indicating Words:**")
            for word, weight in features.get("spam", [])[:10]:
                st.markdown(f"- `{word}` (weight: {weight:.4f})")
    except Exception as e:
        st.warning(f"Could not extract features: {e}")


def page_about():
    st.markdown("## ℹ️ About")
    st.markdown(
        """
        <div class="info-card">
            <h3>Email Spam Detector</h3>
            <p>An NLP + Machine Learning application that classifies SMS/email
            messages as <b>SPAM</b> or <b>HAM</b> using TF-IDF features and
            classical ML models (Naive Bayes & Logistic Regression).</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("### 🛠️ Technologies")
    for t in [
        "Python 3.10+", "Streamlit", "Scikit-learn", "NLTK",
        "Pandas", "NumPy", "Matplotlib", "Seaborn",
    ]:
        st.markdown(f"- {t}")
    st.markdown("### 📊 Dataset")
    st.markdown("UCI SMS Spam Collection Dataset (~5,500 messages)")
    st.markdown("---")
    st.markdown(
        "<div class='footer'>Email Spam Detector · Powered by Scikit-learn · Built with Streamlit</div>",
        unsafe_allow_html=True,
    )


def main():
    render_header()
    page, model, vectorizer, metrics, training_result = render_sidebar()

    if page == "Classify Message":
        page_classify(model, vectorizer)
    elif page == "Batch Detection":
        page_batch(model, vectorizer)
    elif page == "Model Evaluation":
        page_evaluation(metrics, training_result)
    elif page == "About":
        page_about()


if __name__ == "__main__":
    main()