"""
SentimentScope - Streamlit Web Application
An interactive, beginner-friendly sentiment analysis tool.
"""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st

from src.model import (
    build_sentiment_pipeline,
    evaluate_model,
    load_pipeline,
    predict_sentiment,
    save_pipeline,
    train_model,
)

# Page configuration
st.set_page_config(
    page_title="SentimentScope | AI Sentiment Classifier",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT_DIR = Path(__file__).resolve().parent
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "sentiment_pipeline.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
CM_PLOT_PATH = MODELS_DIR / "confusion_matrix.png"


@st.cache_resource(show_spinner=False)
def get_or_train_pipeline():
    """
    Load the saved pipeline from disk. If not yet trained,
    provide a lightweight initial training to ensure instant usability.
    """
    if MODEL_PATH.exists():
        try:
            return load_pipeline(MODEL_PATH), False
        except Exception:
            pass

    # Quick starter baseline training if train.py hasn't been executed yet
    starter_texts = [
        "A truly remarkable, heartwarming masterpiece with stellar acting.",
        "One of the best movies ever made, absolutely thrilling and brilliant.",
        "I loved this film! Inspiring, fun, and wonderful throughout.",
        "Great storytelling, charming humor, and unforgettable moments.",
        "A magnificent performance and visual feast for the eyes.",
        "An utter disaster, boring, painfully slow, and dreadful.",
        "Terrible acting, poorly written script, and completely unwatchable.",
        "Worst experience imaginable, totally nonsensical and frustrating.",
        "A disappointing waste of talent, time, and budget.",
        "Horrible plot, dull characters, and uninspired dialogue.",
    ] * 20
    starter_labels = ([1] * 5 + [0] * 5) * 20

    pipeline = build_sentiment_pipeline(max_features=5000)
    pipeline = train_model(starter_texts, starter_labels, pipeline)
    return pipeline, True


def load_saved_metrics():
    """Load evaluation metrics JSON if available."""
    if METRICS_PATH.exists():
        try:
            with open(METRICS_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def main():
    # Sidebar
    with st.sidebar:
        st.markdown("## 🎯 SentimentScope")
        st.markdown(
            "An educational NLP project built for the **Generative AI Hackathon**."
        )
        st.markdown("---")
        st.markdown("### 📌 Model Architecture")
        st.markdown("- **Algorithm:** Logistic Regression")
        st.markdown("- **Feature Extractor:** TF-IDF (Unigrams & Bigrams)")
        st.markdown("- **Dataset:** SST-2 (Stanford Sentiment Treebank)")
        st.markdown("- **Task:** Binary Sentiment (Positive / Negative)")
        st.markdown("---")
        st.markdown("### 💻 Quick Start Command")
        st.code("streamlit run app.py", language="bash")
        st.markdown("---")
        st.caption("SentimentScope Baseline • Built with Python & Streamlit")

    # Main Header
    st.title("🎯 SentimentScope")
    st.markdown(
        "Explore how classic machine learning predicts sentiment in text. "
        "Type your own sentence or choose a preset example below!"
    )

    pipeline, is_starter = get_or_train_pipeline()

    if is_starter:
        st.info(
            "ℹ️ **Note:** The full SST-2 model has not been trained yet. SentimentScope is running "
            "on a lightweight starter baseline. Run `python src/train.py` in your terminal to train "
            "on the full 67,000+ SST-2 dataset from Hugging Face!"
        )

    # Navigation Tabs
    tab1, tab2, tab3 = st.tabs(
        ["🔍 Sentiment Analyzer", "📊 Model Performance & Metrics", "💡 How It Works"]
    )

    # TAB 1: Analyzer
    with tab1:
        st.subheader("Interactive Sentiment Analyzer")
        st.markdown("Enter any statement, review, or feedback to test sentiment prediction:")

        # Quick preset buttons
        st.markdown("**Quick Preset Examples:**")
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)

        preset_text = None
        if col_p1.button("✨ Masterpiece Movie"):
            preset_text = "This film was an absolute masterpiece with stellar performances and brilliant writing!"
        if col_p2.button("⚠️ Terrible Disaster"):
            preset_text = "The plot was painfully slow, confusing, and completely unwatchable."
        if col_p3.button("🌟 Great Product"):
            preset_text = "Exceptional quality and fast delivery. Exceeded all my expectations!"
        if col_p4.button("❌ Frustrating Service"):
            preset_text = "Worst customer experience ever. Rude support and broken software."

        # Text Input
        default_value = preset_text if preset_text else "A truly delightful and heartwarming experience from start to finish!"
        user_input = st.text_area(
            "Your Input Text:",
            value=default_value,
            height=120,
            placeholder="Type or paste any text here...",
        )

        col_btn, _ = st.columns([1, 4])
        analyze_clicked = col_btn.button("🚀 Analyze Sentiment", type="primary", use_container_width=True)

        if user_input.strip():
            result = predict_sentiment(pipeline, user_input)
            label = result["label"]
            confidence = result["confidence"]
            probs = result["probabilities"]

            st.markdown("---")
            st.subheader("Prediction Result")

            col_res1, col_res2 = st.columns([1, 1])

            with col_res1:
                if label == "Positive":
                    st.success(f"### 🎉 Sentiment: **POSITIVE**")
                elif label == "Negative":
                    st.error(f"### 😞 Sentiment: **NEGATIVE**")
                else:
                    st.warning(f"### ⚖️ Sentiment: **{label}**")

                st.metric(
                    label="Confidence Score",
                    value=f"{confidence * 100:.1f}%",
                    help="Model's predicted probability for the winning class.",
                )

            with col_res2:
                st.markdown("**Class Probabilities Breakdown:**")
                prob_df = pd.DataFrame(
                    {
                        "Sentiment": ["Positive", "Negative"],
                        "Probability": [probs["Positive"], probs["Negative"]],
                    }
                )
                st.bar_chart(prob_df.set_index("Sentiment"), color="#2563EB")

            # Word / N-Gram breakdown
            st.markdown("#### 🔍 TF-IDF Keyword Insights")
            tfidf_step = pipeline.named_steps.get("tfidf")
            clf_step = pipeline.named_steps.get("classifier")

            if tfidf_step and clf_step:
                feature_names = tfidf_step.get_feature_names_out()
                vocab = tfidf_step.vocabulary_
                weights = clf_step.coef_[0]

                words = [w.lower() for w in user_input.split()]
                recognized = []
                for word in words:
                    clean_w = "".join([c for c in word if c.isalnum()])
                    if clean_w in vocab:
                        idx = vocab[clean_w]
                        w_score = weights[idx]
                        recognized.append((clean_w, float(w_score)))

                if recognized:
                    st.markdown("Words identified in model vocabulary and their sentiment influence:")
                    badge_cols = st.columns(min(len(recognized), 6))
                    for i, (word, score) in enumerate(recognized[:6]):
                        with badge_cols[i % len(badge_cols)]:
                            tag = "Positive (+)" if score > 0 else "Negative (-)"
                            st.caption(f"**{word}**")
                            st.code(f"{score:+.2f} ({tag})")
                else:
                    st.caption("No specific high-weight words found in the top vocabulary for this sentence.")

    # TAB 2: Model Performance & Metrics
    with tab2:
        st.subheader("Model Evaluation on SST-2 Benchmark")
        metrics = load_saved_metrics()

        if metrics:
            st.markdown(
                f"Evaluation results on the **SST-2 Validation Split** "
                f"({metrics.get('num_samples', metrics.get('val_sample_count', '872'))} samples):"
            )

            # Metric cards
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Accuracy", f"{metrics['accuracy'] * 100:.2f}%")
            m_col2.metric("Precision", f"{metrics['precision'] * 100:.2f}%")
            m_col3.metric("Recall", f"{metrics['recall'] * 100:.2f}%")
            m_col4.metric("F1-Score", f"{metrics['f1_score'] * 100:.2f}%")

            st.markdown("---")
            st.subheader("Confusion Matrix")

            cm = metrics["confusion_matrix"]
            col_cm1, col_cm2 = st.columns([1, 1])

            with col_cm1:
                cm_df = pd.DataFrame(
                    cm,
                    index=["Actual Negative (0)", "Actual Positive (1)"],
                    columns=["Predicted Negative (0)", "Predicted Positive (1)"],
                )
                st.dataframe(cm_df, use_container_width=True)

                st.markdown(
                    f"- **True Negatives (TN):** {cm[0][0]}\n"
                    f"- **False Positives (FP):** {cm[0][1]}\n"
                    f"- **False Negatives (FN):** {cm[1][0]}\n"
                    f"- **True Positives (TP):** {cm[1][1]}"
                )

            with col_cm2:
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.heatmap(
                    cm,
                    annot=True,
                    fmt="d",
                    cmap="Blues",
                    xticklabels=["Negative", "Positive"],
                    yticklabels=["Negative", "Positive"],
                    ax=ax,
                )
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Actual")
                ax.set_title("Confusion Matrix Heatmap")
                st.pyplot(fig)
                plt.close(fig)

        else:
            st.warning("⚠️ No saved evaluation metrics found yet.")
            st.info(
                "Run `python src/train.py` in your terminal to evaluate the model on SST-2 "
                "and generate metrics.json and the confusion matrix plot."
            )

    # TAB 3: How It Works
    with tab3:
        st.subheader("💡 NLP & Machine Learning Concepts Explained")

        st.markdown(
            """
            ### 1. What is the SST-2 Dataset?
            The **Stanford Sentiment Treebank (SST-2)** is a premier benchmark dataset widely used
            in Natural Language Processing (NLP). It contains movie review excerpts labeled as either:
            - **Positive (1)**
            - **Negative (0)**

            ### 2. How TF-IDF (Term Frequency - Inverse Document Frequency) Works
            Computers cannot read text directly; text must be converted to numbers (vectors).
            - **Term Frequency (TF):** Measures how frequently a word appears in a sentence.
            - **Inverse Document Frequency (IDF):** Penalizes common words (like 'the', 'is') and boosts
              rare, informative words (like 'masterpiece', 'horrific').
            - **N-Grams:** We use unigrams (single words, e.g., 'good') and bigrams (two words, e.g., 'not good')
              to preserve phrase-level meaning and negations.

            ### 3. How Logistic Regression Classifies Sentiment
            - Logistic Regression learns a mathematical weight for each word or phrase.
            - Positive words get positive weights; negative words get negative weights.
            - The model calculates the weighted sum of words in your sentence and passes it through
              a **Sigmoid function** to output a probability between 0% and 100%.
            - If probability > 50%, it predicts **Positive**; otherwise **Negative**.

            ### 4. Evaluation Metrics
            - **Accuracy:** The percentage of total sentences correctly classified.
            - **Precision:** Of all sentences predicted as positive, how many were truly positive?
            - **Recall:** Of all actual positive sentences, how many did the model capture?
            - **F1-Score:** The harmonic mean of precision and recall, balancing false positives and false negatives.
            - **Confusion Matrix:** A table breaking down True Positives, True Negatives, False Positives, and False Negatives.
            """
        )


if __name__ == "__main__":
    main()
