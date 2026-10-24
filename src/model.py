"""
Model pipeline, training, evaluation, and inference utilities for SentimentScope.
"""
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline

from src.preprocessing import clean_text

LABEL_MAP = {0: "Negative", 1: "Positive"}


def build_sentiment_pipeline(
    max_features: int = 15000,
    ngram_range: Tuple[int, int] = (1, 2),
    C: float = 1.0,
    random_state: int = 42,
) -> Pipeline:
    """
    Construct a TF-IDF + Logistic Regression classification pipeline.

    Args:
        max_features (int): Maximum vocabulary size for TF-IDF.
        ngram_range (tuple): N-gram range (min_n, max_n) to capture single words and word pairs.
        C (float): Regularization parameter for Logistic Regression.
        random_state (int): Seed for reproducibility.

    Returns:
        Pipeline: Scikit-learn Pipeline object ready to be fitted.
    """
    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    ngram_range=ngram_range,
                    max_features=max_features,
                    sublinear_tf=True,
                    strip_accents="unicode",
                    lowercase=True,
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    C=C,
                    max_iter=1000,
                    random_state=random_state,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    return pipeline


def train_model(
    train_texts: List[str],
    train_labels: List[int],
    pipeline: Pipeline = None,
) -> Pipeline:
    """
    Train the sentiment pipeline on provided texts and binary labels.

    Args:
        train_texts (list of str): Training sentences.
        train_labels (list of int): Binary labels (0 for Negative, 1 for Positive).
        pipeline (Pipeline, optional): Existing pipeline or None to create fresh.

    Returns:
        Pipeline: Fitted scikit-learn Pipeline.
    """
    if pipeline is None:
        pipeline = build_sentiment_pipeline()

    cleaned_texts = [clean_text(t) for t in train_texts]
    pipeline.fit(cleaned_texts, train_labels)
    return pipeline


def evaluate_model(
    pipeline: Pipeline,
    eval_texts: List[str],
    eval_labels: List[int],
) -> Dict[str, Any]:
    """
    Evaluate the model on a test or validation split.

    Args:
        pipeline (Pipeline): Trained scikit-learn pipeline.
        eval_texts (list of str): Input sentences for evaluation.
        eval_labels (list of int): Ground truth labels (0 or 1).

    Returns:
        dict: Evaluation metrics (accuracy, precision, recall, f1, confusion matrix).
    """
    cleaned_texts = [clean_text(t) for t in eval_texts]
    predictions = pipeline.predict(cleaned_texts)

    acc = float(accuracy_score(eval_labels, predictions))
    prec = float(precision_score(eval_labels, predictions, zero_division=0))
    rec = float(recall_score(eval_labels, predictions, zero_division=0))
    f1 = float(f1_score(eval_labels, predictions, zero_division=0))
    cm = confusion_matrix(eval_labels, predictions).tolist()

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": cm,
        "num_samples": len(eval_labels),
    }


def predict_sentiment(
    pipeline: Pipeline,
    text: str,
) -> Dict[str, Any]:
    """
    Predict sentiment and confidence score for a single input text.

    Args:
        pipeline (Pipeline): Trained scikit-learn pipeline.
        text (str): Input text sentence.

    Returns:
        dict: Prediction results with label, confidence, and class probabilities.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return {
            "label": "Neutral / Empty",
            "confidence": 0.0,
            "probabilities": {"Negative": 0.5, "Positive": 0.5},
            "error": "Empty input text provided.",
        }

    probs = pipeline.predict_proba([cleaned])[0]
    pred_idx = int(np.argmax(probs))
    label = LABEL_MAP.get(pred_idx, "Unknown")
    confidence = float(probs[pred_idx])

    return {
        "label": label,
        "confidence": confidence,
        "probabilities": {
            "Negative": float(probs[0]),
            "Positive": float(probs[1]),
        },
    }


def save_pipeline(pipeline: Pipeline, filepath: Union[str, Path]) -> Path:
    """Save the trained pipeline to disk."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    return path


def load_pipeline(filepath: Union[str, Path]) -> Pipeline:
    """Load a trained pipeline from disk."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {path}")
    return joblib.load(path)
