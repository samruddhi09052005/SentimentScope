"""
Unit tests for SentimentScope preprocessing, model pipeline, and inference.
"""
from pathlib import Path
import pytest
from src.preprocessing import clean_text
from src.model import (
    build_sentiment_pipeline,
    evaluate_model,
    predict_sentiment,
    save_pipeline,
    load_pipeline,
)


def test_clean_text_normalizes_whitespace():
    raw = "   This   is   a   \n\t test sentence.   "
    expected = "This is a test sentence."
    assert clean_text(raw) == expected


def test_clean_text_handles_none_and_empty():
    assert clean_text(None) == ""
    assert clean_text("") == ""
    assert clean_text("     ") == ""


def test_pipeline_architecture():
    pipeline = build_sentiment_pipeline()
    assert "tfidf" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps


def test_train_and_predict_toy_dataset():
    texts = [
        "I love this product, it is amazing!",
        "Brilliant, wonderful, best experience ever.",
        "I hate this, terrible and awful.",
        "Worst thing ever, completely broken and horrible.",
    ]
    labels = [1, 1, 0, 0]

    pipeline = build_sentiment_pipeline()
    pipeline.fit(texts, labels)

    # Positive test
    pos_res = predict_sentiment(pipeline, "I love this wonderful thing!")
    assert pos_res["label"] == "Positive"
    assert 0.0 <= pos_res["confidence"] <= 1.0
    assert "Positive" in pos_res["probabilities"]
    assert "Negative" in pos_res["probabilities"]

    # Negative test
    neg_res = predict_sentiment(pipeline, "I hate this terrible horrible mess.")
    assert neg_res["label"] == "Negative"
    assert 0.0 <= neg_res["confidence"] <= 1.0


def test_evaluate_model_metrics():
    texts = ["good", "great", "bad", "terrible"]
    labels = [1, 1, 0, 0]

    pipeline = build_sentiment_pipeline()
    pipeline.fit(texts, labels)

    metrics = evaluate_model(pipeline, texts, labels)
    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1_score" in metrics
    assert "confusion_matrix" in metrics

    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["f1_score"] <= 1.0
    assert len(metrics["confusion_matrix"]) == 2
    assert len(metrics["confusion_matrix"][0]) == 2


def test_empty_input_prediction():
    pipeline = build_sentiment_pipeline()
    pipeline.fit(["good", "bad"], [1, 0])

    res = predict_sentiment(pipeline, "   ")
    assert "Neutral" in res["label"] or "Empty" in res["label"]
    assert res["confidence"] == 0.0


def test_save_and_load_pipeline(tmp_path: Path):
    texts = ["happy", "delight", "sad", "angry"]
    labels = [1, 1, 0, 0]

    pipeline = build_sentiment_pipeline()
    pipeline.fit(texts, labels)

    save_path = tmp_path / "test_model.joblib"
    save_pipeline(pipeline, save_path)
    assert save_path.exists()

    loaded = load_pipeline(save_path)
    pred_orig = pipeline.predict(["happy"])[0]
    pred_loaded = loaded.predict(["happy"])[0]
    assert pred_orig == pred_loaded
