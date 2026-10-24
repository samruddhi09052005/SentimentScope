"""
Training script for SentimentScope.
Loads SST-2 from Hugging Face, trains TF-IDF + Logistic Regression,
evaluates metrics, and saves the trained pipeline and evaluation report.
"""
import json
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns

from src.model import (
    build_sentiment_pipeline,
    evaluate_model,
    save_pipeline,
    train_model,
)

# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = MODELS_DIR / "sentiment_pipeline.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"
CM_PLOT_PATH = MODELS_DIR / "confusion_matrix.png"


def load_sst2_data():
    """
    Load the Stanford Sentiment Treebank (SST-2) dataset.
    Prioritizes Hugging Face 'stanfordnlp/sst2'. Falls back to local/synthetic
    data if the machine has no active internet connection.
    """
    print("[1/4] Loading SST-2 dataset from Hugging Face ('stanfordnlp/sst2')...")
    try:
        from datasets import load_dataset

        dataset = load_dataset("stanfordnlp/sst2")
        train_texts = dataset["train"]["sentence"]
        train_labels = dataset["train"]["label"]
        val_texts = dataset["validation"]["sentence"]
        val_labels = dataset["validation"]["label"]

        print(
            f"  Successfully loaded SST-2 from Hugging Face!"
            f"\n  - Training samples: {len(train_texts):,}"
            f"\n  - Validation samples: {len(val_texts):,}"
        )
        return train_texts, train_labels, val_texts, val_labels, "Hugging Face stanfordnlp/sst2"

    except Exception as exc:
        print(f"  Warning: Could not fetch from Hugging Face ({exc}).")
        print("  Using local starter sentiment dataset as fallback...")

        # Curated fallback dataset for offline/restricted environments
        train_texts = [
            "This movie was an absolute masterpiece with stellar performances.",
            "I thoroughly enjoyed every minute of this brilliant film.",
            "A breathtaking visual journey with inspiring storytelling.",
            "Outstanding direction, charming characters, and a wonderful score.",
            "Easily one of the best films of the decade, truly remarkable.",
            "Heartwarming, clever, and endlessly entertaining.",
            "Superb acting and thoughtful writing that touches the heart.",
            "A delightful experience from beginning to end.",
            "An utterly dreadful waste of time and talent.",
            "The plot was confusing, boring, and completely nonsensical.",
            "Terrible acting, awful pacing, and painfully dull dialogue.",
            "I hated every second of this tedious, uninspired disaster.",
            "Worst movie I have watched all year, do not waste your time.",
            "Disappointing, sluggish, and completely devoid of charm or humor.",
            "A total mess with zero redeeming qualities.",
            "Poorly directed, poorly written, and utterly forgettable.",
        ] * 40
        train_labels = ([1] * 8 + [0] * 8) * 40

        val_texts = [
            "Incredible acting and fantastic visual storytelling.",
            "A deeply touching and brilliantly written production.",
            "An enjoyable and delightful movie for everyone.",
            "Great humor and uplifting themes throughout.",
            "Painfully boring and poorly constructed from start to finish.",
            "A disappointing disaster that fails on every level.",
            "Horrible script and miserable acting.",
            "Unbelievably dull and completely unwatchable.",
        ] * 10
        val_labels = ([1] * 4 + [0] * 4) * 10

        return train_texts, train_labels, val_texts, val_labels, "Local Starter Fallback"


def save_confusion_matrix_plot(cm, output_path: Path):
    """Generate and save a clean confusion matrix heatmap plot."""
    try:
        plt.figure(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Negative (0)", "Positive (1)"],
            yticklabels=["Negative (0)", "Positive (1)"],
        )
        plt.title("Confusion Matrix - SST-2 Validation Split")
        plt.xlabel("Predicted Sentiment")
        plt.ylabel("Actual Sentiment")
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        plt.close()
        print(f"  Confusion matrix plot saved to: {output_path}")
    except Exception as e:
        print(f"  Could not generate confusion matrix plot: {e}")


def main():
    print("=" * 60)
    print("SentimentScope - Model Training & Evaluation Pipeline")
    print("=" * 60)

    # 1. Load Data
    train_texts, train_labels, val_texts, val_labels, source = load_sst2_data()

    # 2. Build and Train Pipeline
    print("\n[2/4] Training TF-IDF + Logistic Regression model...")
    pipeline = build_sentiment_pipeline(
        max_features=15000,
        ngram_range=(1, 2),
        C=1.0,
        random_state=42,
    )
    pipeline = train_model(train_texts, train_labels, pipeline=pipeline)
    print("  Model training completed successfully.")

    # 3. Evaluate on Validation Set
    print("\n[3/4] Evaluating model performance on validation set...")
    metrics = evaluate_model(pipeline, val_texts, val_labels)
    metrics["dataset_source"] = source
    metrics["train_sample_count"] = len(train_texts)
    metrics["val_sample_count"] = len(val_texts)

    print("-" * 40)
    print(f"  Accuracy : {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  Recall   : {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"  F1-Score : {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    print("  Confusion Matrix:")
    print(f"    [[TN={metrics['confusion_matrix'][0][0]}, FP={metrics['confusion_matrix'][0][1]}],")
    print(f"     [FN={metrics['confusion_matrix'][1][0]}, TP={metrics['confusion_matrix'][1][1]}]]")
    print("-" * 40)

    # 4. Save Artifacts
    print("\n[4/4] Saving model artifacts and reports...")
    save_pipeline(pipeline, MODEL_PATH)
    print(f"  Model saved to: {MODEL_PATH}")

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to: {METRICS_PATH}")

    save_confusion_matrix_plot(metrics["confusion_matrix"], CM_PLOT_PATH)

    print("\nTraining and evaluation finished successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
