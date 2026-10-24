# 🎯 SentimentScope

> **Beginner-Friendly AI Sentiment Classifier for Generative AI Hackathons**

SentimentScope is an end-to-end Natural Language Processing (NLP) machine learning application. It classifies input text as either **Positive** or **Negative** using a classic, interpretable machine learning baseline: **TF-IDF Vectorization** paired with **Logistic Regression**, trained and evaluated on the Stanford Sentiment Treebank (**SST-2**) dataset from Hugging Face.

---

## 📑 Table of Contents
1. [Project Overview](#-project-overview)
2. [Project Structure](#-project-structure)
3. [Prerequisites & Requirements](#-prerequisites--requirements)
4. [Step-by-Step Setup Guide (Windows PowerShell)](#-step-by-step-setup-guide-windows-powershell)
   - [Step 1: Create a Virtual Environment](#step-1-create-a-virtual-environment)
   - [Step 2: Activate the Virtual Environment](#step-2-activate-the-virtual-environment)
   - [Step 3: Install Project Dependencies](#step-3-install-project-dependencies)
   - [Step 4: Train and Evaluate the Baseline Model](#step-4-train-and-evaluate-the-baseline-model)
   - [Step 5: Run Automated Tests](#step-5-run-automated-tests)
   - [Step 6: Launch the Streamlit Web Application](#step-6-launch-the-streamlit-web-application)
5. [Evaluation Metrics Explained](#-evaluation-metrics-explained)
6. [Git Hygiene & Model Storage](#-git-hygiene--model-storage)
7. [Hackathon Next Steps: Bridging to Generative AI](#-hackathon-next-steps-bridging-to-generative-ai)

---

## 🔍 Project Overview

- **Task:** Binary Sentiment Classification (Positive vs. Negative).
- **Dataset:** [Stanford Sentiment Treebank (SST-2)](https://huggingface.co/datasets/stanfordnlp/sst2) via Hugging Face.
- **Model Baseline:** 
  - **Feature Extraction:** TF-IDF (Term Frequency - Inverse Document Frequency) with unigrams and bigrams (`(1, 2)`).
  - **Classifier:** Logistic Regression with L-BFGS solver.
- **Web Interface:** Interactive Streamlit web app with real-time confidence scores, class probabilities, and metric dashboards.
- **Metrics Evaluated:** Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.

---

## 📁 Project Structure

```text
SentimentScope/
├── .gitignore               # Keeps models, checkpoints, and datasets out of Git
├── README.md                # Project documentation and step-by-step guide
├── requirements.txt         # Python package dependencies
├── app.py                   # Streamlit interactive web application
├── src/
│   ├── __init__.py          # Package initialization
│   ├── preprocessing.py     # Text cleaning and normalization
│   ├── model.py             # TF-IDF pipeline, training, evaluation, inference
│   └── train.py             # SST-2 dataset loader, model trainer, metrics exporter
├── tests/
│   └── test_pipeline.py     # Unit tests for preprocessing, pipeline, and inference
└── models/                  # Generated artifacts (ignored by Git)
    ├── sentiment_pipeline.joblib  # Saved trained model pipeline
    ├── metrics.json               # Recorded evaluation scores
    └── confusion_matrix.png       # Visual confusion matrix heatmap
```

---

## 🛠️ Prerequisites & Requirements

- **Operating System:** Windows 10/11
- **Shell:** Windows PowerShell
- **Python:** Python 3.9, 3.10, 3.11, or 3.12 installed on your system.

---

## 💻 Step-by-Step Setup Guide (Windows PowerShell)

Open **Windows PowerShell** in the project directory (`C:\Users\ACS\SentimentScope`) and run the following exact commands:

### Step 1: Create a Virtual Environment

A virtual environment keeps your project dependencies isolated and clean:

```powershell
python -m venv .venv
```

---

### Step 2: Activate the Virtual Environment

Enable script execution for your session and activate the environment:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

> *(You should now see `(.venv)` in your PowerShell prompt).*

---

### Step 3: Install Project Dependencies

Upgrade `pip` and install all necessary packages from `requirements.txt`:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

### Step 4: Train and Evaluate the Baseline Model

Run the training pipeline. This script downloads the SST-2 dataset from Hugging Face (`stanfordnlp/sst2`), trains the TF-IDF + Logistic Regression model, computes validation metrics, and saves the artifacts to `models/`:

```powershell
python src/train.py
```

**Expected outputs in `models/`:**
- `sentiment_pipeline.joblib`: Serialized trained model pipeline.
- `metrics.json`: Accuracy, Precision, Recall, F1-Score, and Confusion Matrix data.
- `confusion_matrix.png`: Heatmap plot comparing predicted vs actual labels.

---

### Step 5: Run Automated Tests

Verify that text preprocessing, model pipeline training, and prediction functions work as expected:

```powershell
pytest
```

---

### Step 6: Launch the Streamlit Web Application

Start the web interface in your browser:

```powershell
streamlit run app.py
```

Streamlit will print a local URL (e.g., `http://localhost:8501`). Open this URL in your web browser to interact with SentimentScope!

---

## 📊 Evaluation Metrics Explained

When evaluating binary text classification on SST-2:

| Metric | What it Measures | Why it Matters |
| :--- | :--- | :--- |
| **Accuracy** | Overall proportion of correct predictions: `(TP + TN) / Total` | Quick high-level gauge of performance across balanced classes. |
| **Precision** | Positive predictions that were truly positive: `TP / (TP + FP)` | Measures how trustworthy a "Positive" sentiment prediction is. |
| **Recall** | Actual positive sentences that were found: `TP / (TP + FN)` | Measures if the model misses positive reviews. |
| **F1-Score** | Harmonic mean of Precision and Recall: `2 * (P * R) / (P + R)` | Balances false positives and false negatives into a single score. |
| **Confusion Matrix** | 2x2 grid displaying TN, FP, FN, and TP | Visualizes exact error distributions between classes. |

---

## 🛡️ Git Hygiene & Model Storage

In machine learning projects, binary weights and datasets should not be committed to GitHub because:
1. Datasets and model checkpoints can be hundreds of megabytes or gigabytes.
2. Git tracks text diffs, not binary blobs.
3. Models can easily be regenerated from reproducible training code.

The `.gitignore` in this repository automatically excludes:
- `models/*.joblib`, `models/*.pkl`, `models/*.bin`
- `data/`
- `.venv/`
- `*.png`, `*.log`

---

## 🚀 Hackathon Next Steps: Bridging to Generative AI

This project serves as your foundation baseline. In your hackathon presentation:
1. **Baseline Comparison:** Compare your TF-IDF + Logistic Regression accuracy against fine-tuned Transformer models (like DistilBERT or RoBERTa) or LLM zero-shot prompts.
2. **Speed & Efficiency:** Highlight how TF-IDF + Logistic Regression runs in milliseconds with near-zero CPU and memory overhead compared to LLMs.
3. **Hybrid Architectures:** Combine Logistic Regression confidence with LLM reasoning for ambiguous sentences.
