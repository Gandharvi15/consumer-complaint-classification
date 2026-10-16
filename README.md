# Consumer Financial Complaint Classification

An end-to-end NLP and machine learning pipeline to classify unstructured financial consumer complaint narratives into their corresponding financial product categories.

---

## Overview

Financial institutions and regulatory bodies receive high volumes of consumer complaints daily. Manually sorting these complaints into the correct departments is time-consuming and prone to routing errors. 

This project processes raw textual complaints, extracts features using TF-IDF (and TruncatedSVD for dense embeddings), evaluates multiple classification algorithms, and implements an inference script for real-time predictions[cite: 1, 2].

---

## Dataset

* **Source**: Consumer Financial Protection Bureau (CFPB) Complaint Database.
* **Dataset Link**: https://www.consumerfinance.gov/data-research/consumer-complaints/
* **Input Features**: `Consumer complaint narrative` (unstructured free-form text).
* **Target Label**: `Product` (18 financial product categories, e.g., Debt collection, Mortgage, Credit reporting).
* **Preprocessed Samples**: 367,123 valid complaint records (cleaned of null narratives and duplicates).

---

## Pipeline Architecture

```text
Raw Complaint Text 
       │
       ▼
Data Cleaning (Deduplication, Drop Missing)
       │
       ▼
Stratified Train/Test Split (80/20)
       │
       ▼
Feature Extraction (TF-IDF Vectorizer, max_features=30,000, n-grams=(1,2))
       │
       ├─────────────────────────────────┬─────────────────────────────────┐
       ▼                                 ▼                                 ▼
Sparse Models (TF-IDF)            Dimensionality Reduction           Dense Models (SVD)
- Logistic Regression             (TruncatedSVD, n=300)              - K-Nearest Neighbors
- SGD Classifier                                                     - XGBoost Classifier
- Balanced SGD Classifier
```

Model Performance & Evaluation

Models were evaluated on a stratified test set of **73,425 samples** using **Macro F1-Score** to account for significant class imbalance across the 18 product categories.

| Model | Macro F1 | Accuracy |
| :--- | :---: | :---: |
| **Logistic Regression** | **0.5639** | **76.77%** |
| Balanced SGD Classifier | 0.5220 | 70.18% |
| XGBoost Classifier | 0.4750 | 70.57% |
| K-Nearest Neighbors (KNN) | 0.4406 | 64.74% |
| SGD Classifier (Default) | 0.3398 | 67.57% |



> **Selected Model**: **Logistic Regression** achieved the highest Macro F1 score and overall accuracy, performing reliably across both high-frequency classes (Debt Collection, Mortgage) and lower-frequency categories.

---

Technologies UsedLanguage: Python 3.10+   Data Processing: Pandas, NumPy   Machine Learning & NLP: Scikit-learn, XGBoost,Visualization: Matplotlib, Model Serialization: Joblib

