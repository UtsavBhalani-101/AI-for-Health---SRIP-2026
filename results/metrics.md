# Model Evaluation Results

**Model:** Simple 1D CNN  
**Evaluation Strategy:** Leave-One-Participant-Out Cross-Validation (LOPO)  
**Task:** Binary classification — Normal (0) vs. Abnormal (1) breathing windows  
**Dataset:** 5 participants, 30-second windows with 50% overlap  

---

## Design Decision — Binary vs. Multi-Class Labeling

The raw annotations contain multiple event types: Hypopnea, Obstructive Apnea, Mixed Apnea, and Body Events. With only 5 participants and several categories appearing extremely rarely (e.g., body events occur only a handful of times across the entire dataset), multi-class classification leads to severe class imbalance with some classes carrying too few samples to train on reliably.

LOPO cross-validation compounds this further — a rare event type present in the training participants may be completely absent in the held-out test participant, or vice versa, making metrics unstable and generalization unreliable.

The task was therefore reformulated as **binary classification**: all abnormal breathing events are merged into a single **Abnormal** class, with remaining windows labeled **Normal**. This improves training stability, reduces sensitivity to rare categories, and lets the model focus on detecting the *presence* of a breathing irregularity rather than distinguishing between uncommon subtypes — a more robust framing given the small dataset size.

---

## Per-Fold Results

| Fold | Test Participant | Accuracy | Precision (Abnormal) | Recall (Abnormal) | F1 (Abnormal) |
|------|-----------------|----------|----------------------|-------------------|---------------|
| 1    | AP01            | 0.30     | 0.06                 | 0.82              | 0.11          |
| 2    | AP02            | 0.09     | 0.09                 | 0.99              | 0.16          |
| 3    | AP03            | 0.01     | 0.01                 | 1.00              | 0.02          |
| 4    | AP04            | 0.51     | 0.11                 | 0.62              | 0.19          |
| 5    | AP05            | 0.41     | 0.18                 | 0.51              | 0.26          |

---

## Confusion Matrices

### Fold 1 — Test: AP01

|                | Predicted Normal | Predicted Abnormal |
|----------------|------------------|--------------------|
| **True Normal**   | 473              | 1243               |
| **True Abnormal** | 17               | 77                 |

### Fold 2 — Test: AP02

|                | Predicted Normal | Predicted Abnormal |
|----------------|------------------|--------------------|
| **True Normal**   | 18               | 1591               |
| **True Abnormal** | 1                | 148                |

### Fold 3 — Test: AP03

|                | Predicted Normal | Predicted Abnormal |
|----------------|------------------|--------------------|
| **True Normal**   | 4                | 1660               |
| **True Abnormal** | 0                | 21                 |

### Fold 4 — Test: AP04

|                | Predicted Normal | Predicted Abnormal |
|----------------|------------------|--------------------|
| **True Normal**   | 858              | 879                |
| **True Abnormal** | 70               | 113                |

### Fold 5 — Test: AP05

|                | Predicted Normal | Predicted Abnormal |
|----------------|------------------|--------------------|
| **True Normal**   | 481              | 767                |
| **True Abnormal** | 158              | 165                |

---

## Average Metrics Across All Folds

| Metric                    | Value  |
|---------------------------|--------|
| **Accuracy**              | 0.26   |
| **Precision (Abnormal)**  | 0.09   |
| **Recall (Abnormal)**     | 0.79   |
| **F1-Score (Abnormal)**   | 0.15   |

---

## Interpretation

The CNN achieved **high recall (0.79)** but **very low precision (0.09)**. This means the model detects most abnormal breathing events but produces many false positives by incorrectly labeling normal windows as abnormal. As a result, the overall accuracy is low (0.26).

This behavior is largely due to the **strong class imbalance** in the dataset, where normal breathing windows significantly outnumber abnormal ones. Although class weighting was applied in the loss function, the model still tends to overpredict the abnormal class.

Performance also varies considerably across participants (accuracy ranging from 0.01 to 0.51), indicating **subject-level distribution shift** in the physiological signals. This highlights the difficulty of generalizing across individuals and is exactly the type of challenge that Leave-One-Participant-Out (LOPO) evaluation is designed to expose.


---

### Limitations and Possible Improvements

The dataset is small (5 participants) and strongly class-imbalanced, which limits model generalization. Future work could include using additional signals (thoracic movement and SpO₂), improved class balancing strategies, and longer training with early stopping. Incorporating multi-channel inputs and larger datasets would likely improve detection performance.

## Baseline Models (Additional Experiments)

Before training the CNN model, classical machine learning baselines were evaluated to understand how well simpler models perform on the dataset. Logistic Regression and XGBoost were trained using the same Leave-One-Participant-Out (LOPO) evaluation strategy.

These models were trained on the same windowed dataset and compared against the CNN results.

| Model               | Key Observation                                                                    |
| ------------------- | ---------------------------------------------------------------------------------- |
| Logistic Regression | Captured some abnormal windows but produced many false positives                   |
| XGBoost             | Showed stronger generalization across participants compared to Logistic Regression |
| 1D CNN              | Achieved higher recall but produced many false positives due to class imbalance    |

These experiments indicate that while deep learning models can capture temporal patterns in the signal, simpler tree-based models may generalize better under strong subject-level distribution shifts and small datasets.

| Model               | Accuracy | Precision | Recall | F1    |
| ------------------- | -------- | --------- | ------ | ----- |
| Logistic Regression | ~0.63    | ~0.16     | ~0.68  | ~0.26 |
| XGBoost             | ~0.79    | ~0.16     | ~0.34  | ~0.21 |
| CNN                 | ~0.26    | ~0.09     | ~0.79  | ~0.15 |
