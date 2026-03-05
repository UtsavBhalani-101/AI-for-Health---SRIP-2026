# Model Evaluation Results

**Model:** Simple 1D CNN  
**Evaluation Strategy:** Leave-One-Participant-Out Cross-Validation (LOPO)  
**Task:** Binary classification — Normal (0) vs. Abnormal (1) breathing windows  
**Dataset:** 5 participants, 30-second windows with 50% overlap 

The final dataset contained approximately 8,700 windows across five participants, with abnormal breathing events representing a small fraction of the total windows.

---

## Design Decision — Binary vs. Multi-Class Labeling

The raw annotations contain multiple event types: Hypopnea, Obstructive Apnea, Mixed Apnea, and Body Events. With only 5 participants and several categories appearing extremely rarely (e.g., body events occur only a handful of times across the entire dataset), multi-class classification leads to severe class imbalance with some classes carrying too few samples to train on reliably.

LOPO cross-validation compounds this further — a rare event type present in the training participants may be completely absent in the held-out test participant, or vice versa, making metrics unstable and generalization unreliable.

The task was therefore reformulated as **binary classification**: all abnormal breathing events are merged into a single **Abnormal** class, with remaining windows labeled **Normal**. This improves training stability, reduces sensitivity to rare categories, and lets the model focus on detecting the *presence* of a breathing irregularity rather than distinguishing between uncommon subtypes — a more robust framing given the small dataset size.

---

## Per-Fold Results

| Fold | Accuracy | Precision | Recall | F1 |
|-----|------|------|------|------|
| AP01 | 0.59 | 0.08 | 0.65 | 0.14 |
| AP02 | 0.59 | 0.12 | 0.62 | 0.21 |
| AP03 | 0.01 | 0.01 | 1.00 | 0.02 |
| AP04 | 0.61 | 0.14 | 0.57 | 0.22 |
| AP05 | 0.57 | 0.25 | 0.52 | 0.33 |

---

## Confusion Matrices

### Fold 1 — Test: AP01

|                   | Predicted Normal | Predicted Abnormal |
| ----------------- | ---------------- | ------------------ |
| **True Normal**   | 1013             | 703                |
| **True Abnormal** | 33               | 61                 |


### Fold 2 — Test: AP02

|                   | Predicted Normal | Predicted Abnormal |
| ----------------- | ---------------- | ------------------ |
| **True Normal**   | 952              | 657                |
| **True Abnormal** | 56               | 93                 |


### Fold 3 — Test: AP03

|                   | Predicted Normal | Predicted Abnormal |
| ----------------- | ---------------- | ------------------ |
| **True Normal**   | 3                | 1661               |
| **True Abnormal** | 0                | 21                 |


### Fold 4 — Test: AP04

|                   | Predicted Normal | Predicted Abnormal |
| ----------------- | ---------------- | ------------------ |
| **True Normal**   | 1071             | 666                |
| **True Abnormal** | 78               | 105                |


### Fold 5 — Test: AP05

|                   | Predicted Normal | Predicted Abnormal |
| ----------------- | ---------------- | ------------------ |
| **True Normal**   | 729              | 519                |
| **True Abnormal** | 154              | 169                |


---

## Average Metrics Across All Folds

| Metric                    | Value  |
|---------------------------|--------|
| **Accuracy**              | ~0.47  |
| **Precision (Abnormal)**  | ~0.12  |
| **Recall (Abnormal)**     | ~0.67  |
| **F1-Score (Abnormal)**   | ~0.18  |

---

## Interpretation

The CNN achieves **moderate recall (~0.67)** but **low precision (~0.12)**.  
This means the model detects many abnormal breathing events but still produces a large number of false positives by incorrectly labeling normal windows as abnormal.

This behavior is largely due to the **strong class imbalance** in the dataset, where normal breathing windows significantly outnumber abnormal ones. Although class weighting was applied in the loss function, the model still tends to overpredict abnormal windows.

Performance also varies considerably across participants (accuracy ranging from 0.01 to 0.61), indicating **subject-level distribution shift** in the physiological signals. This highlights the difficulty of generalizing across individuals and is exactly the challenge that Leave-One-Participant-Out (LOPO) evaluation is designed to expose.


---

### Limitations and Possible Improvements

The dataset is small (5 participants) and strongly class-imbalanced, which limits model generalization. Future work could include using additional signals (thoracic movement and SpO₂), improved class balancing strategies, and longer training with early stopping. Incorporating multi-channel inputs and larger datasets would likely improve detection performance.


## Baseline Models (Additional Experiments)

Before training the CNN model, classical machine learning baselines were evaluated to understand how well simpler models perform on the dataset. Logistic Regression and XGBoost were trained using the same Leave-One-Participant-Out (LOPO) evaluation strategy.

Two input representations were evaluated:

1. **Raw signal windows** (960-sample nasal airflow segments)
2. **Feature-engineered representations**, where statistical summaries such as mean, standard deviation, minimum, and maximum values were computed from each window.

These models were trained on the same windowed dataset and compared against the CNN results.

| Model               | Key Observation                                                                    |
| ------------------- | ---------------------------------------------------------------------------------- |
| Logistic Regression | Captured some abnormal windows but produced many false positives                   |
| XGBoost             | Showed stronger generalization across participants compared to Logistic Regression |
| 1D CNN              | Achieved higher recall but produced many false positives due to class imbalance    |

These experiments suggest that classical models trained on feature-engineered inputs can achieve stronger overall accuracy on this small dataset, while the CNN prioritizes recall by detecting more abnormal windows at the cost of increased false positives.

| Model               | Accuracy | Precision | Recall | F1    |
| ------------------- | -------- | --------- | ------ | ----- |
| Logistic Regression | ~0.63    | ~0.16     | ~0.68  | ~0.26 |
| XGBoost             | ~0.79    | ~0.16     | ~0.34  | ~0.21 |
| CNN                 | ~0.26    | ~0.09     | ~0.79  | ~0.15 |

Multi-class labeling was also evaluated, but it performed significantly worse due to the extreme class imbalance and limited number of samples per event category.
