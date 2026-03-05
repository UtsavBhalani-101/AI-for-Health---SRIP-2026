#  Sleep Breathing Irregularity Detection — SRIP 2026

 Automated detection of abnormal breathing events during sleep using multi-channel physiological signals and deep learning.

---

##  Table of Contents

- [Overview](#overview)
- [Signals Used](#signals-used)
- [Project Structure](#project-structure)
- [Pipeline](#pipeline)
- [Metrics](#metrics)
- [How to Run](#how-to-run)

---

##  Overview


This project analyzes overnight physiological recordings to detect abnormal breathing events during sleep. The dataset contains multi-channel physiological signals from five participants along with annotated breathing irregularities.

Signals are preprocessed using digital filtering and segmented into overlapping time windows. Each window is labeled based on overlap with annotated breathing events. Due to extreme class imbalance across event types, the task is formulated as a **binary classification problem (Normal vs Abnormal)**.

A **1D Convolutional Neural Network (CNN)** is trained using **Leave-One-Participant-Out (LOPO) cross-validation**, ensuring subject-independent evaluation.

---

##  Signals Used

| Signal | Sampling Rate | Description |
|---|---|---|
| **Nasal Airflow** | 32 Hz | Primary airflow signal at the nose |
| **Thoracic Movement** | 32 Hz | Chest wall effort/movement signal |
| **SpO₂** | 4 Hz | Blood oxygen saturation |

---

##  Project Structure

```
internship/
│
├── Data/                        # Raw participant recordings
│
├── Dataset/                     # Windowed datasets generated from signals
│
├── Visualizations/              # PDF visualizations of signals and events
│
├── models/
│   └── cnn_model.py             # 1D CNN architecture
│
├── utils/
│   ├── cleaning.py              # Metadata cleaning functions
│   ├── io.py                    # File loading and validation utilities
│   └── signal_processing.py     # Filtering and windowing functions
│
├── Scripts/
│   ├── vis.py                   # Signal visualization pipeline
│   ├── create_dataset.py        # Dataset generation pipeline
│   └── train_model.py           # CNN training and LOPO evaluation
│
├── results/
│   └── metrics.md               # Model evaluation results
│
├── README.md
└── report.pdf
```

---

##  Pipeline

### 1. Visualization

- Plot full-night signals (Nasal Airflow, Thoracic, SpO₂)
- Overlay annotated breathing events as shaded regions
- Export visualization as a multi-page PDF

### 2. Signal Processing

- **Bandpass filter**: 0.17 – 0.4 Hz (typical human breathing frequency range)
- **Window size**: 30 seconds
- **Overlap**: 50%

### 3. Dataset Creation

- Slide windows across the full recording
- Label each window based on **>50% overlap** with annotated abnormal events
- **Binary labels**: `0` = Normal, `1` = Abnormal

- Signals are segmented into **30-second windows with 50% overlap**
- Windows are labeled based on **>50% overlap** with annotated breathing events
- Binary labels are used:
  - `0` = Normal breathing
  - `1` = Abnormal breathing event

 - Binary labeling is used to avoid extreme class imbalance between individual event types and to improve training stability.

### 4. Model Training

- **Architecture**: 1D CNN classifier
- **Evaluation strategy**: Leave-One-Participant-Out (LOPO) Cross-Validation
- Each fold tests on one unseen participant, trains on the rest

---

##  Metrics

The following metrics are tracked per fold and averaged across all participants:

| Metric | Description |
|---|---|
| **Accuracy** | Overall correct predictions |
| **Precision** | Of predicted abnormal windows, how many were truly abnormal |
| **Recall** | Of all actual abnormal windows, how many were detected |
| **F1 Score** | Harmonic mean of Precision and Recall |
| **Confusion Matrix** | Per-class breakdown of TP, FP, FN, TN |

> Full results are logged in [`results/metrics.md`](results/metrics.md).

## Results Summary

The CNN model was evaluated using Leave-One-Participant-Out cross-validation.

| Fold | Accuracy | Precision | Recall | F1 |
|-----|------|------|------|------|
| AP01 | 0.59 | 0.08 | 0.65 | 0.14 |
| AP02 | 0.59 | 0.12 | 0.62 | 0.21 |
| AP03 | 0.01 | 0.01 | 1.00 | 0.02 |
| AP04 | 0.61 | 0.14 | 0.57 | 0.22 |
| AP05 | 0.57 | 0.25 | 0.52 | 0.33 |

The model achieves high recall for abnormal breathing events but produces many false positives due to strong class imbalance and subject-level variability.

### Baseline Models

To provide a comparison, classical machine learning models were also evaluated using the same LOPO strategy.

| Model | Accuracy | Precision | Recall | F1 |
|------|------|------|------|------|
| Logistic Regression | ~0.63 | ~0.16 | ~0.68 | ~0.26 |
| XGBoost | ~0.79 | ~0.16 | ~0.34 | ~0.21 |
| CNN | ~0.47 | ~0.12 | ~0.67 | ~0.18 |

Tree-based models such as XGBoost showed stronger overall accuracy, while the CNN achieved higher recall but produced more false positives due to class imbalance.

---

##  How to Run

> **Prerequisites**: Activate your virtual environment and ensure all dependencies are installed.
> ```bash
> pip install -r requirements.txt
> ```

### 1. Visualize Signals

Plot and export the full-night physiological signals for a participant:

```bash
python Scripts/vis.py --input Data/AP01
```

> Default output: `Visualizations/` — override with `--output <path>`

### 2. Create Dataset

Generate the windowed and labeled dataset from all participant folders:

```bash
python Scripts/create_dataset.py --input Data
```

> Default output: `Dataset/` — override with `--output <path>`  
> `--filter` is optional but recommended — applies a 0.17–0.4 Hz bandpass filter to the nasal signal before windowing.

```bash
# With optional arguments
python Scripts/create_dataset.py --input Data --output Dataset/ --filter
```

### 3. Train & Evaluate Model

Run LOPO cross-validation and save metrics:

```bash
python Scripts/train_model.py --dataset_path Dataset/
```

---
