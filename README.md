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

- ### 3. Dataset Creation

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
| AP01 | 0.30 | 0.06 | 0.82 | 0.11 |
| AP02 | 0.09 | 0.09 | 0.99 | 0.16 |
| AP03 | 0.01 | 0.01 | 1.00 | 0.02 |
| AP04 | 0.51 | 0.11 | 0.62 | 0.19 |
| AP05 | 0.41 | 0.18 | 0.51 | 0.26 |

The model achieves high recall for abnormal breathing events but produces many false positives due to strong class imbalance and subject-level variability.

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
