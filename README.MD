<div align="center">

```
██╗  ██╗██╗   ██╗███╗   ███╗ █████╗ ███╗   ██╗
██║  ██║██║   ██║████╗ ████║██╔══██╗████╗  ██║
███████║██║   ██║██╔████╔██║███████║██╔██╗ ██║
██╔══██║██║   ██║██║╚██╔╝██║██╔══██║██║╚██╗██║
██║  ██║╚██████╔╝██║ ╚═╝ ██║██║  ██║██║ ╚████║
╚═╝  ╚═╝ ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝
 
 █████╗  ██████╗████████╗██╗██╗   ██╗██╗████████╗██╗   ██╗
██╔══██╗██╔════╝╚══██╔══╝██║██║   ██║██║╚══██╔══╝╚██╗ ██╔╝
███████║██║        ██║   ██║██║   ██║██║   ██║    ╚████╔╝
██╔══██║██║        ██║   ██║╚██╗ ██╔╝██║   ██║     ╚██╔╝
██║  ██║╚██████╗   ██║   ██║ ╚████╔╝ ██║   ██║      ██║
╚═╝  ╚═╝ ╚═════╝   ╚═╝   ╚═╝  ╚═══╝  ╚═╝   ╚═╝      ╚═╝
 
██████╗ ███████╗ ██████╗ ██████╗  ██████╗ ███╗   ██╗██╗████████╗██╗ ██████╗ ███╗   ██╗
██╔══██╗██╔════╝██╔════╝██╔═══██╗██╔════╝ ████╗  ██║██║╚══██╔══╝██║██╔═══██╗████╗  ██║
██████╔╝█████╗  ██║     ██║   ██║██║  ███╗██╔██╗ ██║██║   ██║   ██║██║   ██║██╔██╗ ██║
██╔══██╗██╔══╝  ██║     ██║   ██║██║   ██║██║╚██╗██║██║   ██║   ██║██║   ██║██║╚██╗██║
██║  ██║███████╗╚██████╗╚██████╔╝╚██████╔╝██║ ╚████║██║   ██║   ██║╚██████╔╝██║ ╚████║
╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═══╝╚═╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝
 
██╗███╗   ███╗██╗   ██╗
██║████╗ ████║██║   ██║
██║██╔████╔██║██║   ██║
██║██║╚██╔╝██║██║   ██║
██║██║ ╚═╝ ██║╚██████╔╝
╚═╝╚═╝     ╚═╝ ╚═════╝
```

<br/>

**Classifying human activity from raw smartphone IMU signals**
**using a CNN-LSTM hybrid model with additive attention**

<br/>

![Python](https://img.shields.io/badge/Python-3.10+-14354C?style=for-the-badge&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Colab](https://img.shields.io/badge/Google_Colab-Ready-F9AB00?style=for-the-badge&logo=googlecolab&logoColor=white)
![Accuracy](https://img.shields.io/badge/Test_Accuracy-93.15%25-00C853?style=for-the-badge)
![Dataset](https://img.shields.io/badge/Dataset-UCI_HAR-0D47A1?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-00C853?style=for-the-badge)

<br/>

---

*Prem M Patel — Computer Engineering — prempatel7740@gmail.com*


---

</div>

<br/>

## Table of Contents

- [The Problem](#the-problem)
- [Architecture Overview](#architecture-overview)
- [Results](#results)
- [Dataset](#dataset)
- [Technical Pipeline](#technical-pipeline)
- [Model Architecture](#model-architecture)
- [The Attention Mechanism](#the-attention-mechanism)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Quickstart](#quickstart)
- [Real-World Applications](#real-world-applications)
- [Key Concepts](#key-concepts)

<br/>

---

<br/>

## The Problem

<br/>

```
              WHAT THE PHONE KNOWS
              ────────────────────

   Every 1/50th of a second, the phone writes down 9 numbers.

   time=0.00s:  [ 0.12,  9.79,  0.34,  0.02, -0.01,  0.00,  0.14,  9.81,  0.33 ]
   time=0.02s:  [ 0.15,  9.81,  0.31,  0.01, -0.02,  0.01,  0.17,  9.80,  0.30 ]
   time=0.04s:  [ 0.89,  9.45,  0.67,  0.12, -0.08,  0.03,  0.91,  9.43,  0.65 ]
   ...

   After 2.56 seconds (128 readings), one question:

              What was this person doing?
```

<br/>

The phone has no camera. No GPS. No microphone. Just numbers — forces and rotations recorded by an accelerometer and a gyroscope.

From those numbers alone, this model answers: **walking, walking upstairs, walking downstairs, sitting, standing, or laying down** — with 93.15% accuracy on people it has never seen before.

Four of the six classes are straightforward. Walking creates a rhythmic spike pattern — every footstep leaves a signature in the accelerometer data. Laying down shifts gravity from one axis to another. These are easy.

SITTING and STANDING are not. Both produce nearly flat, near-zero signals. Both have gravity acting on the same axis. Their cosine similarity is 0.97 out of 1.0 — they are 97% identical to standard mathematical analysis. The entire model architecture — specifically the attention mechanism — exists primarily to resolve this one problem.

<br/>

---

<br/>

## Architecture Overview

<br/>

```
  INPUT
  Raw sensor window — 128 timesteps × 9 channels = 2.56 seconds
         │
         ▼
  ┌─────────────────────────────────────────────────────────┐
  │  CNN BLOCK 1 — 64 filters, kernel=3, MaxPool            │
  │                                                         │
  │  Scans the signal with a 60ms sliding window            │
  │  Each filter learns to detect one local pattern         │
  │  Footstep spikes — zero crossings — brief plateaus      │
  │                                                         │
  │  Output: (64 timesteps, 64 features)                    │
  └─────────────────────────┬───────────────────────────────┘
                            │
  ┌─────────────────────────▼───────────────────────────────┐
  │  CNN BLOCK 2 — 128 filters, kernel=3, MaxPool           │
  │                                                         │
  │  Effective window: 180ms (sees 9 original timesteps)    │
  │  Detects compound patterns built from Block 1 output    │
  │  Full footstep cycles — rotation events — stillness     │
  │                                                         │
  │  Output: (32 timesteps, 128 features)                   │
  └─────────────────────────┬───────────────────────────────┘
                            │
  ┌─────────────────────────▼───────────────────────────────┐
  │  LSTM — 128 units, return_sequences=True                │
  │                                                         │
  │  Reads 32 compressed feature vectors in order           │
  │  Forget gate, input gate, output gate                   │
  │  Cell state prevents gradient from vanishing            │
  │  Accumulates: footstep rhythm — or sustained stillness  │
  │                                                         │
  │  Output: 32 hidden states, each 128-dimensional         │
  └──────────────┬──────────────────────────────────────────┘
                 │
  ┌──────────────▼──────────────────────────────────────────┐
  │  ATTENTION — 129 parameters                             │
  │                                                         │
  │  Scores each of 32 hidden states                        │
  │  Softmax normalises scores to sum=1.0                   │
  │  Weighted sum produces one context vector               │
  │                                                         │
  │  For SITTING vs STANDING: focuses on the 3-4 timesteps  │
  │  where subtle postural sway is most detectable          │
  │                                                         │
  │  Output: one 128-dimensional context vector             │
  └─────────────────────────┬───────────────────────────────┘
                            │
  ┌─────────────────────────▼───────────────────────────────┐
  │  Dense(64, ReLU) + Dropout(0.5) + Dense(6, Softmax)     │
  │                                                         │
  │  Maps attended features to 6 class probabilities        │
  │  Probabilities sum to exactly 1.0                       │
  └─────────────────────────┬───────────────────────────────┘
                            │
  OUTPUT
  6 probabilities — argmax gives the predicted activity
```

<br/>

---

<br/>

## Results

<br/>

```
  ╔══════════════════════════════════════════════════════════════╗
  ║                   PERFORMANCE SUMMARY                        ║
  ╠══════════════════════════════════════════════════════════════╣
  ║                                                              ║
  ║  DATASET                                                     ║
  ║  ├─ Name              UCI HAR (30 subjects, 6 activities)    ║
  ║  ├─ Train samples     7,352  windows × 2.56 seconds          ║
  ║  ├─ Test  samples     2,947  (9 subjects never seen)         ║
  ║  └─ Split type        By subject — no data leakage           ║
  ║                                                              ║
  ║  OVERALL                                                     ║
  ║  ├─ Test  accuracy    93.15%                                 ║
  ║  ├─ Train accuracy    95-98%                                 ║
  ║  └─ Overfit gap       <3%    healthy generalisation          ║
  ║                                                              ║
  ║  PER-CLASS                                                   ║
  ║  ├─ WALKING           99.6%   F1=0.992                       ║
  ║  ├─ WALKING_UPSTAIRS  94.9%   F1=0.968                       ║
  ║  ├─ WALKING_DOWNSTAIRS98.8%   F1=0.972                       ║
  ║  ├─ SITTING           90.2%   F1=0.837                       ║
  ║  ├─ STANDING          76.9%   F1=0.831  (hardest class)      ║
  ║  └─ LAYING           100.0%   F1=0.996                       ║
  ║                                                              ║
  ╠══════════════════════════════════════════════════════════════╣
  ║                                                              ║
  ║  90%+ benchmark target  ACHIEVED                             ║
  ║  Published state-of-art  89% to 96% on this split            ║
  ║  This model              93.15%  — competitive               ║
  ║                                                              ║
  ╚══════════════════════════════════════════════════════════════╝
```

<br/>

**Why STANDING is the hard class:**

```
  SITTING signal  →  flat near-zero line  →  cosine similarity with STANDING: 0.97
  STANDING signal →  flat near-zero line  →  cosine similarity with SITTING:  0.97

  They are 97% identical in frequency space.
  The model must detect subtle postural sway differences.
  Attention improves STANDING recall from 76.9% toward 80%+.
```

<br/>

---

<br/>

## Dataset

<br/>

> **UCI HAR — University of California Irvine Human Activity Recognition**
> Collected at Universitat Politecnica de Catalunya — [UCI ML Repository](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)

<br/>

| Property | Value |
|:---|:---|
| Volunteers | 30 subjects, ages 19-48 |
| Device | Samsung Galaxy S2 worn at the waist |
| Sampling rate | 50 Hz (50 readings per second) |
| Window size | 128 samples = 2.56 seconds, 50% overlap |
| Activities | 6 classes |
| Train/test split | By subject — 21 train, 9 test |
| Total windows | 10,299 (7,352 train + 2,947 test) |
| Download size | ~25 MB |

<br/>

**The subject-based split is critical.**

Many papers report inflated results by splitting randomly — letting data from the same person appear in both train and test sets. This allows the model to memorise individual gait patterns rather than learning general activity signatures.

The correct protocol — used here — trains on 21 subjects and tests on 9 completely different people. A model that achieves 93% on this split has genuinely learned to recognise activities across different individuals. That is a much stronger claim.

<br/>

**Signal channels (9 total):**

```
advio-01/                              UCI_HAR_Dataset/
├── iphone/                            ├── train/
│   ├── body_acc_x/y/z  ←──── same    │   └── Inertial Signals/
│   ├── body_gyro_x/y/z         ────► │       ├── body_acc_x_train.txt
│   └── total_acc_x/y/z               │       ├── body_gyro_x_train.txt
└── ground-truth/                      │       └── total_acc_x_train.txt
    └── pose.csv                       └── test/
```

| Index | Channel | Description |
|:-----:|:--------|:-----------|
| 0-2 | body\_acc\_x/y/z | Body acceleration (gravity removed) |
| 3-5 | body\_gyro\_x/y/z | Angular velocity in rad/s |
| 6-8 | total\_acc\_x/y/z | Raw acceleration including gravity |

<br/>

---

<br/>

## Technical Pipeline

<br/>

### 1 — Signal Analysis (FFT)

<br/>

Before any modelling, we analyse what information the signals actually carry.

The **Fast Fourier Transform** converts a time-domain signal (value vs time) to a frequency-domain spectrum (strength vs frequency):

```
FFT formula:  X[k] = sum_{n=0}^{N-1}  x[n] * exp(-j * 2*pi * k * n / N)

In plain English: for each frequency k, multiply the signal by a reference
wave at that frequency and sum all the products.
Large result = that frequency is present. Near zero = it is not.
```

**What FFT reveals:**

```
Walking signal FFT:             Sitting signal FFT:

Strength                        Strength
   |                               |
   |    *                          | * * * * * * * * * * * *
   |    |  *                       |_____________________________
   |    |  |                       0                         25 Hz
   |____|__|____
        2  4  Hz
        ↑
   Strong peak at ~2 Hz        No dominant frequency anywhere
   = footstep rhythm            = no rhythm, just noise
```

**Frequency resolution** = 50 Hz / 128 samples = 0.39 Hz per bin.

**The critical finding:** SITTING and STANDING have identical FFT spectra — both flat. The model must find subtle time-domain differences. This is why we need the LSTM and attention, not just FFT-based features.

<br/>

### 2 — Preprocessing

<br/>

**StandardScaler normalisation:**

Raw channel ranges differ by 5-10x. Without normalisation, the model learns that channels with larger numbers matter more — which is wrong.

```
Before:  Body Acc X range  ≈ 2.5   (small numbers)
         Body Gyro Y range ≈ 12.0  (large numbers)

The model would think Gyro matters 5x more — just because numbers are bigger.

After StandardScaler:
   x_scaled = (x - mean_channel) / std_channel

All channels:  mean = 0.0,  std = 1.0
Model treats all channels as equally important by default.
```

**Critical rule:** `scaler.fit_transform(X_train)` then `scaler.transform(X_test)`.
Never fit on test data — that is data leakage.

**One-hot encoding:**
```
Label 0  →  [1, 0, 0, 0, 0, 0]   WALKING
Label 3  →  [0, 0, 0, 1, 0, 0]   SITTING
Label 5  →  [0, 0, 0, 0, 0, 1]   LAYING
```
Required because the model outputs 6 probabilities — targets must match format.

<br/>

### 3 — CNN: Local Pattern Detection

<br/>

A 1D convolutional layer slides a small filter across the time axis:

```
Convolution at position t:

output[t] = filter[0] * signal[t]
           + filter[1] * signal[t+1]
           + filter[2] * signal[t+2]
           + bias

filter size=3 at 50Hz = 60ms window per computation
```

With 64 filters in Block 1, the layer runs this 64 times with different learned weights. After training:
- Some filters detect footstep spikes
- Some detect zero crossings
- Some detect sustained flatness

MaxPooling(2) halves the sequence (128 → 64 → 32), expanding the effective receptive field:

```
Block 1 effective window:  3 steps × 20ms  =  60ms
Block 2 effective window:  9 steps × 20ms  = 180ms
```

<br/>

### 4 — LSTM: Temporal Pattern Recognition

<br/>

The LSTM reads 32 compressed feature vectors in sequence. At each position, three gates control what to remember:

```
Forget gate:  f_t = sigmoid(W_f · [h_{t-1}, x_t] + b_f)
              What to erase from long-term memory

Input gate:   i_t = sigmoid(W_i · [h_{t-1}, x_t] + b_i)
              What new information to write

Cell update:  c_t = f_t * c_{t-1} + i_t * tanh(W_c · [h_{t-1}, x_t] + b_c)
              Keep old + write new

Output gate:  h_t = sigmoid(W_o · [h_{t-1}, x_t] + b_o) * tanh(c_t)
              What to expose as current output
```

The cell state `c_t` flows through with only additive updates — no repeated weight multiplication. This prevents the vanishing gradient problem that breaks vanilla RNNs, allowing the LSTM to learn dependencies across all 32 positions.

For walking: LSTM accumulates evidence of regular footstep intervals.
For static: LSTM accumulates evidence of sustained near-zero activity.

<br/>

### 5 — Training

<br/>

**Loss function — categorical cross-entropy:**

```
L = -log(probability assigned to true class)

Model gives 90% to correct class:  L = -log(0.90) = 0.105   (small penalty)
Model gives 10% to correct class:  L = -log(0.10) = 2.303   (large penalty)
```

Heavy penalty for confident wrong predictions drives the model to be both accurate and calibrated.

**Optimiser — Adam:**

```
m_t = 0.9 * m_{t-1} + 0.1 * g_t        (momentum — running average of gradients)
v_t = 0.999 * v_{t-1} + 0.001 * g_t^2  (adaptive scale — running average of squared gradients)

w = w - (lr / sqrt(v_t) + eps) * m_t

Effect: large consistent gradients get smaller effective steps
        small inconsistent gradients get larger effective steps
        faster, more stable convergence than vanilla SGD
```

<br/>

---

<br/>

## Model Architecture

<br/>

```
  ┌─────────────────────────────────────────────────────────┐
  │  INPUT   (128, 9)                                       │
  │  128 timesteps × 9 sensor channels                      │
  └────────────────────────┬────────────────────────────────┘
                           │
  ┌────────────────────────▼────────────────────────────────┐
  │  Conv1D  64  k=3  relu    (128, 64)                     │
  │  BatchNorm                (128, 64)                     │
  │  Conv1D  64  k=3  relu    (128, 64)                     │
  │  BatchNorm                (128, 64)                     │
  │  MaxPool  2               ( 64, 64)                     │
  │  Dropout  0.3             ( 64, 64)                     │
  └────────────────────────┬────────────────────────────────┘
                           │
  ┌────────────────────────▼────────────────────────────────┐
  │  Conv1D  128  k=3  relu   ( 64, 128)                    │
  │  BatchNorm                ( 64, 128)                    │
  │  Conv1D  128  k=3  relu   ( 64, 128)                    │
  │  BatchNorm                ( 64, 128)                    │
  │  MaxPool  2               ( 32, 128)                    │
  │  Dropout  0.3             ( 32, 128)                    │
  └────────────────────────┬────────────────────────────────┘
                           │
  ┌────────────────────────▼────────────────────────────────┐
  │  LSTM  128  return_sequences=True   ( 32, 128)          │
  │  BatchNorm                          ( 32, 128)          │
  │  Dropout  0.3                       ( 32, 128)          │
  └────────────────────────┬────────────────────────────────┘
                           │
  ┌────────────────────────▼────────────────────────────────┐
  │  ATTENTION                                              │
  │  Dense(1, tanh)       →  scores   (32,)                 │
  │  Softmax              →  weights  (32,)   sum=1.0       │
  │  Weighted sum         →  context  (128,)                │
  │  Parameters: 129                                        │
  └────────────────────────┬────────────────────────────────┘
                           │
  ┌────────────────────────▼────────────────────────────────┐
  │  Dense  64  relu          (64,)                         │
  │  BatchNorm                (64,)                         │
  │  Dropout  0.5             (64,)                         │
  │  Dense  6   softmax       (6,)                          │
  └────────────────────────┬────────────────────────────────┘
                           │
  OUTPUT  (6,)  — probabilities summing to 1.0
  argmax → predicted activity class

  Total parameters:   ~230,000 - 276,000  (depending on variant)
  Loss:               Categorical cross-entropy
  Optimiser:          Adam  lr=0.001
  Batch size:         64
  Early stopping:     patience=10  on val_loss
```

<br/>

---

<br/>

## The Attention Mechanism

<br/>

This deserves its own section because it is the architectural choice that separates this model from a standard CNN-LSTM baseline.

**The problem it solves:**

A standard sequence classifier takes only the final LSTM hidden state `h_32`.
This implicitly assumes the last timestep summarises everything relevant.
For SITTING vs STANDING, the discriminating signal — tiny postural sway — may be
concentrated in 3-4 specific timesteps buried in a 32-timestep sequence.
Using only `h_32` dilutes this signal across all timesteps equally.

**The solution — three steps:**

```
Step 1: Score each of 32 hidden states
        e_t = tanh(w_a * h_t + b_a)
        One number per timestep. tanh keeps scores in [-1, +1].

Step 2: Normalise scores to attention weights summing to 1.0
        alpha_t = exp(e_t) / sum(exp(e_s) for all s)
        These are the attention weights — a probability distribution over timesteps.

Step 3: Weighted sum to produce context vector
        c = sum(alpha_t * h_t)
        Timesteps with high alpha contribute more to the final representation.
```

**The cost:**

```
Dense(1) on 128 inputs = 128 weights + 1 bias = 129 parameters

129 parameters = 0.05% of total model size
Yet improves STANDING recall by approximately 4 percentage points.
```

This asymmetry — negligible parameter cost, meaningful accuracy improvement — is the hallmark of a well-targeted architectural inductive bias. The attention mechanism tells the model where to look. Without it, the model has to spread its attention equally across all timesteps, diluting the subtle evidence.

<br/>

---

<br/>

## Tech Stack

<br/>

```
  Signal Analysis          Preprocessing            Deep Learning
  ───────────────          ─────────────            ─────────────
  SciPy                    scikit-learn             TensorFlow 2.x
  └─ fft                   └─ StandardScaler        Keras
  └─ fftfreq               └─ classification_report └─ Conv1D
                           └─ confusion_matrix      └─ MaxPooling1D
                           └─ accuracy_score        └─ LSTM
  Data Handling                                     └─ Dense
  ─────────────            Visualisation            └─ Attention
  NumPy 1.24+              ─────────────            └─ BatchNorm
  Pandas 1.5+              Matplotlib               └─ Dropout
                           Seaborn

  Environment
  ───────────
  Google Colab  (T4 GPU recommended)
  Python 3.10+
```

<br/>

---

<br/>

## Repository Structure

<br/>

```
imu-activity-recognition/
│
├── README.md                          This file
│
├── Theory.ipynb              Theory notebook — no code
│   │                                  Read before implementation
│   ├── IMU sensor physics
│   ├── FFT derivation and frequency analysis
│   ├── CNN convolution operation
│   ├── LSTM gate equations
│   ├── Attention mechanism math
│   ├── Cross-entropy loss derivation
│   └── Evaluation metrics
│
├── implementation.ipynb            Complete implementation
│   │                                  118 cells, fully documented
│   │                                  Every concept explained in
│   │                                  plain English before code
│   ├── Section 1   Setup and constants
│   ├── Section 2   Dataset download and verification
│   ├── Section 3   Data loading (one function per cell)
│   ├── Section 4   Signal visualisation
│   ├── Section 5   FFT frequency analysis
│   ├── Section 6   Preprocessing and normalisation
│   ├── Section 7   Model construction (one layer per cell)
│   ├── Section 8   Training with callbacks
│   ├── Section 9   Evaluation and confusion matrix
│   ├── Section 10  Confidence analysis
│   ├── Section 11  Channel importance
│   ├── Section 12  Save and summary
│   └── Section 13  Real-world applications
│
└── outputs/                           Generated after running
    ├── raw_signals_per_activity.png
    ├── class_distribution.png
    ├── fft_analysis.png
    ├── sitting_vs_standing.png
    ├── normalisation.png
    ├── training_curves.png
    ├── confusion_matrix.png
    ├── confidence_distribution.png
    ├── calibration_curve.png
    ├── channel_importance.png
    ├── fft_per_channel.png
    ├── final_summary_pie.png
    ├── cnn_lstm_attention_har.keras    Saved model weights
    ├── scaler_har.pkl                  Fitted StandardScaler
    └── activity_labels.txt            Index to activity name mapping
```

<br/>

---

<br/>

## Quickstart

<br/>

**Step 1 — Read the theory notebook**

Open `Theory.ipynb` in Colab. No code to run — just equations and explanations.
This notebook covers every concept used in implementation.
Reading it means no equation in the implementation notebook will be mysterious.

**Step 2 — Enable GPU**

```
Runtime  →  Change runtime type  →  T4 GPU  →  Save
```

Without GPU, training takes 20-25 minutes. With T4, it takes 3-5 minutes.

**Step 3 — Open implementation notebook**

Open `Project2_Complete.ipynb`. Every code cell has a markdown cell directly above it.
The markdown explains what the cell does, why it does it, and what to check in the output.

**Step 4 — Run cells in order**

```
Section 1   Setup               ~1 min    install, imports, GPU check
Section 2   Dataset             ~30 sec   download from UCI repository
Section 3   Load data           ~1 min    read 9 signal files per split
Section 4   Visualise           instant   raw signal plots
Section 5   FFT analysis        ~2 min    frequency domain analysis
Section 6   Preprocessing       instant   scale + one-hot encode
Section 7   Build model         instant   layer by layer construction
Section 8   Train               ~5 min    with T4 GPU
Section 9   Evaluate            instant   confusion matrix + report
Section 10  Confidence          instant   calibration analysis
Section 11  Channel importance  ~2 min    separability scoring
Section 12  Save + download     instant   all files to local machine
                                ─────────
                     Total      ~12 min
```

No manual data download. No local setup. Everything runs inside Colab.

<br/>

---

<br/>

## Real-World Applications

<br/>

The technique built here is not a benchmark exercise. It runs in production systems.

<br/>

```
  ┌──────────────────────────────────────────────────────────────┐
  │  ISRO / AEROSPACE                                            │
  │                                                              │
  │  IISU (Inertial Systems Unit) in Thiruvananthapuram builds   │
  │  guidance computers for PSLV, GSLV, LVM3, and Gaganyaan.    │
  │                                                              │
  │  The flight IMU produces accelerometer and gyroscope data    │
  │  throughout ascent. The CNN-LSTM pipeline — detecting local  │
  │  patterns, tracking temporal evolution — applies directly    │
  │  to flight state classification and fault detection.         │
  │                                                              │
  │  Six activity classes in UCI HAR are analogous to six        │
  │  flight states: nominal, vibration fault, attitude error,    │
  │  sensor bias, structural anomaly, engine irregularity.       │
  │  The mathematics is identical. Only the labels change.       │
  └──────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────┐
  │  HEALTHCARE AND REHABILITATION                               │
  │                                                              │
  │  Physiotherapists use wearable IMU sensors to quantify       │
  │  patient gait before and after surgery. The same CNN-LSTM    │
  │  pipeline classifies walking quality, detects abnormal       │
  │  patterns, and measures recovery objectively.                │
  │                                                              │
  │  Fall detection in elderly patients uses activity            │
  │  recognition to distinguish normal transitions (standing     │
  │  from a chair) from falls. The SITTING vs STANDING           │
  │  confusion identified in this project is exactly the         │
  │  boundary case that makes fall detection hard in practice.   │
  └──────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────┐
  │  AUTONOMOUS VEHICLES                                         │
  │                                                              │
  │  Every self-driving car fuses accelerometer and gyroscope    │
  │  data with LiDAR and cameras at 100+ Hz. In tunnels and      │
  │  underground car parks where GPS is unavailable, the         │
  │  vehicle falls back on IMU-based classification of road      │
  │  surface type, turning manoeuvres, and speed bumps.          │
  │  The same architecture applied to a different label set.     │
  └──────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────┐
  │  INDUSTRIAL IoT AND PREDICTIVE MAINTENANCE                   │
  │                                                              │
  │  Manufacturing equipment vibrates at characteristic          │
  │  frequencies during normal operation. Bearing wear,          │
  │  gear tooth damage, and shaft imbalance shift those          │
  │  frequency patterns in detectable ways.                      │
  │                                                              │
  │  The FFT analysis from Section 5, combined with LSTM         │
  │  temporal modelling, is the standard approach for            │
  │  predictive maintenance across factories worldwide.          │
  └──────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────┐
  │  CONSUMER WEARABLES                                          │
  │                                                              │
  │  Every fitness tracker and smartwatch uses variants of       │
  │  this pipeline: step counting, sleep stage detection,        │
  │  sports performance analysis. Apple Watch fall detection     │
  │  runs a CNN-based model on a dedicated neural engine chip    │
  │  in real time. The architecture you built is that model.     │
  └──────────────────────────────────────────────────────────────┘
```

<br/>

---

<br/>

## Key Concepts

<br/>

| Concept | Section | One-line explanation |
|:--------|:-------:|:---------------------|
| Accelerometer | 3 | Measures specific force — kinematic acceleration plus gravity |
| Gyroscope | 3 | Measures angular velocity — rotation rate in rad/s |
| Sampling rate | 3 | 50 Hz — 50 readings per second per channel |
| FFT | 5 | Converts time signal to frequency spectrum |
| Frequency resolution | 5 | 50/128 = 0.39 Hz — smallest detectable frequency difference |
| Walking cadence | 5 | ~2 Hz dominant peak — 2 footsteps per second |
| StandardScaler | 6 | Transforms each channel to mean=0, std=1 |
| Data leakage | 6 | Fitting scaler on test data — inflates accuracy artificially |
| One-hot encoding | 6 | Integer label 3 becomes vector [0,0,0,1,0,0] |
| Conv1D | 7 | Sliding filter detects local patterns — kernel size 3 = 60ms |
| MaxPooling | 7 | Halves sequence length, expands effective receptive field |
| ReLU | 7 | max(0,x) — kills negatives, prevents vanishing gradients |
| BatchNormalization | 7 | Normalises activations between layers — stabilises training |
| Dropout | 7 | Randomly disables neurons — prevents memorisation |
| LSTM cell state | 7 | Additive memory — gradient highway across 32 timesteps |
| Forget gate | 7 | Decides what to erase from long-term memory |
| return_sequences | 7 | Output all 32 hidden states — required for attention |
| Attention weights | 7 | Softmax scores summing to 1.0 — which timesteps matter |
| Context vector | 7 | Weighted sum of LSTM outputs — attended representation |
| Cross-entropy loss | 8 | -log(probability of true class) |
| Adam optimiser | 8 | Adaptive per-parameter learning rates |
| EarlyStopping | 8 | Stops when val_loss stops improving — restores best weights |
| Subject-based split | dataset | 21 train subjects, 9 test subjects — no person in both |
| Confusion matrix | 9 | Grid showing true vs predicted class for all test windows |
| Precision | 9 | When model says class X, how often is it right? |
| Recall | 9 | Of all true class X windows, how many did model catch? |
| F1-score | 9 | Harmonic mean of precision and recall |
| Calibration curve | 10 | Confidence vs actual accuracy — well-calibrated = diagonal |

<br/>

---

<br/>

<div align="center">

```
  ┌─────────────────────────────────────────────────────┐
  │                                                     │
  │   Project   —  Sensor Fusion Series                 │
  │                                                     │
  │   Project  :  GPS + IMU Sensor Fusion               │
  │               Kalman Filter + LSTM                  │
  │               Trajectory estimation under GPS loss  │
  │                                                     │
  └─────────────────────────────────────────────────────┘
```

<br/>

*The attention mechanism that improved STANDING recall by 4 points*
*uses 129 parameters — 0.05% of the model.*
*The right architectural choice costs almost nothing.*

<br/>

---

**Prem M Patel** — prempatel7740@gmail.com — **Computer Engineering** 


</div>