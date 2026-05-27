# AI-ML Assignments

A comprehensive collection of artificial intelligence, deep learning, computer vision, and machine learning assignments. This repository serves as an academic and practical portfolio demonstrating end-to-end model implementations, feature engineering pipelines, and rigorous performance evaluations.

---

## 📂 Repository Structure & Overview

The repository contains modular Jupyter notebooks and project files structured as follows:

| # | Assignment File / Directory | Domain | Focus Area & Key Techniques |
|---|---|---|---|
| 1 | `2025AA05327_dnn_assignment.ipynb` | Deep Learning (DNN) | Binary Classification, MLP vs. Linear Models, Medical Diagnosis (Recall-optimized) |
| 2 | `2025AA05327_cnn_assignment.ipynb` | Computer Vision (CNN) | Deep Spatial Feature Extraction, 2D Convolutional Networks, Softmax Classification |
| 3 | `2025AA05327_rnn_assignment.ipynb` | Sequential Learning (RNN) | Recurrent Architecture Benchmarking (Vanilla RNN vs. GRU vs. LSTM) |
| 4 | `CV_assignment1_35_1.ipynb` | Computer Vision (Classical) | Handwritten Character Recognition, Feature Extraction (HOG, LBP), Supervised ML |
| 5 | `2025AA05327_ML_Assignment2/` | Machine Learning | Core ML Principles, Pipeline Preprocessing, Framework Implementations |

---

## 🛠️ Detailed Project Breakdown

### 1. Convolutional Neural Networks (CNN)
* **File:** `2025AA05327_cnn_assignment.ipynb`
* **Objective:** Transition from manual feature engineering to automated visual hierarchy learning for image classification.
* **Exact Pipeline & Code Execution:**
  * **Data Handling:** Reshapes input matrices to explicitly enforce tensor compatibility `(batch, width, height, channels)` and applies pixel-value normalization to scale intensities down to a continuous $[0, 1]$ range.
  * **Architecture Blueprint:**
    * **`Conv2D` Layers:** Employs localized $3 \times 3$ kernel filters matched with **ReLU** non-linear activation functions to systematically isolate edges, textures, and geometric shapes.
    * **`MaxPooling2D` Layers:** Employs $2 \times 2$ downsampling windows to enforce spatial translation invariance and aggressively slash computational overhead.
    * **Regularization & Output:** Injects `Dropout` layers ($0.25$ to $0.5$) during the forward pass to penalize co-dependency, flattens the dimensional output into a 1D feature vector, and routes it through a final dense head using **Softmax** to generate multi-class probability scores.
  * **Optimization Engine:** Compiled via the **Adam** optimizer tracking `CategoricalCrossentropy` loss, integrated with dynamic callbacks like `ReduceLROnPlateau` and `EarlyStopping`.

### 2. Recurrent Neural Networks (RNN)
* **File:** `2025AA05327_rnn_assignment.ipynb`
* **Objective:** Capture sequential relationships and long-term historical context within time-ordered / tokenized sequence data.
* **Exact Pipeline & Code Execution:**
  * **Sequence Preprocessing:** Converts categorical strings or data sequences into standardized integer tokens, forcing uniformity across variable-length lines via sequence padding (`pad_sequences`).
  * **Feature Embedding:** Feeds padded sequence indices into a joint `Embedding` layer to map integers into dense continuous low-dimensional vector representations.
  * **Comparative Architectures Evaluated:**
    * **Vanilla RNN (`SimpleRNN`):** Used as a baseline model to capture short-range dependencies, highlighting architectural bottlenecks induced by the **Vanishing Gradient Problem**.
    * **LSTM (Long Short-Term Memory):** Implements specialized gating controls (**Forget, Input, and Output gates**) to seamlessly route and preserve gradients over expansive sequence spaces.
    * **GRU (Gated Recurrent Unit):** Leverages compressed dual-gating structures (**Reset and Update gates**) to accelerate training runs while maintaining comparable sequence tracking capabilities.
  * **Configuration Details:** Manipulates the `return_sequences` boolean flag to cleanly stack deep recurrent sequences down into a static sequence vector, terminating in a dense layer mapped to the target label workspace.

### 3. Machine Learning — Foundations & Core Principles
* **Directory:** `2025AA05327_ML_Assignment2/`
* **Focus:** Implementation of fundamental machine learning pipelines and mathematical frameworks.
* **Key Implementations:**
  * **Feature Engineering:** Automated strategies handling missing value imputations, structural outliers, and scaling variances (Standard vs. MinMax Scalers).
  * **Model Selection:** Rigorous benchmarking across classic algorithms including linear classifiers, decision-tree paradigms (Random Forests), and distance clusterers.
  * **Validation Protocol:** Implements K-Fold cross-validation loops to establish verifiable generalization metrics and prevent training-set leakage.

### 4. Deep Neural Networks (DNN)
* **File:** `2025AA05327_dnn_assignment.ipynb`
* **Dataset:** Breast Cancer Wisconsin Dataset (569 samples, 30 features).
* **Task:** Medical binary classification (Malignant vs. Benign).
* **Optimization Metric:** **Recall**. In clinical environments, minimizing False Negatives (unmissed critical conditions) takes definitive priority over precision tracking.

### 5. Computer Vision — Classical Approaches
* **File:** `CV_assignment1_35_1.ipynb`
* **Dataset:** EMNIST Letters dataset (20,000 images, standard A-Z characters).
* **Feature Engineering:** Image contrast correction using Histogram Equalization followed by explicit statistical feature tracking via Histogram of Oriented Gradients (**HOG**) and Local Binary Patterns (**LBP**).

---

## 🚀 Getting Started

### Prerequisites
Install the comprehensive scientific calculation and deep learning stack:
```bash
pip install numpy pandas matplotlib scikit-learn tensorflow torch torchvision
