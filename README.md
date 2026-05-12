# Naive Bayes Image Classifier

A Python implementation of a Naive Bayes classifier for image-based classification tasks.

This project predicts labels for images using probabilistic feature analysis and logarithmic likelihood scoring.

---

# Overview

The classifier:

- Computes prior probabilities for each label
- Learns feature probabilities from training data
- Uses logarithmic probability calculations to avoid numerical underflow
- Predicts the most likely label for each image

The implementation follows the standard Naive Bayes assumption that features are conditionally independent given the class label.

---

# Features

- Naive Bayes probabilistic classification
- Log-space likelihood computation
- Binary feature handling
- Multiple class support
- Simple and lightweight Python implementation

---

# Files

| File | Description |
|---|---|
| `naive_bayes_prediction.py` | Main prediction/classification logic |
| `training_data.*` | Training dataset |
| `test_data.*` | Testing dataset |
| `main.py` | Runs the classifier (if applicable) |

---

# How It Works

For each possible label, the classifier computes:

\[
P(label) \times \prod P(feature_i \mid label)
\]

To improve numerical stability, probabilities are calculated in log-space:

\[
\log P(label) + \sum \log P(feature_i \mid label)
\]

The label with the highest score is selected as the prediction.

---

# Requirements

- Python 3.x

Standard libraries used:
- `math`
- `collections`
- `csv` (if applicable)

No external dependencies are required unless additional libraries were added separately.

---

# Running the Program

Run from terminal:

```bash
python3 main.py
