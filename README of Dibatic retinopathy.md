# Diabetic Retinopathy Screening Using Edge AI

A deep-learning prototype for classifying diabetic retinopathy severity from retinal fundus images, with a future goal of efficient deployment on low-power Edge AI devices.

> **Project status:** Initial model training and validation completed. Edge-device deployment, API/UI integration, and further model optimization are still in progress.

## Table of Contents
- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Objectives](#objectives)
- [Dataset](#dataset)
- [System Architecture](#system-architecture)
- [Implementation Status](#implementation-status)
- [Initial Results](#initial-results)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)
- [Limitations and Pending Work](#limitations-and-pending-work)
- [Responsible Use](#responsible-use)

## Overview

Diabetic retinopathy is an eye condition associated with diabetes that can lead to vision loss if it is not identified and managed in time. This project explores the use of deep learning to classify retinal fundus images into five diabetic retinopathy severity categories. The model uses MobileNetV2 transfer learning as a lightweight starting point for future Edge AI deployment.

This is an academic prototype and **not a clinically validated diagnostic system**.

## Problem Statement

Manual screening of retinal images requires trained professionals and access to appropriate screening facilities. The project investigates whether a lightweight AI model can support preliminary image classification and, after further development and validation, potentially run on low-power devices.

## Objectives

- Build a model that classifies retinal fundus images into five severity classes.
- Use transfer learning to establish an initial model efficiently.
- Evaluate the model on a held-out validation set.
- Improve classification performance and investigate deployment on low-power Edge AI devices such as Raspberry Pi or smartphones.

## Dataset

The prototype uses the **APTOS 2019 Blindness Detection** dataset.

The working dataset contained **2,930 usable images**, split into:
- Training: **2,344 images**
- Validation: **586 images**

The five label categories are:

| Label | Category |
|---:|---|
| 0 | No DR |
| 1 | Mild |
| 2 | Moderate |
| 3 | Severe |
| 4 | Proliferative DR |

Follow the dataset provider's terms and conditions when accessing or redistributing data. Do not commit the dataset or private data to this repository.

## System Architecture

1. **Input:** Retinal fundus images.
2. **Preprocessing:** Prepare images for model input.
3. **Feature extraction and classification:** MobileNetV2 with transfer learning.
4. **Training and validation:** Train on the training split and evaluate on the held-out validation split.
5. **Future deployment:** Optimize and integrate the model into an application for a low-power edge device. This deployment is not yet complete.

## Implementation Status

### Completed
- Prepared the APTOS 2019 dataset for the experiment.
- Created training and validation splits.
- Implemented a MobileNetV2-based five-class classifier using transfer learning.
- Trained the initial model for five epochs.
- Evaluated validation accuracy and validation loss.
- Saved the trained model in HDF5 (`.h5`) format.

### In progress / pending
- Improve accuracy and assess per-class performance.
- Fine-tune the model and investigate class-imbalance handling.
- Add further evaluation, such as a confusion matrix and per-class precision/recall.
- Build API and user-interface integration.
- Measure inference latency and memory use on target hardware.
- Optimize and test deployment on a low-power Edge AI device.

## Initial Results

The initial five-epoch experiment produced the following results:

| Metric | Result |
|---|---:|
| Training images | 2,344 |
| Validation images | 586 |
| Number of classes | 5 |
| Training epochs | 5 |
| Best recorded validation accuracy | 60.07% (Epoch 3) |
| Final validation accuracy | 57.34% (Epoch 5) |
| Final validation loss | 1.052 |

These are preliminary results from one initial experiment. They do not establish clinical reliability. Inference latency, edge-device performance, and clinical validation have not yet been measured.

## Technology Stack

- Python
- TensorFlow / Keras
- MobileNetV2 transfer learning
- NumPy and Pandas for data handling
- scikit-learn for dataset splitting and evaluation utilities, where used
- Matplotlib for plots, where used
- APTOS 2019 retinal fundus image dataset

## Repository Structure

A suggested structure for organizing the project:

```text
Diabetic-Retinopathy-Edge-AI/
├── README.md
├── requirements.txt
├── .gitignore
├── notebooks/
│   └── model_training.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
├── models/
│   └── .gitkeep
├── results/
│   └── training_results.png
└── docs/
    └── project_report.pdf
```

This is a suggested layout. Include only files that actually exist in your repository; update filenames to match your implementation.

## How to Run

The exact commands depend on the files in your repository. A typical setup is:

1. Clone the repository:
   ```bash
   git clone <YOUR_REPOSITORY_URL>
   cd Diabetic-Retinopathy-Edge-AI
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   ```

   On Windows:
   ```bash
   .venv\Scripts\activate
   ```

   On macOS/Linux:
   ```bash
   source .venv/bin/activate
   ```

3. Install the dependencies listed in `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the APTOS 2019 dataset from its official Kaggle competition page and place it in the paths expected by your notebook or scripts.

5. Run the notebook or training script that is actually present in the repository. Update this section with the exact command and dataset paths used by your code.

## Limitations and Pending Work

- Current validation accuracy is limited and requires improvement.
- Performance needs to be evaluated per class, especially for less-represented severity categories.
- More comprehensive evaluation and error analysis are required.
- API/UI integration and edge-device deployment are not complete.
- Latency, memory consumption, and power use have not been measured on target hardware.
- Clinical validation and assessment by qualified healthcare professionals would be required before any real-world medical use.

## Responsible Use

This project is for educational and research purposes only. It must not be used to diagnose, rule out, or make treatment decisions about diabetic retinopathy. Predictions may be incorrect. Anyone with concerns about eye health should consult a qualified eye-care professional.

## Acknowledgements

- APTOS 2019 Blindness Detection dataset and its contributors.
- TensorFlow/Keras and the MobileNetV2 model implementation.

## License

Add a license only after choosing one that fits your project and any dataset/model dependencies. Check the applicable dataset terms before redistributing any data.
