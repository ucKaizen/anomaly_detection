---
title: Anomaly Detection UI
emoji: 🚀
color: blue
sdk: gradio
app_file: app.py  # Or the name of your main Gradio script if it's different
# model_id: ucKaizen/your-pretrained-model # Optional: If you have a specific model on the Hub
tags:
- anomaly detection
- image classification
- gradio
- machine learning
- unsupervised learning
- semi-supervised learning
---

# Anomaly Detection - Unsupervised and Semi Supervised
![GitHub](https://img.shields.io/github/license/ucKaizen/anomaly_detection?style=flat-square)  
![GitHub last commit](https://img.shields.io/github/last-commit/ucKaizen/anomaly_detection?style=flat-square)  
![GitHub issues](https://img.shields.io/github/issues/ucKaizen/anomaly_detection?style=flat-square)

*Detect anomalies in rice images using unsupervised and semi-supervised learning with a user-friendly Gradio interface.*
Dataset a subset of https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset

## Overview

This project implements an anomaly detection system for rice images (Basmati vs. Jasmine) using deep learning and unsupervised learning techniques. It leverages EfficientNetB0 for feature extraction, PCA for dimensionality reduction, and PyOD models (e.g., Isolation Forest, LOF, OCSVM) for anomaly detection. The project includes a Gradio-based UI that allows users to:

- Select between unsupervised and semi-supervised modes.
- Choose different PyOD models for anomaly detection.
- Tune model parameters (e.g., `contamination`, `n_estimators`).
- Visualize results with a PCA scatter plot, classification metrics, and a list of detected outliers.

The project is built with Python and uses libraries like TensorFlow, scikit-learn, PyOD, OpenCV, Matplotlib, and Gradio. It’s designed to run seamlessly in GitHub Codespaces for development and testing.

## Features

- **Anomaly Detection**: Detects Jasmine rice images as anomalies among Basmati images.
- **Modes**: Supports both unsupervised and semi-supervised learning.
- **Models**: Includes multiple PyOD models (Isolation Forest, Local Outlier Factor, One-Class SVM).
- **Interactive UI**: Gradio interface with dropdowns, sliders, and visualizations.
- **Visualization**: PCA scatter plot to visualize data distribution and detected outliers.
- **Metrics**: Provides classification report, AUC score, and a list of detected outliers with filenames.


Deploy to HF:
    export HF_TOKEN=XXX 
    git remote add space "https://ucKaizen:${HF_TOKEN}@huggingface.co/spaces/ucKaizen/anomaly_detection"
    git push --force space main