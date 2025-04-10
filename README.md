# Anomaly Detection UI

![GitHub](https://img.shields.io/github/license/your-username/rice-anomaly-detection-ui?style=flat-square)  
![GitHub last commit](https://img.shields.io/github/last-commit/your-username/rice-anomaly-detection-ui?style=flat-square)  
![GitHub issues](https://img.shields.io/github/issues/your-username/rice-anomaly-detection-ui?style=flat-square)

*Detect anomalies in rice images using unsupervised and semi-supervised learning with a user-friendly Gradio interface.*

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

## Prerequisites

To run this project, you’ll need:
- A GitHub account with access to [GitHub Codespaces](https://github.com/features/codespaces).
- The [Rice Image Dataset](https://www.kaggle.com/datasets/muratkokludataset/rice-image-dataset) (you’ll need to upload it to Codespaces or adjust the dataset path in `app.py`).

## Getting Started with GitHub Codespaces

GitHub Codespaces provides a cloud-based development environment pre-configured with all the tools you need to run this project. Follow these steps to get started:

1. **Open the Project in Codespaces**:
   - Navigate to the repository: [your-username/rice-anomaly-detection-ui](https://github.com/your-username/rice-anomaly-detection-ui).
   - Click the green **Code** button, then select **Open with Codespaces** > **New codespace**.
   - Wait for the Codespace to set up (this may take a few minutes).

2. **Set Up the Environment**:
   - Codespaces will automatically create a virtual environment and install dependencies if a `requirements.txt` file is detected. However, to ensure everything is set up correctly, run:
     ```bash
     make