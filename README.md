# Ensemble CNN for Bone Fracture Classification

## Project Overview

This project classifies bone X-ray images into two categories: **Fractured** and **Not Fractured** using deep learning.

It explores three models—Custom CNN, ResNet18, and MobileNetV2—and an ensemble approach to compare their classification performance.

## Technologies Used

- Python
- PyTorch and Torchvision
- NumPy
- Pillow (PIL)
- Scikit-learn
- Matplotlib

## Models Implemented

- **Custom CNN:** A convolutional neural network for image classification.
- **ResNet18:** A residual neural network used for image classification.
- **MobileNetV2:** A lightweight CNN architecture.
- **Ensemble Model:** Combines model predictions to generate a final classification.

## Project Structure

The repository contains Python scripts for model training and evaluation, saved `.pth` model checkpoints, and dependency information.

## Installation

### 1. Clone the repository

Replace `YOUR-USERNAME` with your GitHub username.

```bash
git clone https://github.com/YOUR-USERNAME/ensemble-cnn-bone-fracture-classification.git
cd ensemble-cnn-bone-fracture-classification
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset

Prepare the dataset according to the file paths and folder structure expected by the code. The dataset is not included in this repository.

## Evaluation

The individual models and ensemble approach are evaluated on a test dataset to compare their classification performance.

## Future Improvements

- Evaluate precision, recall, and F1-score.
- Add model interpretability using Grad-CAM.
- Test performance on an independent dataset.
- Develop a simple interface for image classification.

## Disclaimer

This project is intended for educational and research purposes only. It is not a clinically validated diagnostic tool.

