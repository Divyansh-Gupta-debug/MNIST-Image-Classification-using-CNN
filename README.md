## Overview
This project implements a Convolutional Neural Network (CNN) to classify handwritten
digits from the MNIST dataset using PyTorch. The objective is to design, train, and
evaluate a deep learning model for image classification.

This project was developed as part of the *Introduction to Intelligent Systems* course.

## Files
├── MNIST Image Classification.ipynb
└── README.md

## Dataset
- MNIST handwritten digit dataset
- Loaded using `torchvision.datasets.MNIST`

## Model Details
- Convolutional layers with ReLU activation
- Pooling layers for spatial reduction
- Fully connected layers
- Output layer with 10 neurons (digits 0–9)


## Training & Evaluation
- Trained for multiple epochs
- Tracked training and validation accuracy and loss
- Evaluated using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
- Visualized confusion matrix
- Analyzed misclassified digits

## Tech Stack
- Python
- PyTorch
- Torchvision
- NumPy
- Matplotlib

## How to Run
```bash
jupyter notebook "MNIST Image Classification.ipynb"
Notes
All experiments, visualizations, and evaluations are contained within the notebook.
