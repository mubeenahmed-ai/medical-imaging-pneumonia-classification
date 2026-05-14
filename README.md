# Medical Image Classification: Pneumonia Detection

This project develops and evaluates deep learning architectures for the automated detection of pneumonia from chest X-ray images. It provides a comparative analysis between a custom-built Convolutional Neural Network (CNN) and a pre-trained ResNet model using Transfer Learning.

## Project Overview
* **Domain:** Medical Computer Vision / Healthcare AI.
* **Architectural Comparison:** Benchmarking a custom 4-layer CNN against a deep ResNet-18/50 architecture.
* **Transfer Learning:** Leveraging ImageNet-pre-trained weights and fine-tuning the final fully connected layers for binary classification (Normal vs. Pneumonia).
* **Generalization Analysis:** Evaluating the "learning vs. memorization" trade-off by analyzing validation loss curves and feature extractor efficacy.

## Tech Stack
* **Language:** Python
* **Frameworks:** PyTorch / TensorFlow (Keras)
* **Libraries:** NumPy, Matplotlib, OpenCV, Scikit-learn
* **Hardware Acceleration:** CUDA (trained on NVIDIA T4/A100)

## Key Technical Features
* **Custom CNN Pipeline:** Implementation of pooling, dropout, and ReLU activation layers to extract spatial hierarchies from grayscale X-ray data.
* **Feature Extraction Analysis:** Discussion of universal visual features (edges, gradients, textures) and their application to specific medical anomalies like pulmonary opacities.
* **Optimization Strategies:** Utilization of Adam/SGD optimizers, data augmentation to prevent overfitting, and learning rate scheduling.
* **Performance Metrics:** Comprehensive evaluation using Accuracy, Precision, Recall, and F1-Score, focusing on minimizing false negatives in a clinical context.

## Results & Insights
* **Transfer Learning Efficacy:** The pre-trained ResNet significantly outperformed the custom CNN in generalization, demonstrating the power of pre-trained feature extractors in low-data regimes.
* **Overfitting Mitigation:** Analysis of training vs. validation delta, highlighting how architectural depth and pre-training impact the model's ability to handle unseen patient data.

## Project Structure
* `pneumonia-classification-analysis.ipynb`: The complete technical workflow from data preprocessing to comparative model evaluation.
