# Convolutional Neural Network (CNN) for Image Classification

This project implements a Convolutional Neural Network (CNN) using PyTorch for classifying images into two categories: cats and dogs. The model is trained on a dataset with preprocessing steps and evaluated on a test dataset. It also includes a function to make a single image prediction.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Data Preprocessing](#data-preprocessing)
- [Model Architecture](#model-architecture)
- [Training the Model](#training-the-model)
- [Evaluating the Model](#evaluating-the-model)
- [Making a Prediction](#making-a-prediction)
- [Usage](#usage)
- [Test Accuracy](#test-accuracy)
- [Dataset](#dataset)

## Introduction
This repository demonstrates how to build a CNN model for image classification using PyTorch. The network is designed to classify images as either cats or dogs. The dataset is loaded from Google Drive and includes separate training and test sets.

## Requirements
To run this project, you need to have the following libraries installed:
- Python 3.x
- PyTorch
- torchvision
- PIL (Python Imaging Library)
- Google Colab (optional)

Install dependencies using:
```bash
pip install torch torchvision Pillow
```

## Data Preprocessing
The data is preprocessed with the following transformations:
- Resizing images to 64x64 pixels.
- Random horizontal flips, affine transformations, and normalization using ImageNet mean and standard deviation values.

Both training and test sets undergo transformations, with additional augmentations applied to the training set.

## Model Architecture
The CNN consists of two convolutional layers followed by max-pooling, fully connected layers, and a sigmoid output layer for binary classification.

### Key Components:
- **Conv1**: First convolutional layer with 32 filters.
- **MaxPooling**: Pooling layer to reduce spatial dimensions.
- **Conv2**: Second convolutional layer with 32 filters.
- **Fully Connected Layers**: Two fully connected layers, ending with a sigmoid output for binary classification (cat or dog).

## Training the Model
The model is trained for 25 epochs using the Adam optimizer and binary cross-entropy loss. The model's training loss is printed at each epoch.

## Evaluating the Model
The model is evaluated on the test set to compute its accuracy. It predicts whether the image belongs to a cat or a dog by setting a threshold on the output of the sigmoid function.

## Making a Prediction
A function is provided to predict the class of a single image. The image is preprocessed and passed through the model to output either 'cat' or 'dog' as the predicted class.

## Usage

1. **Mount Google Drive** (for Colab users):
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
2. **Prepare the Dataset**: The dataset is available [here](https://drive.google.com/drive/folders/10xnFZuik7tP7MhnrXJY4C3aQx8WzBvoH?usp=drive_link). Organize your training and test datasets in the appropriate directories in Google Drive.
3. **Train the Model**: Run the training script to train the model on the dataset.
4. **Evaluate the Model**: Test the model's accuracy on the test dataset.
5. **Make Predictions**: Use the provided function to predict a class for a new image.

## Test Accuracy
The model achieved a **Test Accuracy** of **0.7851** (78.51%).

## Dataset
The dataset used in this project can be accessed through this [Google Drive link](https://drive.google.com/drive/folders/10xnFZuik7tP7MhnrXJY4C3aQx8WzBvoH?usp=drive_link). Please download it and organize it accordingly before running the training.
