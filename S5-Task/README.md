# MNIST Handwritten Digit Classification

This repository contains the PyTorch implementation for classifying the MNIST handwritten digit dataset. The code is separated into three main files:

1. `model.py`: Contains the definitions of two neural network architectures.
2. `utils.py`: Contains the utility functions needed for data processing and training.
3. `S5.ipynb`: The main Jupyter notebook that puts everything together and runs the training and testing processes.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [File Descriptions](#file-descriptions)
3. [How to Run](#how-to-run)
4. [Results](#results)

## Prerequisites

The project requires the following packages:

- Python 3.x
- PyTorch 1.x
- Torchvision
- Numpy
- Matplotlib
- tqdm
- torchsummary

## File Descriptions

### 1. `model.py` 

This file contains the definitions of two convolutional neural network models:

- `Net`: A ConvNet with 4 convolution layers and 2 fully connected layers.
- `Net2`: Similar to `Net`, but all the layers in this model are defined without a bias.

### 2. `utils.py`

This file contains various utility functions:

- `train()`: Function to train the model for one epoch.
- `test()`: Function to evaluate the model on the test data.
- `GetCorrectPredCount()`: Helper function to count the number of correct predictions.

### 3. `S5.ipynb`

This Jupyter notebook is the main entry point for this project. It imports the functions and models defined in `model.py` and `utils.py`, performs data augmentation, initializes the model, and runs the training and testing loops.

## How to Run

1. Install all the prerequisites.
2. Clone this repository.
3. Run the `S5.ipynb` notebook in Jupyter.

## Results

The results, including the training and testing accuracies and losses, are plotted in the `S5.ipynb` notebook after the model is trained.

---
