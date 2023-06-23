# CNN Architectures for Image Classification

This repository contains the PyTorch implementation for classifying the MNIST handwritten digit dataset using various Convolutional Neural Network (CNN) architectures. Each model in the `model.py` file employs a different type of normalization technique: Batch Normalization, Group Normalization, and Layer Normalization.

1. `model.py`: Contains the definition of the CNN models with different normalization techniques.
2. `utils.py`: Contains the utility functions needed for data processing and training.
3. `main.ipynb`: The main Jupyter notebook that puts everything together and runs the training and testing processes.

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

This file contains the definitions of three CNN models:

- `BNNet`:  Batch Normalization.
- `GNNet`:  Group Normalization.
- `LNNet`: Layer Normalization.

All three architectures have the same structure of four convolutional layers and two pooling layers, followed by an adaptive average pooling layer and a fully connected layer.

### 2. `utils.py`

This file contains various utility functions:

- `train()`: Function to train the model for one epoch.
- `test()`: Function to evaluate the model on the test data.

### 3. `S8.ipynb`

This Jupyter notebook is the main entry point for this project. It imports the functions and models defined in `model.py` and `utils.py`, performs data preprocessing, initializes the models, and runs the training and testing loops for each model.

## How to Run

1. Install all the prerequisites.
2. Clone this repository.
3. Run the `S8.ipynb` notebook in Jupyter.

## Results

The results, including the training and testing accuracies and losses for each model, are plotted in the `main.ipynb` notebook after the models are trained.
