# Breast-Cancer-Identification-
# Breast Cancer Prediction using Decision Tree Classifier

This project demonstrates the use of a Decision Tree Classifier to predict breast cancer using the Breast Cancer Wisconsin (Diagnostic) Data Set from the sklearn library.

## Table of Contents
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Introduction
This project uses a Decision Tree Classifier to predict whether a given tumor is benign or malignant. The model is fine-tuned using GridSearchCV to find the best hyperparameters.

## Dataset
The dataset used in this project is the Breast Cancer Wisconsin (Diagnostic) Data Set, which is available in the sklearn library. The target variable has two classes:
- `0`: Malignant (WDBC-Malignant)
- `1`: Benign (WDBC-Benign)

The dataset contains 569 instances with 30 feature columns. More information about the dataset can be found [here](https://scikit-learn.org/stable/datasets/toy_dataset.html#breast-cancer-dataset).

## Installation
To run this project, you'll need to have Python and the following libraries installed:
- numpy
- scikit-learn

You can install the required libraries using pip:


- pip install numpy scikit-learn

## Usage
1. To use this project, follow these steps:
2. Load the dataset using sklearn.datasets.load_breast_cancer.
3. Split the dataset into training and testing sets using train_test_split.
4. Define the hyperparameter grid for the Decision Tree Classifier.
5. Use GridSearchCV to find the best hyperparameters.
6. Train the model with the best parameters and make predictions on the test set.
7. Evaluate the model using a confusion matrix and accuracy score.


## Results
- Best Parameter: {'max_depth': 4}

- Confusion Matrix:
[[ 59   1]
 [  2 109]]
- Accuracy Score: 0.9824561403508771


## I hope this is helpful. Let me know if there's anything else you need!
