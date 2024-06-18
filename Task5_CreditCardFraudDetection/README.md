# Credit Card Fraud Detection

# This project demonstrates how to detect credit card fraud using a logistic regression model.
# The dataset used is a publicly available dataset of credit card transactions, where the 
# 'Class' column indicates whether a transaction is fraudulent (1) or not (0).

# Required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from google.colab import drive

# Step 1: Mount Google Drive
# Mount Google Drive to access the dataset stored in your Google Drive.
drive.mount('/content/drive')

# Step 2: Load the Dataset
# Provide the path to your file in Google Drive. Update the file_path variable with the path to your dataset.
file_path = '/content/drive/My Drive/CreditCardFraud/creditcard.csv' 
file = pd.read_csv(file_path)

# Step 3: Data Exploration
# Display the first 10 rows of the dataset to get an initial understanding of the data.
print(file.head(10))

# Display a summary of the dataset including statistics like mean, standard deviation, etc.
print(file.describe())

# Check for missing values in the dataset.
print(file.isnull().sum())

# Display the class distribution to understand the imbalance in the dataset.
print(file['Class'].value_counts())

# Step 4: Data Preprocessing
# Separate the dataset into normal (non-fraudulent) and fraud (fraudulent) transactions.
normal = file[file['Class'] == 0]
fraud = file[file['Class'] == 1]

# Display the shape of normal and fraud datasets to see the imbalance.
print(normal.shape)
print(fraud.shape)

# Describe the 'Amount' feature for normal and fraud classes.
print(normal['Amount'].describe())
print(fraud['Amount'].describe())

# Step 5: Balancing the Dataset
# Sample the normal class to match the number of fraud cases to balance the dataset.
normal_sample = normal.sample(n=len(fraud), random_state=2)

# Combine the normal sample and fraud dataset to create a new balanced dataset.
new_file = pd.concat([normal_sample, fraud], axis=0)

# Display the first 10 rows of the new balanced dataset.
print(new_file.head(10))

# Display the class distribution of the new balanced dataset to ensure it is balanced.
print(new_file['Class'].value_counts())

# Step 6: Splitting the Data
# Split the dataset into features (X) and target variable (Y).
X = new_file.drop(columns='Class', axis=1)
Y = new_file['Class']

# Split the dataset into training and testing sets.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Step 7: Model Training
# Initialize and train the Logistic Regression model.
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Step 8: Model Evaluation
# Make predictions on the training set.
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train) * 100
print(f"Training Data Accuracy: {training_data_accuracy:.2f}%")

# Make predictions on the testing set.
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test) * 100
print(f"Test Data Accuracy: {test_data_accuracy:.2f}%")

# Display the confusion matrix to see the performance of the model in terms of true/false positives/negatives.
print("Confusion Matrix:")
print(confusion_matrix(Y_test, X_test_prediction))

# Display the classification report to see detailed metrics like precision, recall, and F1-score.
print("\nClassification Report:")
print(classification_report(Y_test, X_test_prediction))
