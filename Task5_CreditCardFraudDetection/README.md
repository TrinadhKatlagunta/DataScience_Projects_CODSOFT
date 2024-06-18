# Credit Card Fraud Detection

This folder contains the code for the Credit Card Fraud Detection project developed as part of the Codsoft internship program.

## Instructions

Before running the program, please ensure you follow these instructions:

1. **Use Google Colab**: This program should be run in Google Colab.

2. **Download the Dataset**: The dataset is publicly available on Kaggle. Follow the steps below to download the dataset and upload it to your Google Drive.

3. **Upload Dataset to Google Drive**: Once downloaded, upload the `creditcard.csv` file to a folder in your Google Drive.

4. **Run the Program**: Execute the provided code in a Google Colab notebook to train and evaluate the logistic regression model for fraud detection.

## Download Dataset

To download the dataset, follow these steps:

1. Go to the following link: 
   - [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/adrianmcmahon/imdb-india-movies)

2. Download the `creditcard.csv` file from Kaggle.

## Upload Dataset to Google Drive

After downloading the dataset, follow these steps to upload it to Google Drive:

1. **Open Google Drive**:
   - Go to [Google Drive](https://drive.google.com/) in your web browser.

2. **Create a Folder (Optional)**:
   - You can create a new folder in Google Drive to organize your files. Right-click in Google Drive and select "New folder". Name it something like `CreditCardFraud`.

3. **Upload Your File**:
   - Click on the "New" button on the left side of the Google Drive interface.
   - Select "File upload" from the dropdown menu.
   - Navigate to the location of your downloaded `creditcard.csv` file on your local machine and select it to upload.
   - Move the uploaded file into the `CreditCardFraud` folder if you created one.

## Running the Program

1. **Open Google Colab**:
   - Go to [Google Colab](https://colab.research.google.com/) in your web browser.

2. **Create a New Notebook**:
   - Click on "File" > "New Notebook".

3. **Copy and Paste the Code**:
   - Copy the code provided below and paste it into a cell in your Colab notebook:

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from google.colab import drive

# Step 1: Mount Google Drive
drive.mount('/content/drive')

# Step 2: Provide the path to your file in Google Drive
file_path = '/content/drive/My Drive/CreditCardFraud/creditcard.csv'  # Update this path
file = pd.read_csv(file_path)

# Step 3: Data Analysis and Model Training
# Display the first 10 rows of the dataset
print(file.head(10))

# Display a summary of the dataset
print(file.describe())

# Check for missing values
print(file.isnull().sum())

# Display the class distribution
print(file['Class'].value_counts())

# Separate the dataset into normal and fraud classes
normal = file[file['Class'] == 0]
fraud = file[file['Class'] == 1]

# Display the shape of normal and fraud datasets
print(normal.shape)
print(fraud.shape)

# Describe the 'Amount' feature for normal and fraud classes
print(normal['Amount'].describe())
print(fraud['Amount'].describe())

# Sample the normal class to match the number of fraud cases
normal_sample = normal.sample(n=len(fraud), random_state=2)

# Combine the normal sample and fraud dataset
new_file = pd.concat([normal_sample, fraud], axis=0)

# Display the first 10 rows of the new balanced dataset
print(new_file.head(10))

# Display the class distribution of the new balanced dataset
print(new_file['Class'].value_counts())

# Split the dataset into features and target variable
X = new_file.drop(columns='Class', axis=1)
Y = new_file['Class']

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

# Initialize and train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

# Make predictions on the training set
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train) * 100
print(f"Training Data Accuracy: {training_data_accuracy:.2f}%")

# Make predictions on the testing set
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test) * 100
print(f"Test Data Accuracy: {test_data_accuracy:.2f}%")

# Display the confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(Y_test, X_test_prediction))

# Display the classification report
print("\nClassification Report:")
print(classification_report(Y_test, X_test_prediction))
