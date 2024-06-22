import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from google.colab import drive

drive.mount('/content/drive')

file_path = '/content/drive/My Drive/CreditCardFraud/creditcard.csv' 
file = pd.read_csv(file_path)

print(file.head(10))

print(file.describe())

print(file.isnull().sum())

print(file['Class'].value_counts())

normal = file[file['Class'] == 0]
fraud = file[file['Class'] == 1]

print(normal.shape)
print(fraud.shape)

print(normal['Amount'].describe())
print(fraud['Amount'].describe())

normal_sample = normal.sample(n=len(fraud),random_state=2)

new_file = pd.concat([normal_sample,fraud],axis=0)

print(new_file.head(10))
print(new_file['Class'].value_counts())

X = new_file.drop(columns='Class', axis=1)
Y = new_file['Class']

X_train, X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)

model = LogisticRegression(max_iter=1000)
model.fit(X_train,Y_train)

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)*100
print(f"Training Data Accuracy: {training_data_accuracy:.2f}%")

X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction,Y_test)*100
print(f"Test Data Accuracy: {test_data_accuracy:.2f}%")

print("Confusion Matrix:")
conf_matrix=confusion_matrix(Y_test,X_test_prediction)
print(conf_matrix)
print("\nExplanation of Confusion Matrix:")
print(f"True Negatives  (TN): {conf_matrix[0][0]} - Correctly identified normal transactions")
print(f"False Positives (FP): {conf_matrix[0][1]} - Incorrectly identified fraud transactions as normal")
print(f"False Negatives (FN): {conf_matrix[1][0]} - Incorrectly identified normal transactions as fraud")
print(f"True Positives  (TP): {conf_matrix[1][1]} - Correctly identified fraud transactions")

print("\nClassification Report:")
print(classification_report(Y_test, X_test_prediction))
print("\nExplanation of Classification Report:\n")
print("Precision: Indicates how many of the predicted positive cases were actual positive cases (fraud).\n")
print("Recall (Sensitivity): Indicates how many of the actual positive cases (fraud) were correctly identified by the model.\n")
print("F1 Score: The harmonic mean of precision and recall, providing a balance between the two metrics.\n")
print("Support: The number of actual occurrences of each class in the test data.")
