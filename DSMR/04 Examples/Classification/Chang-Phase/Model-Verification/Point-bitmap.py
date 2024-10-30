# -*- coding: utf-8 -*-
"""
Created on Fri Jun 23 15:50:24 2023
绘制图像，
给定对应的数据集和模型，输出实验值和真实值的点图
@author: 25304
"""
#%%
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import seaborn as sns

random_state_forall = 420

def ClassificationPlot(data, model_loaded):
    # Prepare the data
    data = data.iloc[:, 1:]  # Drop the first column if needed
    x = data.iloc[:, 1:]  # Features
    y = data.iloc[:, 0]   # Target labels

    # Normalize features (optional for classification)
    max_x = x.max()
    min_x = x.min()
    x = (x - min_x) / (max_x - min_x)

    # Drop columns with all NaN values and replace remaining NaNs with 0
    x.dropna(axis=1, how='all', inplace=True)
    x.fillna(0, inplace=True)

    # Split the data into training and testing sets
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y, test_size=0.2, random_state=random_state_forall)

    # Predict classes using the loaded model
    predicted_classes_train = model_loaded.predict(Xtrain)
    predicted_classes_test = model_loaded.predict(Xtest)

    # Calculate accuracy
    accuracy_train = accuracy_score(Ytrain, predicted_classes_train)
    accuracy_test = accuracy_score(Ytest, predicted_classes_test)

    # Print results
    print(f"Training Accuracy: {accuracy_train * 100:.2f}%")
    print(f"Testing Accuracy: {accuracy_test * 100:.2f}%")
    print(classification_report(Ytest, predicted_classes_test))

    # Combine predictions for confusion matrix
    combined_y = pd.concat([Ytrain, Ytest])
    combined_predictions = np.concatenate([predicted_classes_train, predicted_classes_test])

    # Confusion matrix for all data
    cm_combined = confusion_matrix(combined_y, combined_predictions)

    # Define custom labels for the confusion matrix
    custom_labels = ['BCC', 'FCC','BCC+FCC']
    
    # Plot confusion matrix for all data
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_combined, annot=True, fmt='d', cmap='Blues', 
                xticklabels=custom_labels, yticklabels=custom_labels, cbar=False)
    
    # Set font properties
    plt.xlabel('Prediction', fontsize=14, fontname='Arial')
    plt.ylabel('True Class', fontsize=14, fontname='Arial')
    plt.xticks(fontsize=12, fontname='Arial')
    plt.yticks(fontsize=12, fontname='Arial')
    
    # Remove the right side ticks
    plt.tick_params(axis='y', which='both', left=True, right=False)
    plt.tick_params(axis='x', which='both', bottom=True, top=False)
    plt.title('Training & Testing set', fontsize=16, fontname='Arial')
    plt.savefig('confusion_matrix_combined.png', dpi=300)
    plt.show()

    
    return 1


# Load model and use
with open('model_ctbc.pickle', 'rb') as f:
    model_loaded = pickle.load(f)

# Import dataset
data = pd.read_csv(r"5 Split.csv", encoding='gb2312')

ClassificationPlot(data, model_loaded)
