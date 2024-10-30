# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 10:56:56 2024
直接导出预测结果
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

# Load the model
with open('model_ctbc.pickle', 'rb') as f:
    model_loaded = pickle.load(f)


# 导入验证数据
data_verify = pd.read_csv(r"Case 1 - verify.csv", encoding ='gb2312')

#导入原始数据
data_raw = pd.read_csv(r"9 Split.csv", encoding ='gb2312')

# Prepare the raw data
data_raw = data_raw.iloc[:, 1:]  # Drop the first column if it's an index or unwanted column
X = data_raw.iloc[:, 1:]  # Features
Y = data_raw.iloc[:, 0]   # Target labels

# Prepare the verification data
data_verify = data_verify.iloc[:, 1:]
x = data_verify.iloc[:, 1:]  # Features
y = data_verify.iloc[:, 0]   # Target labels

# Normalize features (optional for classification)
max_x = X.max()
min_x = X.min()
x = (x - min_x) / (max_x - min_x)

# Predict using the loaded model
predicted_classes = model_loaded.predict(x)

# Calculate accuracy
accuracy_verify = accuracy_score(y, predicted_classes)


# Calculate confusion matrix
cm = confusion_matrix(y, predicted_classes)

# Define custom labels for the confusion matrix
custom_labels = ['Non-Perovskite', 'Perovskite']  # Adjust as needed based on your classes

# Plot confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=custom_labels, yticklabels=custom_labels, cbar=False)

# Set font properties
plt.xlabel('Prediction', fontsize=14, fontname='Arial')
plt.ylabel('True Class', fontsize=14, fontname='Arial')
plt.xticks(fontsize=12, fontname='Arial')
plt.yticks(fontsize=12, fontname='Arial')

# Remove the right side ticks
plt.tick_params(axis='y', which='both', left=True, right=False)
plt.tick_params(axis='x', which='both', bottom=True, top=False)

plt.title('Validation set', fontsize=16, fontname='Arial')
plt.savefig('confusion_matrix.png', dpi=300)
plt.show()
#%%
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_auc_score
from itertools import cycle

# 假设你有3个类别，将真实标签和预测标签进行二值化
y_bin = label_binarize(y, classes=[0, 1])  # 真实类标签二值化
predicted_prob = model_loaded.predict_proba(x)  # 获取每个类的预测概率

# 计算两个类别的ROC曲线和AUC值
fpr = dict()
tpr = dict()
roc_auc = dict()

# 针对类别0（负类）绘制ROC曲线
fpr[0], tpr[0], _ = roc_curve(1 - y_bin[:, 0], predicted_prob[:, 0])  # 负类
roc_auc[0] = auc(fpr[0], tpr[0])

# 针对类别1（正类）绘制ROC曲线
fpr[1], tpr[1], _ = roc_curve(y_bin[:, 0], predicted_prob[:, 1])  # 正类
roc_auc[1] = auc(fpr[1], tpr[1])

# 绘制类别0和类别1的ROC曲线
plt.figure(figsize=(8, 6))

# 绘制类别0的ROC曲线
plt.plot(fpr[0], tpr[0], color='blue', lw=2, label=f'ROC curve of class non-perovskite (AUC = {roc_auc[0]:0.4f})')

# 绘制类别1的ROC曲线
plt.plot(fpr[1], tpr[1], color='darkorange', lw=2, label=f'ROC curve of class perovskite (AUC = {roc_auc[1]:0.4f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2)  # 对角线表示随机分类器
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=20, fontname='Arial')
plt.ylabel('True Positive Rate', fontsize=20, fontname='Arial')
plt.xticks(fontsize=18, fontname='Arial')
plt.yticks(fontsize=18, fontname='Arial')
plt.legend(loc="lower right", fontsize=12)
plt.savefig('roc_curve_two_classes.png', dpi=300)
plt.show()

#%%

# Print results
print(f"Classification Accuracy: {accuracy_verify * 100:.2f}%")
print(classification_report(y, predicted_classes))

# Save combined verification data
combined_data_verify = pd.DataFrame({
    'Actual': y,
    'Predicted': predicted_classes
})
combined_data_verify.to_csv('verify_classification_data.csv', index=False)


