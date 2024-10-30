# -*- coding: utf-8 -*-
"""
Created on Mon Sep  2 10:56:56 2024
直接导出预测结果
@author: 25304
"""
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
random_state_forall = 420

#导入模型
with open('model_etr.pickle', 'rb') as f:
    model_loaded = pickle.load(f)


# 导入验证数据
data_verify = pd.read_csv(r"HEA-hardness-verify.csv", encoding ='gb2312')

#导入原始数据
data_raw = pd.read_csv(r"8 Split.csv", encoding ='gb2312')



# Prepare the raw data
data_raw = data_raw.iloc[:, 1:]
X = data_raw.iloc[:, 1:]
Y = data_raw.iloc[:, 0]

# Prepare the raw data
data_verify = data_verify.iloc[:, 1:]
x = data_verify.iloc[:, 1:]
y = data_verify.iloc[:, 0]

# Normalize features
max_x = X.max()
min_x = X.min()
x = (x - min_x) / (max_x - min_x)


predicted_values = model_loaded.predict(x)


real_values = y
plt.figure()
plt.xlabel('Experimental viscosity/(poise)')
plt.ylabel('Predicted viscosity/(poise)')

# Plot training data points
plt.scatter(y, predicted_values, edgecolors='blue', facecolors='none', s=100, label='Training data')

# Add reference line
plt.plot([min(real_values), max(real_values)], [min(real_values), max(real_values)], color='red', linestyle='-', label='Reference Line')

# Add legend
plt.legend()

# Save and show the plot
plt.savefig('Pointbitmap.png', dpi=300)
plt.show()



 
r2_verify = r2_score(y, predicted_values)
rmse_verify = np.sqrt(mean_squared_error(y, predicted_values))

# 打印结果
print(f"Training R²: {r2_verify}")
print(f"Training RMSE: {rmse_verify} poise")

combined_data_verify = pd.DataFrame({
    'y': y,
    'predicted_values': predicted_values
})
combined_data_verify.to_csv('verify_data.csv', index=False)