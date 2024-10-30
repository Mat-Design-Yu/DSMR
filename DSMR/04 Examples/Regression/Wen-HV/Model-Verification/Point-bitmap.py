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
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
random_state_forall = 420
    
def Pointbitmap(data, model_loaded):
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import train_test_split

    # Prepare the data
    data = data.iloc[:, 1:]
    # data.drop_duplicates(inplace=True)  # Remove duplicate data
    # data = data.reset_index(drop=True)
    x = data.iloc[:, 1:]
    y = data.iloc[:, 0]

    # Normalize features
    max_x = x.max()
    min_x = x.min()
    x = (x - min_x) / (max_x - min_x)

    # Drop columns with all NaN values and replace remaining NaNs with 0
    x.dropna(axis=1, how='all', inplace=True)
    x.fillna(0, inplace=True)

    # Split the data into training and testing sets
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(x, y, test_size=0.2, random_state=random_state_forall)

    # Predict values using the loaded model
    predicted_values_train = model_loaded.predict(Xtrain)
    predicted_values_test = model_loaded.predict(Xtest)

    # Combine the values into a single DataFrame
    combined_data_train = pd.DataFrame({
        'Ytrain': Ytrain,
        'predicted_values_train': predicted_values_train
    })
    combined_data_test = pd.DataFrame({
        'Ytest': Ytest,
        'predicted_values_test': predicted_values_test
    })
    
    # 计算训练集和测试集的 R² 和 RMSE
    r2_train = r2_score(Ytrain, predicted_values_train)
    rmse_train = np.sqrt(mean_squared_error(Ytrain, predicted_values_train))
    
    r2_test = r2_score(Ytest, predicted_values_test)
    rmse_test = np.sqrt(mean_squared_error(Ytest, predicted_values_test))
    
    # 打印结果
    print(f"Training R²: {r2_train}")
    print(f"Training RMSE: {rmse_train} poise")
    print(f"Testing R²: {r2_test}")
    print(f"Testing RMSE: {rmse_test} poise")

    combined_data_train.to_csv('train_data.csv', index=False)
    combined_data_test.to_csv('test_data.csv', index=False)

    print("train_data.csv and test_data.csv files have been created and written successfully.")

    # Plotting the results
    real_values = y
    plt.figure()
    plt.xlabel('Experimental viscosity/(poise)')
    plt.ylabel('Predicted viscosity/(poise)')

    # Plot training data points
    plt.scatter(Ytrain, predicted_values_train, edgecolors='black', facecolors='none', s=100, label='Training data')

    # Plot testing data points
    plt.scatter(Ytest, predicted_values_test, color='blue', s=100, label='Testing data')

    # Add reference line
    plt.plot([min(real_values), max(real_values)], [min(real_values), max(real_values)], color='red', linestyle='-', label='Reference Line')

    # Add legend
    plt.legend()

    # Save and show the plot
    plt.savefig('Pointbitmap.png', dpi=300)
    plt.show()

    return 1


#%%
#ML+PySR-ETR
# # 加载模型并使用
# with open('model_etr.pickle', 'rb') as f:
#     model_loaded = pickle.load(f)
    
# # 导入数据集
# data = pd.read_csv(r"Addequations -39.csv", encoding ='gb2312')

# Pointbitmap(data,model_loaded)


#%%
#ML-XGBR
# 加载模型并使用
# with open('model_xgb.pickle', 'rb') as f:
#     model_loaded = pickle.load(f)
with open('model_etr.pickle', 'rb') as f:
    model_loaded = pickle.load(f)

# 导入数据集
# data = pd.read_csv(r"ViscosityData-all.csv", encoding ='gb2312')
# data = pd.read_csv(r"ViscosityData-raw.csv", encoding ='gb2312')
data = pd.read_csv(r"8 Split.csv", encoding ='gb2312')


Pointbitmap(data,model_loaded)
#%%