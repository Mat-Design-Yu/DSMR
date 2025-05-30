# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 09:26:18 2023
自动化切换数据集，实现长时间自动化计算
@author: 25304
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  #分配训练集与测试集
from sklearn.model_selection import GridSearchCV    #网格搜索
from sklearn.model_selection import cross_val_score #交叉验证
import matplotlib.pyplot as plt   #画图
from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
from skopt import BayesSearchCV
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_log_error
import pickle
import math

#评估指标
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from sklearn.model_selection import cross_val_score, cross_val_predict


#常用分类算法
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC  # Support Vector Classifier
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier as RFC
from xgboost import XGBClassifier as XGBC
from catboost import CatBoostClassifier as CTBC

#%% 数据切分
num_splits = 10


data = pd.read_csv(r"5 Split.csv"
                   #, encoding ='gb2312'
                   )
data = data.iloc[:,1:]
df = data

split_size = math.ceil(len(df)/num_splits)


list_of_dfs = []
# 创建一个空的DataFrame用于存储结果
result_df = pd.DataFrame()

# 循环进行分割和组合
k=0
for i in range(num_splits):  
    # 从原始DataFrame中按顺序抽取一份数据
    elimination_df = data.iloc[k: k+split_size]        
    # 将抽取的数据删除，保留想保留的部分
    result_df = pd.merge(df, elimination_df, how='left', indicator=True).query('_merge == "left_only"').drop('_merge', axis=1)
    
    list_of_dfs.append(result_df)
    if k >= len(df):
        print("Breaking loop as k exceeds the length of DataFrame.")
        break
    else:
        k = k+split_size
# 打印结果DataFrame
print(list_of_dfs)

# 创建一个目录用于存放 CSV 文件
output_directory = 'datasets'
os.makedirs(output_directory, exist_ok=True)

# 循环遍历 DataFrame 列表，将每个 DataFrame 转化为 CSV 文件，并保留样本索引
for index, df in enumerate(list_of_dfs):
    csv_filename = os.path.join(output_directory, f'{index + 1} Split.csv')
    df.to_csv(csv_filename, index=True, index_label='Index')
    print(f'DataFrame {index + 1} saved as CSV: {csv_filename}')

#%% Auto-APE部分
#全局参数
random_state_forall = 420
num_of_models = 6
n_iter_forall = 100
name_of_models = ['lr','svc','dtc','rfc','xgbc',
                  'ctbc']


def Core(data):
    data = data.iloc[:,1:]
    data.drop_duplicates(inplace=True)  #剔除重复的数据
    data = data.reset_index(drop=True)
    x = data.iloc[:,1:]
    y = data.iloc[:,0]


    #对特征进行归一化，避免数据量级对结果的影响，此处目标y不进行归一化
    max_x = x.max()
    min_x = x.min()
    x = (x - min_x) / (max_x - min_x)
    # max_y = y.max()
    # min_y = y.min()
    # y = (y - min_y) / (max_y - min_y)

    x.dropna (axis=1, how='all', inplace=True) #删除含Nan的列,直接删除全为Nan的列，如果不删，后面相关系数筛选模型会没有模型通过
    x.fillna(0, inplace=True) #将剩下的Nan替换为0，

    ###七三、八二分训练集和测试集######
    from sklearn.model_selection import train_test_split
    Xtrain,Xtest,Ytrain,Ytest = train_test_split(x,y,test_size=0.2,random_state=random_state_forall)
    
    #------支持float的步进函数--------------
    def float_range(start, stop, step):
        ''' 支持 float 的步进函数
    
            输入 Input:
                start (float)  : 计数从 start 开始。默认是从 0 开始。
                end   (float)  : 计数到 stop 结束，但不包括 stop。
                step (float)  : 步长，默认为 1，如为浮点数，参照 steps 小数位数。
    
            输出 Output:
                浮点数列表
    
            例子 Example:
                >>> print(float_range(3.612, 5.78, 0.22))
                [3.612, 3.832, 4.052, 4.272, 4.492, 4.712, 4.932, 5.152, 5.372]
        '''
        start_digit = len(str(start))-1-str(start).index(".") # 取开始参数小数位数
        stop_digit = len(str(stop))-1-str(stop).index(".")    # 取结束参数小数位数
        step_digit = len(str(step))-1-str(step).index(".")    # 取步进参数小数位数
        digit = max(start_digit, stop_digit, step_digit)      # 取小数位最大值
        return [(start*10**digit+i*step*10**digit)/10**digit for i in range(int((stop-start)//step))]
    
    
    #------构建基准模型--------------
    
    lr = LR()       # Logistic Regression
    svc = SVC()     # Support Vector Classifier
    dtc = DTC()     # Decision Tree Classifier
    rfc = RFC()     # Random Forest Classifier
    xgbc = XGBC()   # XGBoost Classifier
    ctbc = CTBC()   # CatBoost Classifier

    #------贝叶斯参数调优--------------

    # Logistic Regression
    lr_grid_param = { 
        'C': (1e-6, 1e+6, 'log-uniform')  # Regularization strength
    }
    lr_Bayes = BayesSearchCV(lr, lr_grid_param, n_iter=n_iter_forall, random_state=random_state_forall, n_jobs=-1)
    lr_Bayes.fit(Xtrain, Ytrain)
    
    # Support Vector Classifier
    svc_grid_param = {
        'C': (1e-6, 1e+6, 'log-uniform'),  # Regularization parameter
        'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
        'gamma': (1e-6, 1e+1, 'log-uniform')  # Kernel coefficient for 'rbf', 'poly' and 'sigmoid'
    }
    svc_Bayes = BayesSearchCV(svc, svc_grid_param, n_iter=n_iter_forall, random_state=random_state_forall, n_jobs=-1)
    svc_Bayes.fit(Xtrain, Ytrain)
    
    # Decision Tree Classifier
    dtc_grid_param = {
        'max_depth': (1, 50,10),  # Maximum depth of the tree
        'min_samples_split': (2, 20,4),  # Minimum number of samples to split a node
        'min_samples_leaf': (1, 20,4)    # Minimum number of samples required at a leaf node
    }
    dtc_Bayes = BayesSearchCV(dtc, dtc_grid_param, n_iter=n_iter_forall, random_state=random_state_forall, n_jobs=-1)
    dtc_Bayes.fit(Xtrain, Ytrain)
    
    # Random Forest Classifier
    rfc_grid_param = {
        'n_estimators': (10, 1000,100),  # Number of trees
        'max_depth': (1, 50,10),
        'min_samples_split': (2, 20,4),
        'min_samples_leaf': (1, 20,4)
    }
    rfc_Bayes = BayesSearchCV(rfc, rfc_grid_param, n_iter=n_iter_forall, random_state=random_state_forall, n_jobs=-1)
    rfc_Bayes.fit(Xtrain, Ytrain)
    
    # XGBoost Classifier   
    xgbc_grid_param = {
        'n_estimators': (50, 500),  # Number of boosting rounds
        'learning_rate': (0.01, 1.0, 'log-uniform'),  # Learning rate
        'max_depth': (3, 50),  # Maximum tree depth
        'colsample_bytree': (0.1, 1.0),  # Feature fraction used per tree
    }
    xgbc_Bayes = BayesSearchCV(xgbc, xgbc_grid_param, n_iter=n_iter_forall, random_state=random_state_forall, n_jobs=-1)
    xgbc_Bayes.fit(Xtrain, Ytrain)
    
    # CatBoost Classifier
    ctbc_grid_param = {
        'iterations': (100, 1000),  # Number of boosting iterations
        'depth': (3, 10),  # Depth of the tree
        'learning_rate': (0.01, 1.0, 'log-uniform'),  # Learning rate
        'l2_leaf_reg': (1, 10)  # L2 regularization term on leaf weights
    }
    ctbc_Bayes = BayesSearchCV(ctbc, ctbc_grid_param, n_iter=n_iter_forall, random_state=random_state_forall, n_jobs=-1)
    ctbc_Bayes.fit(Xtrain, Ytrain)

    
    print("参数择优完毕")
    #--------使用贝叶斯调优参数重新fit-----------------
    # Logistic Regression
    lr = LR(
        **lr_Bayes.best_params_
    ).fit(Xtrain, Ytrain)
    
    # Support Vector Classifier
    svc = SVC(
        **svc_Bayes.best_params_
    ).fit(Xtrain, Ytrain)
    
    # Decision Tree Classifier
    dtc = DTC(
        **dtc_Bayes.best_params_
    ).fit(Xtrain, Ytrain)
    
    # Random Forest Classifier
    rfc = RFC(
        **rfc_Bayes.best_params_
    ).fit(Xtrain, Ytrain)
    
    # XGBoost Classifier
    xgbc = XGBC(
        **xgbc_Bayes.best_params_
    ).fit(Xtrain, Ytrain)
    
    # CatBoost Classifier
    ctbc = CTBC(
        **ctbc_Bayes.best_params_
    ).fit(Xtrain, Ytrain)

    
    #---将构建的模型放入列表--------
    models=[]
    models.append(lr)
    models.append(svc)
    models.append(dtc)
    models.append(rfc)
    models.append(xgbc)
    models.append(ctbc)

    
    
    #预测
    yhat_lr = lr.predict(Xtest)
    yhat_svc = svc.predict(Xtest)
    yhat_dtc = dtc.predict(Xtest)
    yhat_rfc = rfc.predict(Xtest)
    yhat_xgbc = xgbc.predict(Xtest)
    yhat_ctbc = ctbc.predict(Xtest)
    
    yhat_list=[]
    for i in range(len(models)):
        yhat_list.append(models[i].predict(Xtest))
    
    
    model_pass_check_list = models
    
    
    
    # ---------Define evaluation metrics--------
    evaluations = [
        'Accuracy', 'Precision', 'Recall', 'F1',
        '10kf_Accuracy', '10kf_Precision', '10kf_Recall', '10kf_F1',
    ]
    
    acc_data = []
    for i in range(len(model_pass_check_list)):
        # Model evaluations
        model_evaluations = []
    
        # Predictions on test set
        y_pred = model_pass_check_list[i].predict(Xtest)
        
        # Classification metrics on test set
        model_evaluations.append(accuracy_score(Ytest, y_pred))  # Accuracy
        model_evaluations.append(precision_score(Ytest, y_pred, average='weighted'))  # Precision (weighted for multi-class)
        model_evaluations.append(recall_score(Ytest, y_pred, average='weighted'))  # Recall (weighted)
        model_evaluations.append(f1_score(Ytest, y_pred, average='weighted'))  # F1 Score (weighted)
        
        # # ROC AUC (Only valid for binary or multilabel-indicator format)
        # if len(set(Ytest)) == 2:
        #     model_evaluations.append(roc_auc_score(Ytest, model_pass_check_list[i].predict_proba(Xtest)[:, 1]))  # ROC AUC
        # else:
        #     model_evaluations.append(roc_auc_score(Ytest, model_pass_check_list[i].predict_proba(Xtest), multi_class='ovr', average='weighted'))
    
        # model_evaluations.append(log_loss(Ytest, model_pass_check_list[i].predict_proba(Xtest)))  # Log Loss
    
        # Cross-validation metrics (10-fold)
        model_evaluations.append(cross_val_score(model_pass_check_list[i], x, y, cv=10, scoring='accuracy').mean())  # 10kf Accuracy
        model_evaluations.append(cross_val_score(model_pass_check_list[i], x, y, cv=10, scoring='precision_weighted').mean())  # 10kf Precision
        model_evaluations.append(cross_val_score(model_pass_check_list[i], x, y, cv=10, scoring='recall_weighted').mean())  # 10kf Recall
        model_evaluations.append(cross_val_score(model_pass_check_list[i], x, y, cv=10, scoring='f1_weighted').mean())  # 10kf F1 Score
    
        # # 10kf ROC AUC (if binary classification)
        # if len(set(y)) == 2:
        #     model_evaluations.append(cross_val_score(model_pass_check_list[i], x, y, cv=10, scoring='roc_auc').mean())  # 10kf ROC AUC
        # else:
        #     model_evaluations.append(cross_val_score(model_pass_check_list[i], x, y, cv=10, scoring='roc_auc_ovr_weighted').mean())
    
        # Append to results
        acc_data.append(model_evaluations)

    #转化为DataFrame格式
    acc_df = pd.DataFrame(acc_data,index=model_pass_check_list,columns=evaluations,dtype=float)


    # Model Sorting
    sort_name_list = [
        "Accuracy_sort", "Precision_sort", "Recall_sort", "F1_sort", 
        "10kf_Accuracy_sort", "10kf_Precision_sort", "10kf_Recall_sort", "10kf_F1_sort",
    ]
    
    for i in range(len(evaluations)):
        acc_df.sort_values(by=evaluations[i], inplace=True, ascending=True)
        sort_index = list(range(len(model_pass_check_list)))
        sort_index.reverse()
        acc_df[sort_name_list[i]] = sort_index
        
        # # For Log Loss, we want to sort in ascending order (lower is better)
        # if sort_name_list[i] == "Log_Loss_sort":
        #     acc_df[sort_name_list[i]] = sort_index
        # else:
        #     # For accuracy, precision, recall, F1, and ROC AUC, higher values are better, so reverse the index
        #     sort_index.reverse()
        #     acc_df[sort_name_list[i]] = sort_index
    
    # Overall sorting score
    # Summing across sorted ranks for all metrics to get a total score (lower is better)
    acc_df['sort_score'] = acc_df.iloc[:, len(evaluations):].sum(axis=1)
    
    # Sorting models by the total sort score, with better models ranked higher
    acc_df.sort_values(by='sort_score', inplace=True, ascending=True)
    
    # Return the sorted DataFrame and the models
    return acc_df, models



#%%-----结果存储----------------

import pandas as pd
import os
import joblib
import shutil

# 假设你的数据集文件夹名称为datasets
dataset_folder = "datasets"
output_folder = "results"  # 用于保存结果的文件夹名称

# 检查结果文件夹是否存在，如果不存在则创建
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

name_of_models = ['lr','svc','dtc','rfc','xgbc',
                  'ctbc']

# 遍历数据集文件夹中的文件
for filename in os.listdir(dataset_folder):
    if filename.endswith(".csv"):
        dataset_name = os.path.splitext(filename)[0]  # 提取文件名（不包含扩展名）作为数据集名称

        dataset_path = os.path.join(dataset_folder, filename)  # 构建数据集文件的完整路径
        output_dataset_folder = os.path.join(output_folder, dataset_name)  # 构建结果文件夹的完整路径

        # 检查结果文件夹是否存在，如果不存在则创建
        if not os.path.exists(output_dataset_folder):
            os.makedirs(output_dataset_folder)

        data = pd.read_csv(dataset_path)  # 读取数据集为 DataFrame
        # 在此处调用 core() 函数进行计算，将结果保存到结果文件夹中
        result = Core(data)
        acc_df, models = result

        # 将评价指标保存为CSV文件
        acc_csv_filename = os.path.join(output_dataset_folder, "modelAcc.csv")
        acc_df.to_csv(acc_csv_filename)

        save_directory = os.path.join(output_dataset_folder, "models")  # 模型保存的文件夹名称
        # 检查目录是否存在，如果不存在则创建
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)

        # 遍历模型列表并保存为pickle文件
        for name, model in zip(name_of_models,models):
            model_name = f'model_{name}.pickle'
            model_path = os.path.join(save_directory, model_name)
            
            with open(model_path, 'wb') as model_file:
                pickle.dump(model, model_file)
            
            print(f"保存模型 {model_name}")
        
        print("模型保存完成。")

        # 将评价指标CSV文件复制到模型文件夹中
        acc_csv_destination = os.path.join(save_directory, "modelAcc.csv")
        shutil.copyfile(acc_csv_filename, acc_csv_destination)
        print(f"复制评价指标文件到 {acc_csv_destination} 完成")

        # 将测试数据集保存为CSV文件
        data_filename = os.path.join(output_dataset_folder, filename)
        data.to_csv(data_filename, index=False)
#%% 统计结果


# 指定文件夹路径
base_directory = 'results'

# 存储结果的列表
result_data = []

# 遍历子文件夹，并按文件名排序
subfolders = sorted(os.listdir(base_directory))

for subfolder in subfolders:
    subfolder_path = os.path.join(base_directory, subfolder)
    
    # 检查是否是文件夹
    if os.path.isdir(subfolder_path):
        # 构建 modelacc.csv 文件路径
        csv_path = os.path.join(subfolder_path, 'modelacc.csv')
        
        # 检查文件是否存在
        if os.path.exists(csv_path):
            # 读取 CSV 文件
            df = pd.read_csv(csv_path)
            
            # 提取 10kf_RMSE 列的第一个数值
            first_value = df['10kf_Accuracy'].iloc[0]
            
            # 将结果添加到列表中，包括子文件夹名称
            result_data.append({'Subfolder': subfolder, 'First_Value': abs(first_value)})

# 创建包含结果的 DataFrame
result_df = pd.DataFrame(result_data)

# 打印结果 DataFrame
print(result_df)

# 将 DataFrame 输出为 CSV 文件
output_csv_path = 'Ranking_statistics.csv'
result_df.to_csv(output_csv_path, index=False)

# 打印输出文件路径
print(f'Results saved as CSV: {output_csv_path}')