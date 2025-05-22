import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, auc, roc_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier 
#building all kinds of evaluating parameters
from sklearn.metrics import classification_report 
from sklearn.metrics import matthews_corrcoef 
import xgboost as xgb
from xgboost import plot_importance

import math
import scipy.stats as ss
import warnings
warnings.filterwarnings('ignore')

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

df = pd.read_csv('C:/Users/danie/Documents/spyder/creditcard.csv')

# Step 2: Split the DataFrame into training and testing sets
# Set a seed for reproducibility
np.random.seed(42)

# Generate random indices for splitting
indices = np.random.permutation(df.index)
split_ratio = 0.8  # 80% training, 20% testing

# Calculate the split index
split_index = int(len(indices) * split_ratio)

# Split the indices into training and testing sets
train_indices, test_indices = indices[:split_index], indices[split_index:]

# Step 3: Create training and testing DataFrames
df_train = df.loc[train_indices]
df_train.columns = df_train.columns.str.lower().str.replace(' ', '_')
df_test = df.loc[test_indices]
df_test.columns = df_test.columns.str.lower().str.replace(' ', '_')

target='class'
    
X_train, X_val, y_train, y_val =df_train.drop([target],axis=1),df_test.drop([target],axis=1),df_train[target],df_test[target]    
seed = np.random.seed(6)

# LBGM演算法
param1 = {'n_estimators': 949, 'max_depth': 9, 'learning_rate': 0.031359496322390004, 'min_child_weight': 1.6586878455948058, 'min_child_samples': 18, 'subsample': 0.7435222447895633, 'subsample_freq': 2}
lgbm_opt = LGBMClassifier(**param1,random_state=seed,device="cpu")
sc=cross_val_score(lgbm_opt,X_train.values,y_train.values,cv=4, scoring = 'roc_auc').mean()       
print("CV score of LGBM Optuna is ",sc)
sc_f1 = cross_val_score(lgbm_opt,X_train.values,y_train.values,cv=4, scoring = 'f1').mean()
# 跑測試集
lgb_model_final = lgbm_opt.fit(X_train.values, y_train.values)
# 這是跳出0/1的結果
y_pred = lgb_model_final.predict(X_val.values)
accuracy_score(y_val, y_pred) 
# 這是跳出是0/1的機率
y_pred_proba = lgb_model_final.predict_proba(X_val)
# 根據閥值進行預測
# preds = (y_pred_proba >= 0.5).astype(int)
preds = y_pred_proba.argmax(axis=1)

# 衡量模型優劣
accuracy = accuracy_score(y_val, y_pred)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
print('accurecy :', str(accuracy))
cm = confusion_matrix(y_val, y_pred, labels=[ 0,1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['0','1'])
disp.plot()

# 計算 ROC 曲線
fpr, tpr, roc_thresholds = roc_curve(y_val, y_pred_proba[:, 1])
roc_auc = auc(fpr, tpr)

# 繪製 ROC 曲線
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], '--', color='gray', label='Random')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()


# RF演算法
#building the Random Forest Classifier
#random forest model creation 
params = {
    "n_estimators": 200,
    "max_features": 'sqrt',
    "max_depth": 50,
    "min_samples_split": 10,
    "min_samples_leaf": 4
}
rfc = RandomForestClassifier(random_state=seed, **params) 
rfc.fit(X_train, y_train) 
#predictions 
yPred = rfc.predict(X_val) 
yPred_proba = rfc.predict_proba(X_val)
# n_outliers = len(fraud) 
# n_errors = (yPred != yTest).sum() 
print("The model used is Random Forest classifier") 
  
acc = accuracy_score(y_val, yPred) 
print("The accuracy is {}".format(acc)) 
  
prec = precision_score(y_val, yPred) 
print("The precision is {}".format(prec)) 
  
rec = recall_score(y_val, yPred) 
print("The recall is {}".format(rec)) 
  
f1 = f1_score(y_val, yPred) 
print("The F1-Score is {}".format(f1)) 
  
cm = confusion_matrix(y_val, yPred, labels=[ 0,1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['0','1'])
disp.plot()

sc=cross_val_score(rfc,X_train.values,y_train.values,cv=4, scoring = 'roc_auc').mean()       
sc_f1 = cross_val_score(rfc,X_train.values,y_train.values,cv=4, scoring = 'f1').mean()
print(sc, sc_f1)

# 計算 ROC 曲線
fpr, tpr, roc_thresholds = roc_curve(y_val, yPred_proba[:, 1])
roc_auc = auc(fpr, tpr)
print(roc_auc)
# 繪製 ROC 曲線
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], '--', color='gray', label='Random')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

# XGBoost



xgb_model = xgb.XGBClassifier(
    learning_rate=0.05,
    n_estimators=300,
    gamma=0,
    subsample=0.8,
    max_depth = 5,
    colsample_bytree=0.8,
    objective='binary:logistic',
    n_jobs = 12,
    scale_pos_weight=1,
    seed=20231121,
    enable_categorical=True,
    tree_method='hist'
)

xgb_model.fit(X_train, y_train)

yPred_xgb = xgb_model.predict(X_val) 
yPred_proba_xgb = xgb_model.predict_proba(X_val)
# n_outliers = len(fraud) 
# n_errors = (yPred != yTest).sum() 
print("The model used is Random Forest classifier") 
  
acc = accuracy_score(y_val, yPred_xgb) 
print("The accuracy is {}".format(acc)) 
  
prec = precision_score(y_val, yPred_xgb) 
print("The precision is {}".format(prec)) 
  
rec = recall_score(y_val, yPred_xgb) 
print("The recall is {}".format(rec)) 
  
f1 = f1_score(y_val, yPred_xgb) 
print("The F1-Score is {}".format(f1)) 
  
cm = confusion_matrix(y_val, yPred_xgb, labels=[ 0,1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['0','1'])
disp.plot()

# 計算 ROC 曲線
fpr, tpr, roc_thresholds = roc_curve(y_val, yPred_proba_xgb[:, 1])
roc_auc = auc(fpr, tpr)
print(roc_auc)
# 繪製 ROC 曲線
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0, 1], [0, 1], '--', color='gray', label='Random')
plt.title('ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()

sc=cross_val_score(xgb_model,X_train.values,y_train.values,cv=4, scoring = 'roc_auc').mean()       
sc_f1 = cross_val_score(xgb_model,X_train.values,y_train.values,cv=4, scoring = 'f1').mean()
print(sc, sc_f1)











