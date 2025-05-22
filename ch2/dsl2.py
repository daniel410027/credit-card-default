import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import math
import scipy.stats as ss
import warnings
from datetime import datetime
warnings.filterwarnings('ignore')

file_path = r"C:\Users\danie\Documents\spyder\fraudTest.csv"
selected_column = pd.read_csv(file_path)
np.random.seed(42)

df = selected_column.iloc[:, [4, 5, 8, 12, 13, 14, 15, 16, 17, 19, 20, 21, 22]]  # 選擇第 3、6、8、10 個欄位
df = df.iloc[:, [1, 3, 4, 5, 6, 8, 9, 10, 11, 12]]
df['dob'] = df['dob'].apply(lambda x: int(x[:4]))  # 提取年份部分并转换为整数
df['dob'] = 2024 - df['dob']  # 计算年龄
# Step 2: Split the DataFrame into training and testing sets
# Set a seed for reproducibility

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

print(df_train.info())
print(df_test.info())

df_train

target='is_fraud'

print(df_train.skew())
print(df_test.skew())

print(df_train.corr()[target].sort_values(ascending=False))
print('-------------- \n')
print(df_test.corr()[target].sort_values(ascending=False))

def skew_autotransform(DF, include = None, exclude = None, plot = False, threshold = 1, exp = False):
    
    #Get list of column names that should be processed based on input parameters
    if include is None and exclude is None:
        colnames = DF.columns.values
    elif include is not None:
        colnames = include
    elif exclude is not None:
        colnames = [item for item in list(DF.columns.values) if item not in exclude]
    else:
        print('No columns to process!')
    
    #Helper function that checks if all values are positive
    def make_positive(series):
        minimum = np.amin(series)
        #If minimum is negative, offset all values by a constant to move all values to positive teritory
        if minimum <= 0:
            series = series + abs(minimum) + 0.01
        return series
    
    
    #Go throug desired columns in DataFrame
    for col in colnames:
        before_df=pd.DataFrame(DF[col], columns=[col])
        #Get column skewness
        skew = DF[col].skew()
        transformed = True
        
        if plot:
            #Prep the plot of original data
            sns.set_style("darkgrid")
            sns.set_palette("Blues_r")
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            ax1 = sns.distplot(DF[col], ax=axes[0])
            ax1.set(xlabel='Original ' + col)
#             sns.distplot(before_df[col]);
#             fig = plt.figure()
#             res = stats.probplot(before_df[col], plot=plt)       
        #If skewness is larger than threshold and positively skewed; If yes, apply appropriate transformation
        if abs(skew) > threshold and skew > 0:
            skewType = 'positive'
            #Make sure all values are positive
            DF[col] = make_positive(DF[col])
            
            if exp:
               #Apply log transformation 
               DF[col] = DF[col].apply(math.log)
            else:
                #Apply boxcox transformation
                DF[col] = ss.boxcox(DF[col])[0]
            skew_new = DF[col].skew()
         
        elif abs(skew) > threshold and skew < 0:
            skewType = 'negative'
            #Make sure all values are positive
            DF[col] = make_positive(DF[col])
            
            if exp:
               #Apply exp transformation 
               DF[col] = DF[col].pow(10)
            else:
                #Apply boxcox transformation
                DF[col] = ss.boxcox(DF[col])[0]
            skew_new = DF[col].skew()
        
        else:
            #Flag if no transformation was performed
            transformed = False
            skew_new = skew
        
        #Compare before and after if plot is True
        if plot:
            print('\n ------------------------------------------------------')     
            if transformed:
                print('\n %r had %r skewness of %2.2f' %(col, skewType, skew))
                print('\n Transformation yielded skewness of %2.2f' %(skew_new))
                sns.set_palette("Paired")
                ax2 = sns.distplot(DF[col], ax=axes[1], color = 'r')
                ax2.set(xlabel='Transformed ' + col)
                sns.distplot(DF[col]);
                fig = plt.figure()
                res = stats.probplot(DF[col], plot=plt)
                plt.title('After')
                plt.xlabel('Theoretical Quantiles')
                plt.ylabel('Ordered Values')
                
                plt.show()
            else:
                print('\n NO TRANSFORMATION APPLIED FOR %r . Skewness = %2.2f' %(col, skew))
                ax2 = sns.distplot(DF[col], ax=axes[1])
                ax2.set(xlabel='NO TRANSFORM ' + col)
                plt.show()                
    return DF

num_featuers=df_train.columns.tolist()
num_featuers.remove(target)

train_skewness = df_train[num_featuers].select_dtypes(include=['number']).apply(lambda x: x.skew())
    
    # Filter features with skewness greater than threshold or less than -threshold
train_skewed_features = train_skewness[(train_skewness > 0.5) | (train_skewness < -0.5)]

train_skewed_feats= list(train_skewed_features.keys())

df_train[train_skewed_feats]=skew_autotransform(df_train[train_skewed_feats].copy(deep=True), plot = True, exp = False, threshold = 0.5).values

test_skewness = df_test[num_featuers].select_dtypes(include=['number']).apply(lambda x: x.skew())
    
    # Filter features with skewness greater than threshold or less than -threshold
test_skewed_features = test_skewness[(test_skewness > 0.5) | (test_skewness < -0.5)]

test_skewed_feats= list(test_skewed_features.keys())

df_test[test_skewed_feats]=skew_autotransform(df_test[test_skewed_feats].copy(deep=True), plot = True, exp = False, threshold = 0.5).values

print(df_train.corr()[target].sort_values(ascending=False))
print('-------------- \n')
print(df_test.corr()[target].sort_values(ascending=False))

def multi_hist_plot(df1,df2,feature):
    
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 12))
    sns.histplot(data = df1 , ax=axes[0], x = feature , hue = target,palette = sns.color_palette(["yellow" , "green",'red','blue','black','orange','purple']) ,multiple = "stack" ).set_title(f"{feature} Vs ")
    axes[0].set_title('Histogram of Train Data '+feature)
    sns.histplot(data = df2 , ax=axes[1], x = feature , hue = target,palette = sns.color_palette(["yellow" , "green",'red','blue','black','orange','purple']) ,multiple = "stack" ).set_title(f"{feature} Vs ")
    axes[1].set_title('Histogram of Test Data '+feature)
    plt.tight_layout()  # Adjust layout to prevent overlap    
    plt.show()

for i in num_featuers:
    multi_hist_plot(df_train,df_test,i)
    
X_train, X_val, y_train, y_val =df_train.drop([target],axis=1),df_test.drop([target],axis=1),df_train[target],df_test[target]    
    
seed = np.random.seed(6)

# param1 = {'n_estimators': 844, 'max_depth': 5, 'learning_rate': 0.011533926188770152, 
#           'min_child_weight': 0.7353926016580375, 'min_child_samples': 22, 
#           'subsample': 0.7440626280651244, 'subsample_freq': 3
# #           , 'colsample_bytree': 0.5193554106905941
#          }

param1 = {'n_estimators': 844, 'max_depth': 5, 'learning_rate': 0.011533926188770152, 
          'min_child_weight': 0.7353926016580375, 'min_child_samples': 22, 
          'subsample': 0.7440626280651244, 'subsample_freq': 3
#           , 'colsample_bytree': 0.5193554106905941
         }


lgbm_opt = LGBMClassifier(**param1,random_state=seed,device="cpu")
sc=cross_val_score(lgbm_opt,X_train.values,y_train.values,cv=4, scoring = 'roc_auc').mean()    
    
print("CV score of LGBM Optuna is ",sc)  

lgb_model_final = lgbm_opt.fit(X_train.values, y_train.values)

y_pred = lgb_model_final.predict(X_val.values)

accuracy_score(y_val, y_pred) 

y_pred_proba = lgb_model_final.predict_proba(X_val)

preds = y_pred_proba.argmax(axis=1)

accuracy = accuracy_score(y_val, y_pred)*100
print('accurecy :', str(accuracy))
cm = confusion_matrix(y_val, preds, labels=[ 0,1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['0','1'])
disp.plot()