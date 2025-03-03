import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.ensemble import VotingRegressor,RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import ElasticNet,Ridge,Lasso
from sklearn.metrics import mean_squared_error
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt
from pre_processing_pipeline import full_pipeline
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from feature_add import add_new_features,add_new_features_no_clean

path_Elastic_net = os.path.join('House_Price_Prediction_Advanced_Regression_Techniques_Kaggle','model_pickle','ElasticNet.pkl')
path_Svr = os.path.join('House_Price_Prediction_Advanced_Regression_Techniques_Kaggle','model_pickle','SVR.pkl')
path_XGBoost = os.path.join('House_Price_Prediction_Advanced_Regression_Techniques_Kaggle','model_pickle','XGBoost.pkl')
path_ridge= os.path.join('House_Price_Prediction_Advanced_Regression_Techniques_Kaggle','model_pickle','Ridge.pkl')
path_lasso= os.path.join('House_Price_Prediction_Advanced_Regression_Techniques_Kaggle','model_pickle','Lasso.pkl')
path_forest = os.path.join('House_Price_Prediction_Advanced_Regression_Techniques_Kaggle','model_pickle','RandomForest.pkl')


Elastic_net = joblib.load(path_Elastic_net)
Svr = joblib.load(path_Svr)
XGBoost = joblib.load(path_XGBoost)
ridge= joblib.load(path_ridge)
lasso=joblib.load(path_lasso)
forest = joblib.load(path_forest)


#load data
path_train = os.path.join("House_Price_Prediction_Advanced_Regression_Techniques_Kaggle","data_set","train.csv")
df_train = pd.read_csv(path_train)
df_train = add_new_features(df_train)


#train test split
train_set , test_set = train_test_split(df_train, test_size=0.2, random_state=42)
X_train = train_set.drop(columns = ['SalePrice','LogSalePrice'])
y_train = train_set['LogSalePrice']
X_test = test_set.drop(columns = ['SalePrice','LogSalePrice'])
y_test = test_set['LogSalePrice']

Elastic_net_param = Elastic_net.get_params()
Svr_param = Svr.get_params()
XGBoost_param = XGBoost.get_params()
ridge_param = ridge.get_params()
lasso_param = lasso.get_params()
forest_param = forest.get_params()

Elastic_net_model = ElasticNet(**Elastic_net_param)
Svr_model = SVR(**Svr_param)
XGBoost_model = XGBRegressor(**XGBoost_param)
ridge_model = Ridge(**ridge_param)
lasso_model = Lasso(**lasso_param)
forest_model = RandomForestRegressor(**forest_param)
# CatBoost_model = CatBoostRegressor(
#     silent=True,
#     bagging_temperature=0,
#     border_count=16,
#     depth=3,
#     iterations=500,
#     l2_leaf_reg=10,
#     learning_rate=0.1,
#     rsm=0.9
# )
CatBoost_model = CatBoostRegressor(
    silent=True,
    bagging_temperature=0,
    border_count=8,
    depth=4,
    iterations=500,
    l2_leaf_reg=10,
    learning_rate=0.1,
    rsm=0.9
)


voting_reg = VotingRegressor(estimators=[  
    ('svr', Svr_model), 
    ('Ri',ridge_model),
    ('La',lasso_model),
    ('Fr',forest_model), 
    ('XG',XGBoost_model),
    ('net', Elastic_net_model),
    ('cat',CatBoost_model)
    
], weights = [1,1,1,0.5,5,1,5])



X_train_tra = full_pipeline.fit_transform(X_train)
y_train = y_train.to_numpy()
voting_reg.fit(X_train_tra,y_train)

path_voting_reg = os.path.join('House_Price_Prediction_Advanced_Regression_Techniques_Kaggle','model_pickle','voting_reg.pkl')

joblib.dump(voting_reg,path_voting_reg)

# testing fit visualization
X_test_tra = full_pipeline.transform(X_test)
y_test = y_test.to_numpy()
y_test_pred = voting_reg.predict(X_test_tra)
res_test= (np.exp(y_test) - np.exp(y_test_pred)) #original data is log transformed
   
    
sns.histplot(res_test, kde =True , stat='probability')
plt.xlabel('residuals', fontsize=12)  # x-axis label
plt.ylabel('Frequency', fontsize=12)  # y-axis label
plt.title('test data', fontsize=14)  # Title
    


score_log_test = np.sqrt(mean_squared_error(y_test_pred,y_test))  # MSE 
score_test = np.sqrt(mean_squared_error(np.exp(y_test_pred),np.exp(y_test)))
print('test_data')
# print('MSE_log: ', score_log_test)
print('MSE: ', score_test)
print( score_log_test)
    

# training fit visualization
y_train_pred = voting_reg.predict(X_train_tra)
res_train = (np.exp(y_train) - np.exp(y_train_pred))

    
sns.histplot(res_train, kde =True , stat='probability')
plt.xlabel('residuals', fontsize=12)  # x-axis label
plt.ylabel('Frequency', fontsize=12)  # y-axis label
plt.title('train data', fontsize=14)  # Title
    
    
    
    
score_log_train = np.sqrt(mean_squared_error(y_train_pred,y_train))  # MSE 
score_train = np.sqrt(mean_squared_error(np.exp(y_train_pred),np.exp(y_train)))
    
print('train_data')
# print('MSE_log: ', score_log_train)
print('MSE: ', score_train)
print( score_log_train)
# Create a single plot
plt.figure(figsize=(8, 8))

 # First dataset
plt.scatter(np.exp(y_test), np.exp(y_test_pred), color='blue', alpha=0.7, label='Model 1: Predicted vs Actual')
plt.plot([5, 700000], [5, 700000], color='red', linestyle='--', linewidth=1, label='45° Line')

plt.scatter(np.exp(y_train), np.exp(y_train_pred), color='green', alpha=0.7, label='Model 1: Predicted vs Actual')
plt.plot([5, 700000], [5, 700000], color='red', linestyle='--', linewidth=1, label='45° Line')

    # Plot enhancements
plt.xlabel('Actual Values', fontsize=12)
plt.ylabel('Predicted Values', fontsize=12)
plt.legend()
plt.grid(alpha=0.3)

plt.show()

    
    
# submission
X = df_train.drop(columns = ['SalePrice','LogSalePrice'])
X_tra = full_pipeline.fit_transform(X)
y = df_train['LogSalePrice']

path_test = os.path.join("House_Price_Prediction_Advanced_Regression_Techniques_Kaggle","data_set","test.csv")
df_test = pd.read_csv(path_test)
df_test = add_new_features_no_clean(df_test)

test_tra = full_pipeline.transform(df_test)

voting_reg.fit(X_tra,y) 
y_pred = voting_reg.predict(test_tra)


df = pd.read_csv(os.path.join("House_Price_Prediction_Advanced_Regression_Techniques_Kaggle","data_set","test.csv"))
df['SalePrice'] = np.exp(y_pred)

path_save = os.path.join('House_Price_Prediction_Advanced_Regression_Techniques_Kaggle','data_set','final_submission.csv')
df[['Id','SalePrice']].to_csv(path_save,index=False)


