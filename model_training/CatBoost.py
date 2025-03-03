
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from pre_processing_pipeline import full_pipeline
from grid_search_and_plot_result import grid_search_and_plot_result
import joblib
from feature_add import add_new_features

#load data
path_train = os.path.join("House_Price_Prediction_Advanced_Regression_Techniques_Kaggle","data_set","train.csv")
df_train = pd.read_csv(path_train)
df_train = add_new_features(df_train)


#train_test_split
train_set , test_set = train_test_split(df_train, test_size=0.2, random_state=42)
X_train = train_set.drop(columns = ['SalePrice','LogSalePrice'])
y_train = train_set['LogSalePrice']
X_test = test_set.drop(columns = ['SalePrice','LogSalePrice'])
y_test = test_set['LogSalePrice']

#grid search parameters
# param_grid_CatBoost = {
#     'iterations': [200,500,800],      # Number of boosting iterations
#     'learning_rate': [0.01,0.1,0.5], # Learning rate
#     'depth': [2,4,8],                # Depth of the tree
#     'l2_leaf_reg': [8,10,15],           # L2 regularization
#     'bagging_temperature': [0,2,10],   # Random strength for bagging
#     'border_count': [8,10,16],      # Number of splits for numerical features
# }
param_grid_CatBoost = {
    'iterations': [500],      # Number of boosting iterations
    'learning_rate': [0.1,0.5], # Learning rate
    'depth': [3,4,8],                # Depth of the tree
    'l2_leaf_reg': [10,15],           # L2 regularization
    'bagging_temperature': [0,10],   # Random strength for bagging
    'border_count': [8,10,16],      # Number of splits for numerical features
}

#transform data for machine learning
X_train_tra = full_pipeline.fit_transform(X_train)
X_test_tra = full_pipeline.transform(X_test)
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

#grid search and plot results
best_model_CatBoost = grid_search_and_plot_result(param_grid_CatBoost,CatBoostRegressor(silent=True),X_train_tra,y_train,X_test_tra,y_test)

         
         
#save the best model
save_path = os.path.join('House_Price_Prediction_Advanced_Regression_Techniques_Kaggle','model_pickle','CatBoost.pkl')
joblib.dump(best_model_CatBoost,save_path)



