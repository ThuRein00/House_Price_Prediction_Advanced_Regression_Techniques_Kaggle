
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
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
# param_grid_XGBoost = {
#     'n_estimators': [200,500,800],        # Number of trees
#     'learning_rate': [0.01,0.05,0.1],    # Step size shrinkage
#     'max_depth': [1,3,5],                # Maximum depth of a tree
#     'subsample': [0.8,0.9,1],          # Subsample ratio of the training instance
#     'colsample_bytree': [0.5,0.7],   # Subsample ratio of columns when constructing each tree          
#     'reg_alpha': [0,0.01,0.1,5],            # L1 regularization term
#     'reg_lambda': [0.001,0.1,10],         # L2 regularization term
# }
# param_grid_XGBoost = {
#     'n_estimators': [800],        # Number of trees
#     'learning_rate': [0.05],    # Step size shrinkage
#     'max_depth': [1,2,3],                # Maximum depth of a tree
#     'subsample': [0.9],          # Subsample ratio of the training instance
#     'colsample_bytree': [0.5],   # Subsample ratio of columns when constructing each tree          
#     'reg_alpha': [0,0.01,0.1,5],            # L1 regularization term
#     'reg_lambda': [0.001,0.1,10,20],         # L2 regularization term
# }
param_grid_XGBoost = {
    'n_estimators': [700],        # Number of trees
    'learning_rate': [0.1],    # Step size shrinkage
    'max_depth': [1,2],                # Maximum depth of a tree
    'subsample': [0.9],          # Subsample ratio of the training instance
    'colsample_bytree': [0.2],   # Subsample ratio of columns when constructing each tree
    'gamma': [0],                # Minimum loss reduction for further partition
    'reg_alpha': [0.2],            # L1 regularization term
    'reg_lambda': [1.3],         # L2 regularization term
}
#transform data for machine learning
X_train_tra = full_pipeline.fit_transform(X_train)
X_test_tra = full_pipeline.transform(X_test)
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

#grid search and plot results
best_model_XGBoost = grid_search_and_plot_result(param_grid_XGBoost,XGBRegressor(),X_train_tra,y_train,X_test_tra,y_test)

         
         
#save the best model
save_path = os.path.join('House_Price_Prediction_Advanced_Regression_Techniques_Kaggle','model_pickle','XGBoost.pkl')
joblib.dump(best_model_XGBoost,save_path)



