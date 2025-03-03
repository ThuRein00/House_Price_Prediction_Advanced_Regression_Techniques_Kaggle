
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
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
param_grid_Forest ={
    'n_estimators': [200,500],# Number of trees
    'max_depth': [3,5,8],  # Maximum depth of trees
    'min_samples_split': [2,5, 8],  # Minimum samples to split a node
    'min_samples_leaf': [1,5,8],   # Minimum samples at a leaf node
    'max_features': [0.3,0.5,0.7],  # Number of features to consider for splits
    'bootstrap': [True]  # Whether to use bootstrapping
}

#transform data for machine learning
X_train_tra = full_pipeline.fit_transform(X_train)
X_test_tra = full_pipeline.transform(X_test)
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

#grid search and plot results
best_model_Forest = grid_search_and_plot_result(param_grid_Forest,RandomForestRegressor(random_state=42),X_train_tra,y_train,X_test_tra,y_test)

         
#save the best model
save_path = os.path.join('House_Price_Prediction_Advanced_Regression_Techniques_Kaggle','model_pickle','RandomForest.pkl')
joblib.dump(best_model_Forest,save_path)



