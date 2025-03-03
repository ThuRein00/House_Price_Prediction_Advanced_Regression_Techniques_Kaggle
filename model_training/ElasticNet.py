
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
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
param_grid_elastic_net = {
    'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10],
    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
    'max_iter': [5000],
    'tol': [1e-5, 1e-4, 1e-3]
}

#transform data for machine learning
X_train_tra = full_pipeline.fit_transform(X_train)
X_test_tra = full_pipeline.transform(X_test)
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()

#grid search and plot results
best_model_elastic_net = grid_search_and_plot_result(param_grid_elastic_net,ElasticNet(),X_train_tra,y_train,X_test_tra,y_test)
         
#save the best model
save_path = os.path.join('House_Price_Prediction_Advanced_Regression_Techniques_Kaggle','model_pickle','ElasticNet.pkl')
joblib.dump(best_model_elastic_net,save_path)