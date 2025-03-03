
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
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


X_train = df_train.drop(columns = ['SalePrice','LogSalePrice'])
y_train = df_train['LogSalePrice']

#grid search parameters
param_grid_lasso = {
    'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    'max_iter': [5000],
    'tol': [1e-4, 1e-3, 1e-2]
}

#transform data for machine learning
X_train_tra = full_pipeline.fit_transform(X_train)
X_test_tra = full_pipeline.transform(X_test)
y_train = y_train.to_numpy()
y_test = y_test.to_numpy()



#grid search and plot results
best_model_lasso = grid_search_and_plot_result(param_grid_lasso,Lasso(),X_train_tra,y_train,X_train_tra,y_train)
         
#save the best model
save_path = os.path.join('House_Price_Prediction_Advanced_Regression_Techniques_Kaggle','model_pickle','Lasso.pkl')
joblib.dump(best_model_lasso,save_path)