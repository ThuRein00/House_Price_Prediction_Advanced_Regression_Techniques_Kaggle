
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error,mean_squared_log_error

def grid_search_and_plot_result(param_grid,model,X_tra,Y,x_tra,y):
    
    param_grid = param_grid
    X_train = X_tra
    y_train = Y
    model = model
    #grid search
    grid_search = GridSearchCV(model, param_grid, cv=5,scoring='neg_mean_squared_error',return_train_score=True,n_jobs = -1)
    grid_search.fit(X_train,y_train)

    X_test = x_tra
    y_test = y
    y_test_pred = grid_search.best_estimator_.predict(X_test)
    res_test= (np.exp(y_test) - np.exp(y_test_pred)) #original data is log transformed
   
    #residuals hist plot
    sns.histplot(res_test, kde =True , stat='probability')
    plt.xlabel('residuals', fontsize=12)  # x-axis label
    plt.ylabel('Frequency', fontsize=12)  # y-axis label
    plt.title('test data', fontsize=14)  # Title
    

    #MSE
    score_log_test = np.sqrt(mean_squared_error(y_test_pred,y_test))  
    score_test = np.sqrt(mean_squared_error(np.exp(y_test_pred),np.exp(y_test)))
    print('test_data')
    print('MSE_log: ', score_log_test)
    print('MSE: ', score_test)
    

    # train_set fit visualization
    y_train_pred = grid_search.best_estimator_.predict(X_train)
    res_train = (np.exp(y_train) - np.exp(y_train_pred))

    #residuals hist plot
    sns.histplot(res_train, kde =True , stat='probability')
    plt.xlabel('residuals', fontsize=12)  # x-axis label
    plt.ylabel('Frequency', fontsize=12)  # y-axis label
    plt.title('train data', fontsize=14)  # Title
    
    #MSE
    score_log_train = np.sqrt(mean_squared_error(y_train_pred,y_train))  # MSE 
    score_train = np.sqrt(mean_squared_error(np.exp(y_train_pred),np.exp(y_train)))
    
    print('train_data')
    print('MSE_log: ', score_log_train)
    print('MSE: ', score_train)
    
    
    #predicted vs actual plt
    plt.figure(figsize=(8, 8))
    plt.scatter(y_test, y_test_pred, color='blue', alpha=0.7, label='Predicted vs Actual')
    plt.plot([9, 15], [9, 15], color='red', linestyle='--', linewidth=1, label='45° Line')

    plt.scatter(y_train, y_train_pred, color='green', alpha=0.7, label='Predicted vs Actual')
    plt.plot([9, 15], [9, 15], color='red', linestyle='--', linewidth=1, label='45° Line')

    # Plot enhancements
    plt.xlabel('Actual Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Show the plot
    plt.show()
    
    

    # to visualize results
    print('best score from grid search: ',np.sqrt(-grid_search.best_score_))
    print('best_estimator: ' , grid_search.best_estimator_)

    cvres = grid_search.cv_results_
    for mean_test_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            if  mean_test_score == grid_search.best_score_ :
                print('best_estimator:',  params)
         

    return grid_search.best_estimator_
    