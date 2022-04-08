# Dataframe manipulations
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt

# Sklearn suite
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression, LassoLars, TweedieRegressor
from sklearn.preprocessing import PolynomialFeatures


def x_y_splits(train, validate, test):
    '''
    This function takes in our train, validate, and test dataframes and returns their respective x and y components. The target variable (y) is log error.
    '''
    # Creation of x and y components for train
    x_train = train.drop(columns='logerror')
    y_train = train.logerror
    # Creation of x and y components for validate
    x_validate = validate.drop(columns='logerror')
    y_validate = validate.logerror
    # Creation of x and y components for test
    x_test = test.drop(columns='logerror')
    y_test = test.logerror
    # Return the x and y components
    return x_train, y_train, x_validate, y_validate, x_test, y_test


def baseline_selection(y_train, y_validate):
    '''
    This function takes our train and validate y components and computes a mean and median baseline for modeling purposes. It compares their
    RMSE values and returns whichever is a better usecase. 
    '''
    # Creation of mean value and adding to our dataframes
    pred_mean = y_train.logerror.mean()
    y_train['pred_mean'] = pred_mean
    y_validate['pred_mean'] = pred_mean
    # Creation of median value and adding to our dataframes
    pred_median = y_train['logerror'].median()
    y_train['pred_median'] = pred_median
    y_validate['pred_median'] = pred_median 
    # Evaluating RMSE value for mean baseline
    rmse_train_m = mean_squared_error(y_train.logerror, y_train.pred_mean)**(1/2)
    rmse_validate_m = mean_squared_error(y_validate.logerror, y_validate.pred_mean)**(1/2)
    # Taking the average of our train and validate analysis
    mean_RMSE = round((rmse_train_m + rmse_validate_m) / 2, 6)
    # Evaluating RMSE value for median baseline
    rmse_train = mean_squared_error(y_train.logerror, y_train.pred_median)**(1/2)
    rmse_validate = mean_squared_error(y_validate.logerror, y_validate.pred_median)**(1/2)
    # Taking the average of our train and validate analysis
    median_RMSE = round((rmse_train + rmse_validate) / 2, 6)

    # Determine which is better to use as baseline
    if mean_RMSE < median_RMSE:
        print(f'Mean log error provides a better baseline. Returning mean RMSE of {mean_RMSE}.')
        return mean_RMSE
    else: 
        print(f'Median log error provides a better baseline. Returning median RMSE of {median_RMSE}.')
        return median_RMSE


def ols_model(x_train, y_train, x_validate, y_validate):
    '''
    This function takes in our x and y components for train and validate and prints the results of RMSE for train and validate modeling.
    '''
    # Creation and fitting of the model
    lm = LinearRegression(normalize=True)
    lm.fit(x_train, y_train.logerror)
    # Prediction creation and RMSE value for train
    y_train['log_lm_pred'] = lm.predict(x_train)
    rmse_train = round(mean_squared_error(y_train.logerror, y_train.log_lm_pred)**(1/2), 6)
    # Prediction creation and RMSE for validate
    y_validate['log_lm_pred'] = lm.predict(x_validate)
    rmse_validate = round(mean_squared_error(y_validate.logerror, y_validate.log_lm_pred)**(1/2), 6)
    # Outputing the RMSE values
    print("RMSE for OLS using LinearRegression\nTraining: ", rmse_train,
            "\nValidation: ", rmse_validate)


def LassoLars_model(x_train, y_train, x_validate, y_validate):
    '''
    This function takes in our x and y components for train and validate and prints the results of RMSE for train and validate modeling.
    '''
    # Creation and fitting of the model
    lars = LassoLars(alpha=1.0)
    lars.fit(x_train, y_train.logerror)
    # Prediction creation and RMSE value for train
    y_train['log_pred_lars'] = lars.predict(x_train)
    rmse_train = round(mean_squared_error(y_train.logerror, y_train.log_pred_lars)**(1/2), 6)
    # Prediction creation and RMSE value for validate
    y_validate['log_pred_lars'] = lars.predict(x_validate)
    rmse_validate = round(mean_squared_error(y_validate.logerror, y_validate.log_pred_lars)**(1/2), 6)
    # Outputing the RMSE values
    print("RMSE for Lasso + Lars\nTraining: ", rmse_train,
     "\nValidation: ", rmse_validate)


def Poly_reg_model(x_train, y_train, x_validate, y_validate, d):
    '''
    This function takes in our x and y components for train and validate and prints the results of RMSE for train and validate modeling.
    '''
    # Creation of polynomial model
    pf = PolynomialFeatures(degree=d, interaction_only=True)
    # Fitting on x_train and transforming on all dataframes
    x_train_degree = pf.fit_transform(x_train)
    x_validate_degree = pf.transform(x_validate)
    #x_test_degree = pf.transform(x_test)
    # Creating the LinearRegression model to use the polynomial data
    lm2 = LinearRegression(normalize=True)
    # Fitting on the polynomial data
    lm2.fit(x_train_degree, y_train.logerror)
    y_train['log_pred_lm_deg'] = lm2.predict(x_train_degree)
    # Calculating the RMSE value
    rmse_train = round(mean_squared_error(y_train.logerror, y_train.log_pred_lm_deg)**(1/2), 6)
    # Doing the above work with the validate
    y_validate['log_pred_lm_deg'] = lm2.predict(x_validate_degree)
    rmse_validate = round(mean_squared_error(y_validate.logerror, y_validate.log_pred_lm_deg)**(1/2), 6)
    # Print and comparison of RMSE value for train and validate outcomes
    print("RMSE for Polynomial Model, with two degrees\nTraining: ", rmse_train,
     "\nValidation: ", rmse_validate)




def Tweedie_model(x_train, y_train, x_validate, y_validate, x_test, y_test):
    '''
    This function takes in our x and y components for train, validate and test and prints the results of RMSE for all three models. 
    '''
    # Creation and fitting of the model
    glm = TweedieRegressor(power=0, alpha=0)
    glm.fit(x_train, y_train.logerror)
    # Prediction creation and RMSE value for train
    y_train['log_pred_glm'] = glm.predict(x_train)
    rmse_train = round(mean_squared_error(y_train.logerror, y_train.log_pred_glm)**(1/2), 6)
    # Prediction creation and RMSE value for validate
    y_validate['log_pred_glm'] = glm.predict(x_validate)
    rmse_validate = round(mean_squared_error(y_validate.logerror, y_validate.log_pred_glm)**(1/2), 6)
    # Prediction creation and RMSE value for test
    y_test['log_pred_glm'] = glm.predict(x_test)
    rmse_test = round(mean_squared_error(y_test.logerror, y_test.log_pred_glm)**(1/2), 6)
    # Outputing the RMSE values
    print("RMSE using Tweedie, power=0 & alpha=0\nTraining: ", rmse_train,
     "\nValidation: ", rmse_validate,
     "\nTest: ", rmse_test)


def residual_visual(y_test):
    '''
    This function takes the y component of our test dataframe and outputs the Residual results from the predicted and actual values. 
    '''
    plt.figure(figsize=(12,5))
    plt.axhline(y=0, ls = ':')
    plt.scatter(y_test.logerror, y_test.log_pred_glm - y_test.logerror,
           alpha=.5, color='red')
    plt.xlabel('Actual Value')
    plt.ylabel('Residual/Error (Predicted log error - Actual log error)')
    plt.title('Residual Visual')
    plt.show()