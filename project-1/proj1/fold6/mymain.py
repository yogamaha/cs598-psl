import sys
import time
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from scipy.stats.mstats import winsorize
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.linear_model import Ridge, RidgeCV, Lasso, LassoCV
from sklearn.metrics import mean_squared_error
from math import sqrt
from tqdm import tqdm
from sklearn.ensemble import GradientBoostingRegressor
import random

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def linear_trainset_preprocessing(df):


    x_train = df.drop(columns=['PID','Longitude','Latitude','Sale_Price'], axis = 1)
    y_train = np.log(df["Sale_Price"])


    #Only "Garage_Yr_Blt" has missing values upon examination, replace the missing values with zero.
    x_train.fillna(0, inplace=True)


    # To deal with inbalance issue, remove the variables that has one value dominates with a high percentage greater than 98%.
    variables_to_remove = []
    for column in x_train:
      dominant_percentage = x_train[column].value_counts(normalize=True).max()
      if dominant_percentage > 0.98:
        variables_to_remove.append(column)

    x_train = x_train.drop(columns=variables_to_remove)

    # Calculate medians for numerical columns and use those medians to fill missing values in the test set.
    numerical_columns = x_train.select_dtypes(include=np.number)
    medians_dict = numerical_columns.median().to_dict()


    # For outliers in numerical variables, use winsorization to replace the highest 5% of the data by the value of the data at the 95th percentile
    #store the max values for winsorized columns and use it to clip the test data later.

    numerical_columns = x_train.select_dtypes(include=['number']).columns.tolist()
    winsorized_max = {}

    for column in numerical_columns:
        x_train[column] = winsorize(x_train[column], limits=(0, 0.05))
        winsorized_max[column] = x_train[column].max()

    # Encode catogorical variables to create binary dummy variables
    columns_to_encode = x_train.select_dtypes(include=['object']).columns.tolist()

    for column in columns_to_encode:
        x_train[column] = x_train[column].astype(str)
    encoder = OneHotEncoder(sparse=False,handle_unknown = "ignore")
    encoder.fit(x_train[columns_to_encode])

    x_train_encoded_data = encoder.transform(x_train[columns_to_encode])
    x_train_encoded_df = pd.DataFrame(x_train_encoded_data, columns=encoder.get_feature_names_out(input_features=columns_to_encode))
    x_train_all = pd.concat([x_train_encoded_df, x_train.drop(columns=columns_to_encode)], axis=1)

    # Create a StandardScaler instance and fit it on the training data,
    # Transform the training data using the fitted scaler and transform the test data using the same scaler

    scaler = StandardScaler()
    scaler.fit(x_train_all)
    X_train_final = scaler.transform(x_train_all)

    return X_train_final, y_train, variables_to_remove, medians_dict, winsorized_max,encoder, columns_to_encode, scaler

def linear_testset_preprocessing(df,variables_to_remove,medians_dict, winsorized_max, encoder, columns_to_encode, scaler):


    x_test = df.drop(columns=['PID','Longitude','Latitude'], axis = 1)
    x_test_PID = df[["PID"]]

    x_test = x_test.drop(columns=variables_to_remove)

    # Fill missing values in the test set using the stored medians
    for col, median in medians_dict.items():
        x_test[col].fillna(median, inplace=True)

    # Apply Winsorization to specified columns using bounds from train data
    for column, max in winsorized_max.items():
      x_test[column] = np.clip(x_test[column], 0, max)


    # Transform the test data using the same encoder of train data
    for column in columns_to_encode:
        x_test[column] = x_test[column].astype(str)

    x_test_encoded = encoder.transform(x_test[columns_to_encode])
    x_test_encoded_df = pd.DataFrame(x_test_encoded, columns=encoder.get_feature_names_out(input_features=columns_to_encode))
    x_test_encoded_all = pd.concat([x_test_encoded_df, x_test.drop(columns=columns_to_encode)], axis=1)


    X_test_final = scaler.transform(x_test_encoded_all)

    return X_test_final,x_test_PID

def tree_trainset_preprocessing(df):

      x_train = df.drop(columns=['PID','Sale_Price'], axis = 1)
      y_train = np.log(df["Sale_Price"])


      #Only "Garage_Yr_Blt" has missing values upon examination, replace the missing values with zero.
      x_train.fillna(0, inplace=True)


      # Calculate medians for numerical columns and use those medians to fill missing values in the test set.
      numerical_columns = x_train.select_dtypes(include=np.number)
      medians_dict = numerical_columns.median().to_dict()


      # Encode catogorical variables to create binary dummy variables
      columns_to_encode = x_train.select_dtypes(include=['object']).columns.tolist()
      for column in columns_to_encode:
          x_train[column] = x_train[column].astype(str)

      encoder = OneHotEncoder(sparse=False,handle_unknown = "ignore")
      encoder.fit(x_train[columns_to_encode])

      x_train_encoded_data = encoder.transform(x_train[columns_to_encode])
      x_train_encoded_df = pd.DataFrame(x_train_encoded_data, columns=encoder.get_feature_names_out(input_features=columns_to_encode))
      x_train_all = pd.concat([x_train_encoded_df, x_train.drop(columns=columns_to_encode)], axis=1)


      return x_train_all, y_train, medians_dict,encoder, columns_to_encode

def tree_testset_preprocessing(df,medians_dict, encoder, columns_to_encode):


    x_test = df.drop(columns=['PID'], axis = 1)
    x_test_PID = df[["PID"]]

    # Fill missing values in the test set using the stored medians
    for col, median in medians_dict.items():
        x_test[col].fillna(median, inplace=True)

    # Transform the test data using the same encoder of train data

    for column in columns_to_encode:
        x_test[column] = x_test[column].astype(str)

    x_test_encoded = encoder.transform(x_test[columns_to_encode])
    x_test_encoded_df = pd.DataFrame(x_test_encoded, columns=encoder.get_feature_names_out(input_features=columns_to_encode))
    X_test_final = pd.concat([x_test_encoded_df, x_test.drop(columns=columns_to_encode)], axis=1)


    return X_test_final,x_test_PID

def train_ridge_lasso(X_train,Y_train):

    # use lasso to select variables
    alphas = np.logspace(-10, 10, 100)
    lassocv = LassoCV(alphas = alphas, cv = 10)
    lassocv.fit(X_train, Y_train)
    lasso_model_min = Lasso(alpha = lassocv.alpha_, max_iter=5000)
    lasso_model_min.fit(X_train, Y_train)

    nonzero_indices = np.where(lasso_model_min.coef_ != 0)[0]

    # select alpha from ridgecv and refit ridge with variables selected by lasso
    ridgecv = RidgeCV(alphas = alphas, cv = 10)
    ridgecv.fit(X_train[:, nonzero_indices], Y_train)

    #### alpha_min
    ridge_model_min = Ridge(alpha = ridgecv.alpha_)
    ridge_model_min.fit(X_train[:, nonzero_indices], Y_train)

    return ridge_model_min,nonzero_indices

def predict_ridge_lasso(model,nonzero_indices,x_test,PID):

    y_predict = model.predict(x_test[:, nonzero_indices])

    y_predict = np.exp(y_predict)
    y_predict = pd.DataFrame(y_predict, columns=["Sale_Price"])

    output_df = pd.concat([PID, y_predict], axis=1)
    output_df.to_csv("mysubmission1.txt", sep=',', index=False)

def train_tree_model(X_train,Y_train):


    tree_regressor = GradientBoostingRegressor(
    learning_rate=0.02, n_estimators=1000, subsample=0.5,max_depth=6)
    tree_regressor.fit(X_train, Y_train)

    return tree_regressor

def predict_tree_model(model,x_test,PID):

    y_predict = model.predict(x_test)
    y_predict = np.exp(y_predict)
    y_predict = pd.DataFrame(y_predict, columns=["Sale_Price"])

    output_df = pd.concat([PID, y_predict], axis=1)
    output_df.to_csv("mysubmission2.txt", sep=',', index=False)

if __name__ == "__main__":

    # Accept train.csv and test.csv as inputs.
    train_set = pd.read_csv("train.csv")
    test_set = pd.read_csv("test.csv")

    random.seed(4844)

    # Preprocess the training data
    linear_x_train, linear_y_train, linear_variables_to_remove, linear_medians_dict,linear_winsorized_max,linear_encoder, linear_columns_to_encode, linear_scaler = linear_trainset_preprocessing(train_set)
    tree_x_train, tree_y_train, tree_medians_dict,tree_encoder, tree_columns_to_encode = tree_trainset_preprocessing(train_set)

    print ("Fitting models ... ")

    # fit the two models
    ridge_lasso,nonzero_indices = train_ridge_lasso(linear_x_train,linear_y_train)
    tree_regressor = train_tree_model(tree_x_train,tree_y_train)

    print ("Preprocessing test data ... ")

    #Preprocess test data
    linear_X_test, linear_PID = linear_testset_preprocessing(test_set,linear_variables_to_remove,linear_medians_dict, linear_winsorized_max, linear_encoder, linear_columns_to_encode, linear_scaler)
    tree_X_test, tree_PID = tree_testset_preprocessing(test_set,tree_medians_dict, tree_encoder, tree_columns_to_encode)

    print ("Saving predictions ...")

    # save predictions into two files
    predict_ridge_lasso(ridge_lasso,nonzero_indices,linear_X_test, linear_PID)
    predict_tree_model(tree_regressor,tree_X_test,tree_PID)
