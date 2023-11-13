#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings('ignore')

import sys
import time
import pandas as pd
import numpy as np
#import scipy
#from scipy.stats.mstats import winsorize
import datetime
import dateutil
from tqdm import tqdm

# import statsmodels
# import xgboost
# import prophet
import patsy

from datetime import datetime, timedelta
import statsmodels.api as sm
import time




def myeval(num_folds=10):
    file_path = 'Proj2_Data/test_with_label.csv'
    test_with_label = pd.read_csv(file_path)
    #num_folds = 10
    wae = []

    for i in range(num_folds):
        file_path = f'Proj2_Data/fold_{i+1}/test.csv'
        test = pd.read_csv(file_path)
        test = test.drop(columns=['IsHoliday']).merge(test_with_label, on=['Date', 'Store', 'Dept'])
        #print(test)
        #print(f"test.shape={test.shape}")

        file_path = f'Proj2_Data/fold_{i+1}/mypred.csv'
        test_pred = pd.read_csv(file_path)
        test_pred = test_pred.drop(columns=['IsHoliday'])
        #print(test_pred)
        #print(f"test_pred.shape={test_pred.shape}")

        # Left join with the test data
        new_test = test_pred.merge(test, on=['Date', 'Store', 'Dept'], how='left')
        #print(new_test)

        # Compute the Weighted Absolute Error
        actuals = new_test['Weekly_Sales']
        preds = new_test['Weekly_Pred']
        #print(preds)
        #print(actuals)
        weights = new_test['IsHoliday'].apply(lambda x: 5 if x else 1)
        wae.append(sum(weights * abs(actuals - preds)) / sum(weights))

    i = 1
    for value in wae:
        print(f"wae_by_fold_{i}={value:.3f}")
        i += 1
    print(f"overall wae={sum(wae) / len(wae):.3f}")
    
    # print(wae)
    # print(np.mean(wae))    
        
    return wae


def preprocess(data):
    tmp = pd.to_datetime(data['Date'])
    data['Wk'] = tmp.dt.isocalendar().week
    data['Yr'] = tmp.dt.year
    data['Wk'] = pd.Categorical(data['Wk'], categories=[i for i in range(1, 53)])  # 52 weeks 
#    data['IsHoliday'] = data['IsHoliday'].apply(int)
    return data


def pca_smooth_train(train):
    smooth_dept_trains = []
    departments = train['Dept'].unique()

    for department in departments:
        # Filter rows where Dept is equal to 1
        filtered_train = train[train['Dept'] == department]
        # Select only the columns 'Store', 'Date', and 'Weekly_Sales'
        selected_columns = filtered_train[['Store', 'Date', 'Weekly_Sales']]
        # Pivot table to spread 'Store' values into columns, with 'Weekly_Sales' as values
        train_dept_ts = selected_columns.pivot(index='Date', columns='Store', values='Weekly_Sales').reset_index()

        X_train = train_dept_ts.iloc[:, 1:]

        # Smooth department data
        X_train = X_train.to_numpy()
        X_train = np.nan_to_num(X_train)
        store_means = np.mean(X_train, axis=0)
        X_train = X_train - store_means
        X_train = np.transpose(X_train)

        U, D, V_t = np.linalg.svd(X_train, full_matrices=False)
        D[8:] = 0
        F_train = U @ np.diag(D) @ V_t

        stores_list = train_dept_ts.columns[1:]
        F_train = pd.DataFrame(np.transpose(F_train), columns=stores_list)
        F_train = F_train.add(store_means, axis=1)
        F_train["Date"] = train_dept_ts["Date"]

        smooth_dept_train = pd.melt(F_train, id_vars=['Date'], value_vars = stores_list, \
                                    var_name='Store', value_name='Weekly_Sales')
        smooth_dept_train["Dept"] = department
        smooth_dept_train["Store"] = smooth_dept_train["Store"].astype(np.int64)
        smooth_dept_trains.append(smooth_dept_train)

    smooth_train = pd.concat(smooth_dept_trains, ignore_index=True)
    return smooth_train


def process_model(train_file="train.csv", test_file="test.csv", pred_file="mypred.csv", base_folder=".", make_post_prediction_adjustment=True):

    # Reading train data
    train_file_path = f"{base_folder}/{train_file}"
    test_file_path = f"{base_folder}/{test_file}"
    pred_file_path = f"{base_folder}/{pred_file}"
    
    
    start_time = time.time()
    train = pd.read_csv(train_file_path)

    # Smooth train data
    smoothed = pca_smooth_train(train)

    train_dupe = train[["Date", "IsHoliday"]].drop_duplicates()
    train = smoothed.merge(train_dupe, on=["Date"], how="left")

    # Reading test data
    test = pd.read_csv(test_file_path)

    # pre-allocate a pd to store the predictions
    test_pred = pd.DataFrame()

    train_pairs = train[["Store", "Dept"]].drop_duplicates(ignore_index=True)
    test_pairs = test[["Store", "Dept"]].drop_duplicates(ignore_index=True)
    unique_pairs = pd.merge(
        train_pairs, test_pairs, how="inner", on=["Store", "Dept"]
    )

    train_split = unique_pairs.merge(train, on=["Store", "Dept"], how="left")
    train_split = preprocess(train_split)
    y, X = patsy.dmatrices(
        "Weekly_Sales ~ Weekly_Sales + Store + Dept + Yr + np.power(Yr, 2) + Wk",
        data=train_split,
        return_type="dataframe",
    )
    train_split = dict(tuple(X.groupby(["Store", "Dept"])))

    test_split = unique_pairs.merge(test, on=["Store", "Dept"], how="left")
    test_split = preprocess(test_split)
    y, X = patsy.dmatrices(
        "Yr ~ Store + Dept + Yr + np.power(Yr, 2) + Wk", data=test_split, return_type="dataframe"
    )
    X["Date"] = test_split["Date"]
    test_split = dict(tuple(X.groupby(["Store", "Dept"])))

    keys = list(train_split)

    for key in keys:
        X_train = train_split[key]
        X_test = test_split[key]

        Y = X_train["Weekly_Sales"]
        X_train = X_train.drop(["Weekly_Sales", "Store", "Dept"], axis=1)

        model = sm.OLS(Y, X_train).fit()
        mycoef = model.params.fillna(0)

        tmp_pred = X_test[["Store", "Dept", "Date"]]
        X_test = X_test.drop(["Store", "Dept", "Date"], axis=1)

        tmp_pred["Weekly_Pred"] = np.dot(X_test, mycoef)
        test_pred = pd.concat([test_pred, tmp_pred], ignore_index=True)
        
    test_pred["Weekly_Pred"].fillna(0, inplace=True)

    # Post-prediction adjustment for fold 5
    if make_post_prediction_adjustment:
        dates = pd.to_datetime(test_pred["Date"])
        test_pred["Wk"] = dates.dt.isocalendar().week

        test_pred_51 = test_pred[test_pred["Wk"] == 51]
        test_pred_51["Shift"] = test_pred_51["Weekly_Pred"] / 9
        test_pred_52 = test_pred[test_pred["Wk"] == 52]

        test_pred_52 = test_pred_52.merge(
            test_pred_51[["Store", "Dept", "Shift"]],
            on=["Store", "Dept"], how="left"
        )

        test_pred_51 = test_pred_51.merge(
            test_pred_52[["Store", "Dept"]],
            on=["Store", "Dept"], how="left", indicator=True
        )
        test_pred_51[test_pred_51["_merge"] == "left_only"]["Shift"] = 0

        test_pred_52["Date"] = "2011-12-30"
        test_pred_52["Shift"].fillna(0, inplace=True)
        test_pred_52["Weekly_Pred"].fillna(0, inplace=True)

        # test_pred_51["Weekly_Pred"] = test_pred_51["Weekly_Pred"] - test_pred_51["Shift"]
        test_pred_52["Weekly_Pred"] = test_pred_52["Weekly_Pred"] + test_pred_52["Shift"]

        test_pred_51.drop("Shift", inplace=True, axis=1)
        test_pred_52.drop("Shift", inplace=True, axis=1)

        test_pred = test_pred[(test_pred["Wk"] != 51) & (test_pred["Wk"] != 52)]
        test_pred = pd.concat([test_pred, test_pred_51, test_pred_52], ignore_index=True)
        test_pred.drop(columns=["Wk"], inplace=True)

    # Save the output to CSV
    test_pred["Store"] = test_pred["Store"].astype(np.int64)
    test_pred["Dept"] = test_pred["Dept"].astype(np.int64)
    
    test_pred_final = test.merge(test_pred, on=["Store", "Dept", "Date"], how="left")
    test_pred_final["Weekly_Pred"] = test_pred_final["Weekly_Pred"].fillna(0)
    test_pred_final = test_pred_final.loc[:, ["Store", "Dept", "Date", "IsHoliday", "Weekly_Pred"]]
    test_pred_final.to_csv(pred_file_path, index=False)
    print(pred_file_path)
    
    return


"""
num_folds = 10

for j in tqdm(range(1, num_folds + 1)):

    base_folder = f"Proj2_Data/fold_{j}"
    # Reading train data
    train_file = "train.csv"
    test_file = "test.csv"
    
    process_model(train_file=train_file, test_file=test_file, base_folder=base_folder, make_post_prediction_adjustment=True)


wae = myeval(num_folds)
"""

process_model()


