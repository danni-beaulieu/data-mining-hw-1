import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing

from do_training import train_ridge, train_mae, train_mse
from stat_func import r_squared, loss_mae, loss_mse, loss_ridge, do_predict, do_split


def train_all(xTr, xTv, yTr, yTv, plots, description):
    bias = np.ones(shape=(1, xTr.shape[1]))
    xTr_bias = np.append(bias, xTr, axis=0)
    bias = np.ones(shape=(1, xTv.shape[1]))
    xTv_bias = np.append(bias, xTv, axis=0)

    mse_w_trained = train_mse(xTr_bias, yTr)
    print("MSE Loss (Train) ", loss_mse(mse_w_trained, xTr_bias, yTr))
    print("MSE R-Squared (Train) ", r_squared(mse_w_trained, xTr_bias, yTr))
    print("MSE Loss (Test) ", loss_mse(mse_w_trained, xTv_bias, yTv))
    print("MSE R-Squared (Test) ", r_squared(mse_w_trained, xTv_bias, yTv))
    mse_preds = do_predict(mse_w_trained, xTr_bias)
    if (plots):
        plt.scatter(xTr[0], yTr[0], color='green', alpha=.5)
        plt.plot(xTr[0], mse_preds[0], color='purple', lw=5)
        plt.title("MSE Regression: Data vs Model")
        plt.xlabel("Feature " + description)
        plt.ylabel("Compressive Strength")
        plt.show()

    lambdaa = 0.001
    ridge_w_trained = train_ridge(xTr_bias, yTr, lambdaa)
    print("Ridge Loss (Train) ", loss_ridge(ridge_w_trained, xTr_bias, yTr, lambdaa))
    print("Ridge R-Squared (Train) ", r_squared(ridge_w_trained, xTr_bias, yTr))
    print("Ridge Loss (Test) ", loss_ridge(ridge_w_trained, xTv_bias, yTv, lambdaa))
    print("Ridge R-Squared (Test) ", r_squared(ridge_w_trained, xTv_bias, yTv))
    ridge_preds = do_predict(ridge_w_trained, xTr_bias)
    if(plots):
        plt.scatter(xTr[0], yTr[0], color='green', alpha=.5)
        plt.plot(xTr[0], ridge_preds[0], color='purple', lw=5)
        plt.title("Ridge Regression: Data vs Model")
        plt.xlabel("Feature " + description)
        plt.ylabel("Compressive Strength")
        plt.show()

    mae_w_trained = train_mae(xTr_bias, yTr)
    print("MAE Loss (Train) ", loss_mae(mae_w_trained, xTr_bias, yTr))
    print("MAE R-Squared (Train) ", r_squared(mae_w_trained, xTr_bias, yTr))
    print("MAE Loss (Test) ", loss_mae(mae_w_trained, xTv_bias, yTv))
    print("MAE R-Squared (Test) ", r_squared(mae_w_trained, xTv_bias, yTv))
    mae_preds = do_predict(mae_w_trained, xTr_bias)
    if (plots):
        plt.scatter(xTr[0], yTr[0], color='green', alpha=.5)
        plt.plot(xTr[0], mae_preds[0], color='purple', lw=5)
        plt.title("MAE Regression: Data vs Model")
        plt.xlabel("Feature " + description)
        plt.ylabel("Compressive Strength")
        plt.show()


def do_project(preprocess):
    df = pd.read_csv("Concrete_Data.csv")
    d = df.shape[1] - 1

    if (preprocess):
        nrmlz = preprocessing.MinMaxScaler()
        column_names = df.columns
        df_fit = nrmlz.fit_transform(df)
        df_nrmlz = pd.DataFrame(df_fit, columns=column_names)
        train_df, test_df = do_split(df_nrmlz, 130)
    else:
        train_df, test_df = do_split(df, 130)

    raw_data_train = train_df.to_numpy()
    raw_data_test = test_df.to_numpy()

    X_train = raw_data_train[:,0:(raw_data_train.shape[1] - 1)].T
    Y_train = raw_data_train[:, (raw_data_train.shape[1] - 1)].reshape(1,-1)
    X_test = raw_data_test[:,0:(raw_data_test.shape[1] - 1)].T
    Y_test = raw_data_test[:, (raw_data_test.shape[1] - 1)].reshape(1,-1)

    for i in range (-1,d):
        print("Working with feature: ", i)
        if (i == -1):
            xTr = X_train
            xTv = X_test
            yTr = Y_train
            yTv = Y_test
            train_all(xTr,xTv, yTr, yTv, False, "All")
        else:
            xTr = X_train[i:i+1, :]
            xTv = X_test[i:i+1, :]
            yTr = Y_train[:, :]
            yTv = Y_test[:, :]
            train_all(xTr,xTv, yTr, yTv, True, df.columns[i])



print("Beginning project without preprocessing...")
do_project(False)
print("Beginning project with preprocessing...")
do_project(True)