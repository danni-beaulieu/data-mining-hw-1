import math

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from do_training import train_ridge, train_mae, train_mse
from stat_func import r_squared, loss_mae, loss_mse, loss_ridge, do_predict

df = pd.read_csv("Concrete_Data.csv")
raw_data = df.to_numpy()

X = raw_data[:,0:(raw_data.shape[1] - 1)].T
Y = raw_data[:, (raw_data.shape[1] - 1)].reshape(1,-1)
d, n = X.shape

# X_df = df.iloc[:,:-1]
# y_df = df.iloc[:,-1]

[d, n] = np.shape(X)
part = math.ceil(n * 0.8737)
part = int(part)

# X = X[0:1,:]

for i in range (d):
    xTr = X[i:i+1, 0:part]
    xTv = X[i:i+1, part:n]
    yTr = Y[:, 0:part]
    yTv = Y[:, part:n]

    bias = np.ones(shape=(1,xTr.shape[1]))
    xTr_bias = np.append(bias, xTr, axis=0)
    bias = np.ones(shape=(1,xTv.shape[1]))
    xTv_bias = np.append(bias, xTv, axis=0)

    mse_w_trained = train_mse(xTr_bias,yTr)
    print("MSE Loss (Train) ", loss_mse(mse_w_trained, xTr_bias, yTr))
    print("MSE R-Squared (Train) ", r_squared(mse_w_trained, xTr_bias, yTr))
    print("MSE Loss (Test) ", loss_mse(mse_w_trained, xTv_bias, yTv))
    print("MSE R-Squared (Test) ", r_squared(mse_w_trained, xTv_bias, yTv))
    mse_preds = do_predict(mse_w_trained, xTr_bias)
    plt.scatter(xTr[0], yTr[0], color='green', alpha=.5)
    plt.plot(xTr[0], mse_preds[0], color='purple', lw=5)
    plt.title("MSE Regression: Data vs Model")
    plt.xlabel("Feature " + str(i))
    plt.ylabel("Compressive Strength")
    plt.show()

    lambdaa = 0.001
    ridge_w_trained = train_ridge(xTr_bias,yTr, lambdaa)
    print("Ridge Loss (Train) ", loss_ridge(ridge_w_trained, xTr_bias, yTr, lambdaa))
    print("Ridge R-Squared (Train) ", r_squared(ridge_w_trained, xTr_bias, yTr))
    print("Ridge Loss (Test) ", loss_ridge(ridge_w_trained, xTv_bias, yTv, lambdaa))
    print("Ridge R-Squared (Test) ", r_squared(ridge_w_trained, xTv_bias, yTv))
    ridge_preds = do_predict(ridge_w_trained, xTr_bias)
    plt.scatter(xTr[0], yTr[0], color='green', alpha=.5)
    plt.plot(xTr[0], ridge_preds[0], color='purple', lw=5)
    plt.title("Ridge Regression: Data vs Model")
    plt.xlabel("Feature " + str(i))
    plt.ylabel("Compressive Strength")
    plt.show()

    mae_w_trained = train_mae(xTr_bias,yTr)
    print("MAE Loss (Train) ", loss_mae(mae_w_trained, xTr_bias, yTr))
    print("MAE R-Squared (Train) ", r_squared(mae_w_trained, xTr_bias, yTr))
    print("MAE Loss (Test) ", loss_mae(mae_w_trained, xTv_bias, yTv))
    print("MAE R-Squared (Test) ", r_squared(mae_w_trained, xTv_bias, yTv))
    mae_preds = do_predict(mae_w_trained, xTr_bias)
    plt.scatter(xTr[0], yTr[0], color='green', alpha=.5)
    plt.plot(xTr[0], mae_preds[0], color='purple', lw=5)
    plt.title("MAE Regression: Data vs Model")
    plt.xlabel("Feature " + str(i))
    plt.ylabel("Compressive Strength")
    plt.show()


