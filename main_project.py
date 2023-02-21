import math
import sys

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

from do_training import train_ridge, train_mae, train_mse
from stat_func import r_squared, loss_mae, loss_mse, loss_ridge, do_predict, do_split


def doKFold(func, X, y, stepparams, k):

    num_models = len(stepparams)

    sq_errors_array_tr = [[None for i in range(k)] for j in range(num_models)]
    sq_errors_array_te = [[None for i in range(k)] for j in range(num_models)]
    average_sq_error_tr = [None] * num_models
    average_sq_error_te = [None] * num_models

    kfold = KFold(n_splits=k, shuffle=True, random_state=None)
    folds = [next(kfold.split(X.T)) for i in range(k)]

    for split_i in range(k):
        X_tr= X.T[folds[split_i][0]].T
        X_te = X.T[folds[split_i][1]].T
        y_tr= y.T[folds[split_i][0]].T
        y_te = y.T[folds[split_i][1]].T

        for j in range(len(stepparams)):
            w_trained = func(X_tr, y_tr, stepparams[j])
            predictions_tr = do_predict(w_trained, X_tr)
            sq_errors_array_tr[j][split_i] = mean_squared_error(y_tr, predictions_tr)
            predictions_te = do_predict(w_trained, X_te)
            sq_errors_array_te[j][split_i] = mean_squared_error(y_te, predictions_te)

    bestparam = 0
    avgerror = math.inf
    for j in range(len(stepparams)):
        average_sq_error_tr[j] = sum(sq_errors_array_tr[j])/k
        average_sq_error_te[j] = sum(sq_errors_array_te[j])/k
        print('Squared error of each fold - {}'.format(sq_errors_array_te[j]))
        print('Avg squared error : {}'.format(average_sq_error_te[j]))
        if (average_sq_error_te[j] < avgerror):
            bestparam = j
            avgerror = average_sq_error_te[j]

    return stepparams[bestparam]


def train_all(xTr, xTv, yTr, yTv, plots, description, preprocessed, params, trial):
    bias = np.ones(shape=(1, xTr.shape[1]))
    xTr_bias = np.append(bias, xTr, axis=0)
    bias = np.ones(shape=(1, xTv.shape[1]))
    xTv_bias = np.append(bias, xTv, axis=0)

    mse_bestparam = doKFold(train_mse, xTr_bias, yTr, params, 5)
    print("MSE Best Eta ", mse_bestparam)

    mse_w_trained = train_mse(xTr_bias, yTr, mse_bestparam)
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
        if (preprocessed):
            plt.savefig('trial-' + str(trial) + '/preprocessing-run/mse/' + description + '.png', bbox_inches='tight')
        else:
            plt.savefig('trial-' + str(trial) + '/non-preprocessing-run/mse/' + description + '.png', bbox_inches='tight')
        plt.show()

    train_ridge_f = lambda x,y,lambdaa: train_ridge(x,y, mse_bestparam, lambdaa)
    ridge_bestparam = doKFold(train_ridge_f, xTr_bias, yTr, params, 5)
    print("Ridge Best Eta ", ridge_bestparam)

    lambdaa = ridge_bestparam
    ridge_w_trained = train_ridge(xTr_bias, yTr, mse_bestparam, lambdaa)
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
        if (preprocessed):
            plt.savefig('trial-' + str(trial) + '/preprocessing-run/ridge/' + description + '.png', bbox_inches='tight')
        else:
            plt.savefig('trial-' + str(trial) + '/non-preprocessing-run/ridge/' + description + '.png', bbox_inches='tight')
        plt.show()

    mae_bestparam = doKFold(train_mae, xTr_bias, yTr, params, 5)
    print("MAE Best Eta ", mae_bestparam)

    mae_w_trained = train_mae(xTr_bias, yTr,mae_bestparam)
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
        if (preprocessed):
            plt.savefig('trial-' + str(trial) + '/preprocessing-run/mae/' + description + '.png', bbox_inches='tight')
        else:
            plt.savefig('trial-' + str(trial) + '/non-preprocessing-run/mae/' + description + '.png', bbox_inches='tight')
        plt.show()


def do_project(preprocess, params, trial):
    df = pd.read_csv("Concrete_Data.csv")
    d = df.shape[1] - 1

    if (preprocess):
        processor = preprocessing.MinMaxScaler()
        # processor = preprocessing.StandardScaler()
        column_names = df.columns
        df_fit = processor.fit_transform(df)
        df_processed = pd.DataFrame(df_fit, columns=column_names)
        train_df, test_df = do_split(df_processed, 130)
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
            train_all(xTr,xTv, yTr, yTv, False, "All", preprocess, params, trial)
        else:
            xTr = X_train[i:i+1, :]
            xTv = X_test[i:i+1, :]
            yTr = Y_train[:, :]
            yTv = Y_test[:, :]
            train_all(xTr,xTv, yTr, yTv, True, df.columns[i], preprocess, params, trial)


trial = 3
original_stdout = sys.stdout

with open('trial-' + str(trial) + '/non-preprocessing-run/output.txt', 'w') as np_out:
    sys.stdout = np_out

    print("Beginning project without preprocessing...")
    # [1e-07, 1e-08, 1e-09, 1e-10]
    do_project(False, [1e-07, 1e-08, 1e-09, 1e-10], trial)
    sys.stdout = original_stdout
    np_out.close()

with open('trial-' + str(trial) + '/preprocessing-run/output.txt', 'w') as p_out:
    sys.stdout = p_out
    print("Beginning project with preprocessing...")
    # [1e-02, 1e-03, 1e-04, 1e-05]
    do_project(True, [1e-02, 1e-03, 1e-04, 1e-05], trial)

    sys.stdout = original_stdout
    p_out.close()