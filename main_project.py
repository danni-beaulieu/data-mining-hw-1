import math

import numpy as np
import pandas as pd

from do_training import train
from ridge_regresssion import loss


def r_squared(w, xT, yT):
    XT = xT.transpose()
    Y = yT.transpose()
    XTw = XT.dot(w)

    XTwMinusy = XTw - Y
    XTwMinusyT = XTwMinusy.transpose()
    XTwMinusyT_XTwMinusy = XTwMinusyT.dot(XTwMinusy)
    mse = XTwMinusyT_XTwMinusy / yT.shape[1]

    Y_bar = np.full((Y.shape[0], Y.shape[1]), np.mean(Y))
    XTwMinusyBar = XTw - Y_bar
    XTwMinusyBarT = XTwMinusyBar.transpose()
    XTwMinusyBarT_XTwMinusyBar = XTwMinusyBarT.dot(XTwMinusyBar)
    variance = XTwMinusyBarT_XTwMinusyBar / yT.shape[1]

    return (1 - (mse / variance))


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
xTr = X[:, 0:part]
xTv = X[:, part:n]
yTr = Y[:, 0:part]
yTv = Y[:, part:n]

bias = np.ones(shape=(1,xTr.shape[1]))
xTr = np.append(bias, xTr, axis=0)
bias = np.ones(shape=(1,xTv.shape[1]))
xTv = np.append(bias, xTv, axis=0)

lambdaa = 0.001
w_trained = train(xTr,yTr, lambdaa)
print(loss(w_trained, xTv, yTv, lambdaa))
print(r_squared(w_trained, xTv, yTv))



