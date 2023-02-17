import math

import numpy as np
import pandas as pd

from do_training import train
from stat_func import r_squared, loss


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



