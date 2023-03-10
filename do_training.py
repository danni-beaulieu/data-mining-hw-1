import math

import numpy as np
import random
from do_regression import ridge, maereg, msereg, mseregstochastic
from gradient_descent import grdescent

def train_ridge(xTr,yTr, eta, lambdaa):
# INPUT:
# xTr
# yTr
#
# OUTPUT: w_trained
    f = lambda w: ridge(w, xTr, yTr, lambdaa)
    w_trained = grdescent(f, init_w(np.zeros((xTr.shape[0], 1))), eta, 10000)
    return w_trained


def train_mse(xTr,yTr,eta):
# INPUT:
# xTr
# yTr
#
# OUTPUT: w_trained
    f = lambda w: msereg(w, xTr, yTr)
    w_trained = grdescent(f, init_w(np.zeros((xTr.shape[0], 1))), eta, 10000)
    return w_trained


def train_mae(xTr,yTr, eta):
# INPUT:
# xTr
# yTr
#
# OUTPUT: w_trained
    f = lambda w: maereg(w, xTr, yTr)
    w_trained = grdescent(f, init_w(np.zeros((xTr.shape[0], 1))), eta, 10000)
    return w_trained


def init_w(w):
    n = w.shape[0]
    for i in range(n):
        a = random.uniform(1, n - 1)
        w[i] = a / math.sqrt(n)
    return w
