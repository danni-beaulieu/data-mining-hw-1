import numpy as np
from do_regresssion import ridge, maereg, msereg
from gradient_descent import grdescent

def train(xTr,yTr, lambdaa):
# INPUT:
# xTr
# yTr
#
# OUTPUT: w_trained


    # f = lambda w: ridge(w, xTr, yTr, lambdaa)
    f = lambda w: msereg(w, xTr, yTr)
    # f = lambda w: maereg(w, xTr, yTr)
    w_trained = grdescent(f, np.zeros((xTr.shape[0], 1)), 1e-05, 10000)
    return w_trained