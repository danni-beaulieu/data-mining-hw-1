import numpy as np


def ridge(w,xTr,yTr,lambdaa):
#
# INPUT:
# w weight vector (default w=0)
# xTr:dxn matrix (each column is an input vector)
# yTr:1xn matrix (each entry is a label)
# lambdaa: regression constant
#
# OUTPUTS:
# loss = the total loss obtained with w on xTr and yTr
# gradient = the gradient at w
#
# [d,n]=size(xTr);

    XT = xTr.transpose()
    Y = yTr.transpose()
    XTw = XT.dot(w)
    XTwMinusy = XTw - Y
    XTwMinusyT = XTwMinusy.transpose()
    XTwMinusyT_XTwMinusy = XTwMinusyT.dot(XTwMinusy)
    regularizer = lambdaa * w.transpose().dot(w)
    loss = (XTwMinusyT_XTwMinusy + regularizer) / yTr.shape[1]
    g_loss = 2 * xTr.dot(XTwMinusy) / yTr.shape[1]
    g_reg = 2 * lambdaa * w / yTr.shape[1]
    gradient = g_loss.reshape((g_loss.shape[0],)) + g_reg.reshape((g_reg.shape[0],))
    gradient = gradient.reshape((-1, 1))

    return loss,gradient


def msereg(w,xTr,yTr):
#
# INPUT:
# w weight vector (default w=0)
# xTr:dxn matrix (each column is an input vector)
# yTr:1xn matrix (each entry is a label)
#
# OUTPUTS:
# loss = the total loss obtained with w on xTr and yTr
# gradient = the gradient at w
#
# [d,n]=size(xTr);

    XT = xTr.transpose()
    Y = yTr.transpose()
    XTw = XT.dot(w)
    XTwMinusy = XTw - Y
    XTwMinusyT = XTwMinusy.transpose()
    XTwMinusyT_XTwMinusy = XTwMinusyT.dot(XTwMinusy)
    loss = XTwMinusyT_XTwMinusy / yTr.shape[1]
    g_loss = 2 * xTr.dot(XTwMinusy) / yTr.shape[1]
    gradient = g_loss.reshape((g_loss.shape[0],))
    gradient = gradient.reshape((-1, 1))

    return loss,gradient


def maereg(w,xTr,yTr):
#
# INPUT:
# w weight vector (default w=0)
# xTr:dxn matrix (each column is an input vector)
# yTr:1xn matrix (each entry is a label)
#
# OUTPUTS:
# loss = the total loss obtained with w on xTr and yTr
# gradient = the gradient at w
#
# [d,n]=size(xTr);

    XT = xTr.transpose()
    Y = yTr.transpose()
    XTw = XT.dot(w)
    XTwMinusy = XTw - Y
    XTwMinusyAbs = np.absolute(XTwMinusy)
    XTwMinusy_div_XTwMinusyAbs = np.divide(XTwMinusy,XTwMinusyAbs)
    loss = np.sum(XTwMinusyAbs) / yTr.shape[1]
    g_loss = xTr.dot(XTwMinusy_div_XTwMinusyAbs) / yTr.shape[1]
    gradient = g_loss.reshape((g_loss.shape[0],))
    gradient = gradient.reshape((-1, 1))

    return loss,gradient

