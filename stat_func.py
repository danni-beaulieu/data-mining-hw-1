import numpy as np


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


def loss(w, xT, yT, lambdaa):
    XT = xT.transpose()
    Y = yT.transpose()
    XTw = XT.dot(w)
    XTwMinusy = XTw - Y
    XTwMinusyT = XTwMinusy.transpose()
    XTwMinusyT_XTwMinusy = XTwMinusyT.dot(XTwMinusy)
    regularizer = lambdaa * w.transpose().dot(w)
    loss = XTwMinusyT_XTwMinusy + regularizer
    return loss