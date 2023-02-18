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


def loss_ridge(w, xT, yT, lambdaa):
    XT = xT.transpose()
    Y = yT.transpose()
    XTw = XT.dot(w)
    XTwMinusy = XTw - Y
    XTwMinusyT = XTwMinusy.transpose()
    XTwMinusyT_XTwMinusy = XTwMinusyT.dot(XTwMinusy)
    regularizer = lambdaa * w.transpose().dot(w)
    loss = XTwMinusyT_XTwMinusy + regularizer
    return loss


def loss_mse(w, xT, yT):
    XT = xT.transpose()
    Y = yT.transpose()
    XTw = XT.dot(w)
    XTwMinusy = XTw - Y
    XTwMinusyT = XTwMinusy.transpose()
    XTwMinusyT_XTwMinusy = XTwMinusyT.dot(XTwMinusy)
    loss = XTwMinusyT_XTwMinusy
    return loss


def loss_mae(w, xT, yT):
    XT = xT.transpose()
    Y = yT.transpose()
    XTw = XT.dot(w)
    XTwMinusy = XTw - Y
    XTwMinusyAbs = np.absolute(XTwMinusy)
    loss = np.sum(XTwMinusyAbs)
    return [[loss]]

def do_predict(w, xT):
    XT = xT.transpose()
    XTw = XT.dot(w)
    return XTw.reshape((1, -1))

def do_split(df, nTest):
    test = df.sample(n=nTest, axis=0, replace=False)
    train = df.drop(index=test.index)
    return train, test