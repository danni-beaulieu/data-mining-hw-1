import numpy as np
from numpy import inf


def grdescent(func, w0, stepsize, maxiter, tolerance = 1e-2):
# INPUT:
# func function to minimize
# w0 = initial weight vector
# stepsize = initial gradient descent stepsize
# tolerance = if norm(gradient)<tolerance, it quits
#
# OUTPUT:
# w = final weight vector

    w = w0
    eps = 2.2204e-14 #minimum step size for gradient descent
    if (stepsize < eps):
        stepsize = eps

    loss = inf
    for t in range(maxiter):
        prevLoss = loss
        loss, gradient = func(w)

        # 1-t*s method
        # stepsize = 1 - t * stepsize

        # 1.01/.5 method
        # if (loss < prevLoss):
        #     stepsize = stepsize * 1.01
        # else:
        #     stepsize = stepsize * 0.5

        # 1/t method
        # stepsize = 1.0 / (t + 1)
        w = w - stepsize * gradient

        if (np.linalg.norm(gradient, inf) <= tolerance):
            break
    return w