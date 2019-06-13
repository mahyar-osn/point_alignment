import numpy as np


class Tools(object):
    def __init__(self):
        pass

    @staticmethod
    def initialize_sigma2(X, Y):
        (N, D) = X.shape
        (M, _) = Y.shape
        XX = np.reshape(X, (1, N, D))
        YY = np.reshape(Y, (M, 1, D))
        XX = np.tile(XX, (M, 1, 1))
        YY = np.tile(YY, (1, N, 1))
        diff = XX - YY
        err = np.multiply(diff, diff)
        return np.sum(err) / (D * M * N)

    @staticmethod
    def make_kernel(Y, beta):
        (M, D) = Y.shape
        XX = np.reshape(Y, (1, M, D))
        YY = np.reshape(Y, (M, 1, D))
        XX = np.tile(XX, (M, 1, 1))
        YY = np.tile(YY, (1, M, 1))
        diff = XX - YY
        diff = np.multiply(diff, diff)
        diff = np.sum(diff, 2)
        return np.exp(-diff / (2 * beta))

