import numpy as np
from Optimization import Optimization
from Tools import Tools


class DeformableFitting(Optimization):
    def __init__(self, alpha=7, beta=7, *args, **kwargs):

        super(DeformableFitting, self).__init__(*args, **kwargs)

        self.alpha = 2 if alpha is None else alpha
        self.beta = 2 if alpha is None else beta
        self.W = np.zeros((self.M, self.D))
        self.G = Tools.make_kernel(self.Y, self.beta)
        self.TY = None

    def update_transform(self):
        A = np.dot(np.diag(self.P1), self.G) + self.alpha * self.sigma2 * np.eye(self.M)
        B = np.dot(self.P, self.X) - np.dot(np.diag(self.P1), self.Y)
        self.W = np.linalg.solve(A, B)

    def transform_scaffold(self, Y=None):
        if Y is None:
            self.TY = self.Y + np.dot(self.G, self.W)
            return
        else:
            return Y + np.dot(self.G, self.W)

    def update_variance(self):
        qprev = self.sigma2

        xPx = np.dot(np.transpose(self.Pt1), np.sum(np.multiply(self.X, self.X), axis=1))
        yPy = np.dot(np.transpose(self.P1), np.sum(np.multiply(self.Y, self.Y), axis=1))
        trPXY = np.sum(np.multiply(self.Y, np.dot(self.P, self.X)))

        self.sigma2 = (xPx - trPXY) / (self.Np * self.D)

        if self.sigma2 <= 0:
            self.sigma2 = self.tolerance / 10
        self.err = np.abs(self.sigma2 - qprev)

    def get_fitting_parameters(self):
        return self.G, self.W
