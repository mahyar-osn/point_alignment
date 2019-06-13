import numpy as np
from Tools import Tools


class Optimization(object):
    """
    This is the Expectation Maximization algorithm class to compute the fitting parameters.
    The idea of this algorithm is first to guess the values of parameters, and then use the Bayes' theorem
    to compute the posterior probability distributions P(m|x_n) of mixture components, which is the
    expectation of the algorithm. The updated parameter values are then found by minimizing the expectation og
    the negative log likelihood function.
    """
    def __init__(self, X, Y, sigma2=None, max_iterations=100, tolerance=0.001, w=0, *args, **kwargs):
        """

        :param X: The first point set (the fiducial landmarks from the heart image).
        :param Y: The second point set (the material points fron the scaffold).
        :param sigma2: Isotropic covariance - must be initialized.
        :param max_iterations: Maximum fitting iterations.
        :param tolerance: Error tolerance.
        :param w: The weight of the uniform distribution. must be 0 =< w >= 1.
        :param args:
        :param kwargs:
        """

        if type(X) is not np.ndarray or X.ndim != 2:
            raise ValueError("Fiducial landmarks must be a numpy array with at least 2 dimensions.")
        if type(Y) is not np.ndarray or Y.ndim != 2:
            raise ValueError("Scaffold must be a numpy array with at least 2 dimensions.")
        if X.shape[1] != Y.shape[1]:
            raise ValueError("Scaffold and fiducial landmarks must have the same dimension.")

        self.X = X
        self.Y = Y
        self.sigma2 = sigma2
        (self.N, self.D) = self.X.shape
        (self.M, _) = self.Y.shape
        self.tolerance = tolerance
        self.w = w
        self.max_iterations = max_iterations
        self.iteration = 0
        self.err = self.tolerance + 1
        self.P = np.zeros((self.M, self.N))
        self.Pt1 = np.zeros((self.N,))
        self.P1 = np.zeros((self.M,))
        self.Np = 0
        self.q = None
        self.tools = Tools()

    def fit(self, callback=lambda **kwargs: None):
        self.transform_scaffold()
        if self.sigma2 is None:
            self.sigma2 = self.tools.initialize_sigma2(self.X, self.TY)
        self.q = -self.err - self.N * self.D / 2 * np.log(self.sigma2)
        while self.iteration < self.max_iterations and self.err > self.tolerance:
            self.iterate()
            if callable(callback):
                kwargs = {'iteration': self.iteration, 'error': self.err, 'X': self.X, 'Y': self.TY}
                callback(**kwargs)

        return self.TY, self.get_fitting_parameters()

    def get_fitting_parameters(self):
        raise NotImplementedError("Fitting parameters should be defined in child classes"
                                  "i.e. RigidFitting or DeformableFitting.")

    def iterate(self):
        self.expectation()
        self.maximization()
        self.iteration += 1

    def expectation(self):
        P = np.zeros((self.M, self.N))

        for i in range(0, self.M):
            diff = self.X - np.tile(self.TY[i, :], (self.N, 1))
            diff = np.multiply(diff, diff)
            P[i, :] = P[i, :] + np.sum(diff, axis=1)

        c = (2 * np.pi * self.sigma2) ** (self.D / 2)
        c = c * self.w / (1 - self.w)
        c = c * self.M / self.N

        P = np.exp(-P / (2 * self.sigma2))
        den = np.sum(P, axis=0)
        den = np.tile(den, (self.M, 1))
        den[den == 0] = np.finfo(float).eps
        den += c

        self.P = np.divide(P, den)
        self.Pt1 = np.sum(self.P, axis=0)
        self.P1 = np.sum(self.P, axis=1)
        self.Np = np.sum(self.P1)

    def maximization(self):
        self.update_transform()
        self.transform_scaffold()
        self.update_variance()
