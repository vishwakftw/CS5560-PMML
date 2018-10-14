import numpy as np
from scipy.stats import multivariate_normal as mvn_pdf

class EM(object):
    """
    Class for Expectation-Maximization for a Gaussian Mixture Model.

    Arguments:
        - X : input dataset
        - k : number of mixtures to consider
    """
    def __init__(self, X, k):
        self.X = X
        self.k = k

        # Algorithm specific data
        # Cluster fraction parameter \pi
        self.pi = np.full((self.k, ), 1 / self.k)

        # Cluster mean vectors
        # We shuffle the datapoints randomly, partition it into `k` pieces
        # and take the mean of the partitions as an initialization
        self.mu = np.array([np.mean(xsplit, axis=0) for xsplit in np.vsplit(np.random.permutation(self.X), self.k)])

        # Cluster covariance matrices
        # We assign identity matrices for all clusters
        self.sigma = np.array([np.identity(self.X.shape[1]) for _ in range(self.k)])

        # Responsibility parameters
        # Storing it column major because of nature of updates made
        self.r = np.zeros((self.X.shape[0], self.k), order='F')

    def _E_step(self):
        """
        Function to perform the E-step - re-assigning responsibility probabilities for every datapoint
        """
        for k_ in range(self.k):
            self.r[:, k_] = mvn_pdf(self.X, self.mu[k_], self.sigma[k_])
        self.r = self.r / np.sum(self.r, axis=1)

    def _M_step(self):
        """
        Function to perform the M-step - re-assigning the cluster fractions, means and covariances
        """
        self.pi = np.sum(self.r, axis=0) / self.X.shape[0]
        for k_ in range(self, k):
            self.mu[k_] = np.average(self.X, weights=self.r[:, k_])
            self.sigma[k_] = np.average([np.outer(xi, xi) for xi in self.X], weights=self.r[:, k_])
            self.sigma[k_] = self.sigma[k_] - np.outer(self.mu[k_], self.mu[k_])

    def fit(self, iterations=100, get_progress=False):
        """
        Function to fit the data for a given number of iterations

        Arguments:
            - iterations : Number of iterations to run the algorithm for. Default: 100
            - get_progress : The log-likelihood is given at every iteration of the algorithm if True.
                             Default: False
        """
        for i in range(iterations):
            self._E_step()
            self._M_step()

        if get_progress:
            raise NotImplementedError("Yet to do")
