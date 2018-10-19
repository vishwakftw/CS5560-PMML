import numpy as np
from scipy.stats import multivariate_normal as mvn

class GMM(object):
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
        # This is assigned randomly
        self.mu = [np.random.randn(self.X.shape[1]) for _ in range(self.k)]

        # Cluster covariance matrices
        # The second step is to make the matrices PD
        self.sigma = [np.random.randn(self.X.shape[1], self.X.shape[1]) for _ in range(self.k)]
        self.sigma = [np.matmul(s.T, s) + 1e-05 * np.identity(self.X.shape[1]) for s in self.sigma]

        # Responsibility parameters
        # Storing it column major because of nature of updates made
        self.r = np.zeros((self.X.shape[0], self.k), order='F')

    def _E_step(self, r, pis, mus, sigmas, X):
        """
        Function to perform the E-step - re-assigning responsibility probabilities for every datapoint

        Arguments:
            - r : responsibility parameters
            - pis : cluster probabilities
            - mus : mean vectors
            - sigmas : covariance matrices
            - X : data

        Returns:
            - r : new responsibility parameters
        """
        for k_ in range(r.shape[1]):
            mvn_k = mvn(mus[k_], sigmas[k_])
            r[:, k_] = pis[k_] * mvn_k.pdf(X)
        r = r / np.sum(r, axis=1, keepdims=True)
        return r

    def _M_step(self, r, X):
        """
        Function to perform the M-step - re-assigning the cluster fractions, means and covariances

        Arguments:
            - r : responsibility parameters
            - X : data

        Returns:
            - (pi, mus, sigmas) : New cluster probabilities, cluster means and covariances
        """
        pi = np.sum(r, axis=0) / self.X.shape[0]
        mus = [None for _ in range(r.shape[1])]
        sigmas = [None for _ in range(r.shape[1])]
        for k_ in range(r.shape[1]):
            mus[k_] = np.average(X, weights=r[:, k_], axis=0)
            sigmas[k_] = np.average([np.outer(xi, xi) for xi in X], weights=r[:, k_], axis=0)
            sigmas[k_] = sigmas[k_] - np.outer(mus[k_], mus[k_])
            sigmas[k_] = sigmas[k_] + np.identity(sigmas[k_].shape[0]) * 1e-06  # To prevent singular problems
        return pi, mus, sigmas

    def fit(self, iterations=100, get_progress=True):
        """
        Function to fit the data for a given number of iterations

        Arguments:
            - iterations : Number of iterations to run the algorithm for. Default: 100
            - get_progress : The log-likelihood is given at every iteration of the algorithm if True.
                             Default: True

        Returns:
            - Empty list if get_progress is False
            - List of log-likelihoods if get_progress is True
        """
        LLs = []

        for i in range(1, iterations + 1):
            r = self._E_step(self.r, self.pi, self.mu, self.sigma, self.X)
            self.r = r
            pi, mu, sigma = self._M_step(self.r, self.X)
            self.pi, self.mu, self.sigma = pi, mu, sigma

            if get_progress:
                LL = self._get_log_likelihood()
                print("{} / {}\tLog Likelihood = {}".format(i, iterations, LL))
                LLs.append(LL)

        return LLs

    def _get_log_likelihood(self):
        all_probs = np.array([mvn(self.mu[k_], self.sigma[k_]).pdf(self.X) for k_ in range(self.k)])
        all_probs = self.pi.reshape(-1, 1) * all_probs
        assert all_probs.shape == (self.k, self.X.shape[0]), "Size mismatch in _get_log_likelihood (all_probs)"
        ll = all_probs.sum(axis=0)
        assert ll.shape == (self.X.shape[0],), "Size mismatch in _get_log_likelihood (ll)"
        ll = np.log(ll).sum()
        return ll
