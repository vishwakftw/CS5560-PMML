import numpy as np
from scipy.stats import multivariate_normal as mvn

class GMM(object):
    """
    Class for gradient ascent based algorithm for a Gaussian Mixture Model.

    Arguments:
        - X : input dataset
        - k : number of mixtures to consider
        - seed : Integer for setting seed. Default: None
    """
    def __init__(self, X, k, seed=None):
        self.X = X
        self.k = k

        # Algorithm specific data
        # Cluster fraction parameter \pi
        self.pi = np.full((self.k, ), 1 / self.k)

        if seed is None:
            # Cluster mean vectors
            # This is assigned randomly
            self.mu = [np.random.randn(self.X.shape[1]) for _ in range(self.k)]

            # Cluster covariance matrices
            # The second step is to make the matrices PD
            self.sigma = [np.random.randn(self.X.shape[1], self.X.shape[1]) for _ in range(self.k)]
            self.sigma = [np.matmul(s.T, s) + 1e-05 * np.identity(self.X.shape[1]) for s in self.sigma]
        else:
            rs = np.random.RandomState(seed)
            self.mu = [rs.randn(self.X.shape[1]) for _ in range(self.k)]

            self.sigma = [rs.randn(self.X.shape[1], self.X.shape[1]) for _ in range(self.k)]
            self.sigma = [np.matmul(s.T, s) + 1e-05 * np.identity(self.X.shape[1]) for s in self.sigma]

    def _get_probability_matrix(self, X, cluster_prob, means, covariances):
        """
        Function to return the matrix of size m x d where entry [i,j] represents
        the probability of datapoint i w.r.t. jth MVN

        Arguments:
            - X : data
            - cluster_prob : cluster probability
            - means : mean vectors
            - covariances : covariance matrices

        Returns:
            - d x m matrix as described above
        """
        matrix = np.array([cluster_prob[k_] * mvn(means[k_], covariances[k_]).pdf(X) for k_ in range(len(means))]).T
        return matrix

    def _pi_grad(self, X, cluster_prob, means, covariances):
        """
        Function to get gradients of the cluster probability vector w.r.t. the log likelihood over the dataset

        Arguments:
            - X :data
            - cluster_prob : cluster probability
            - means: mean vectors
            - covariances : covariance matrices

        Returns:
            - gradient of cluster probability vector
        """
        prob_matrix = self._get_probability_matrix(X, cluster_prob, means, covariances)
        # m size vector, representing the sum of probs for all clusters
        prob_matrix_reduce_k = prob_matrix.sum(axis=1)
        pi_grads = []
        for k_ in range(len(means)):
            pi_grad = 0.0
            for i in range(0, X.shape[0]):
                pi_grad += (prob_matrix[i, k_] / prob_matrix_reduce_k[i])
            pi_grads.append(pi_grad / cluster_prob[k_])
        return pi_grads

    def _mu_grad(self, X, cluster_prob, means, covariances):
        """
        Function to get gradients of the mean vectors w.r.t. the log likelihood over the dataset

        Arguments:
            - X : data
            - cluster_prob : cluster probability
            - means : mean vectors
            - covariances : covariance matrices

        Returns:
            - list of vectors consisting of derivatives of individual mean vectors
        """
        prob_matrix = self._get_probability_matrix(X, cluster_prob, means, covariances)
        # m size vector, representing the sum of probs for all clusters
        prob_matrix_reduce_k = prob_matrix.sum(axis=1)
        mu_grads = []
        for k_ in range(len(means)):
            mu_grad = np.zeros((X.shape[1],))
            for i in range(0, X.shape[0]):
                mu_grad += ((prob_matrix[i, k_] / prob_matrix_reduce_k[i]) * (means[k_] - X[i]))
            mu_grad = np.matmul(-np.linalg.inv(covariances[k_]), mu_grad)
            mu_grads.append(mu_grad)
        return mu_grads

    def _sigma_grad(self, X, cluster_prob, means, covariances):
        """
        Function to get gradients of the covariance matrices w.r.t. the log likelihood over the dataset

        Arguments:
            - X : data
            - cluster_prob : cluster probability
            - means : mean vectors
            - covariances : covariance matrices

        Returns:
            - list of matrices consisting of derivatives of individual matrices
        """
        prob_matrix = self._get_probability_matrix(X, cluster_prob, means, covariances)
        # m size vector, representing the sum of probs for all clusters
        prob_matrix_reduce_k = prob_matrix.sum(axis=1)
        sigma_grads = []
        for k_ in range(len(means)):
            const_weight = 0.0
            g2 = np.zeros((X.shape[1], X.shape[1]))
            cov_inv = np.linalg.inv(covariances[k_])
            for i in range(0, X.shape[0]):
                sum_term = prob_matrix[i, k_] / prob_matrix_reduce_k[i]
                const_weight += sum_term
                tmp_vec = np.matmul(cov_inv, X[i] - means[k_])
                g2 += np.outer(tmp_vec, tmp_vec) * sum_term
            g1 = -const_weight * 0.5 * cov_inv
            sigma_grads.append(g1 + g2)
        return sigma_grads

    def _get_log_likelihood(self, X, pis, mus, sigmas):
        all_probs = self._get_probability_matrix(X, pis, mus, sigmas).T
        assert all_probs.shape == (len(mus), X.shape[0]), "Size mismatch in _get_log_likelihood (all_probs)"
        ll = all_probs.sum(axis=0)
        assert ll.shape == (X.shape[0],), "Size mismatch in _get_log_likelihood (ll)"
        ll = np.log(ll).sum()
        return ll

    def _choose_step(self, X, cluster_prob, means, covariances, alpha=1.0, tau=0.5):
        """
        Function to get the best step size based on Wolfe conditions. Temporary steps are taken with
        adaptive step size to check if the Wolfe condition is met, and returns the update with the highest
        gain (loss) in the (negative) log likelihood.

        Arguments:
            - X : data
            - cluster_prob : cluster probability
            - means : mean vectors
            - covariances : covariance matrices
            - alpha : starting learning rate. Default: 1.0
            - tau : multiplicative factor. Default: 0.5

        Returns:
            - the modified parameters and the log likelihood at that step
        """
        initial_ll = self._get_log_likelihood(X, cluster_prob, means, covariances)
        while 1:
            covs_grad = self._sigma_grad(X, cluster_prob, means, covariances)
            means_grad = self._mu_grad(X, cluster_prob, means, covariances)
            cluster_prob_grad = self._pi_grad(X, cluster_prob, means, covariances)
            updated_covs = [covariances[k_] + alpha * covs_grad[k_] + 1e-06 * np.identity(X.shape[1])
                            for k_ in range(len(means))]  # to make the grads PD
            updated_means = [means[k_] + alpha * means_grad[k_] for k_ in range(len(means))]
            updated_cluster_prob = np.array([cluster_prob[k_] + alpha * cluster_prob_grad[k_]
                                             for k_ in range(len(means))])
            try:
                updated_ll = self._get_log_likelihood(X, updated_cluster_prob, updated_means, updated_covs)
            except ValueError:
                alpha = alpha * tau
                continue
            if updated_ll > initial_ll:
                return (updated_cluster_prob, updated_means, updated_covs, updated_ll, alpha)
            alpha = alpha * tau

    def fit(self, iterations=100):
        """
        Function to fit the data for a given number of iterations

        Arguments:
            - iterations : Number of iterations to run the algorithm for. Default: 100

        Returns:
            - List of log-likelihoods
        """
        LLs = []

        for i in range(1, iterations + 1):
            pi, mu, sigma, LL, ss = self._choose_step(self.X, self.pi, self.mu, self.sigma,
                                                      alpha=1.15, tau=0.675)
            self.pi, self.mu, self.sigma = pi, mu, sigma
            print("{} / {}\tLog Likelihood = {}, final step size = {}".format(i, iterations, LL, ss))
            LLs.append(LL)
        return LLs
