from EM import GMM as EGMM
from GA import GMM as GGMM

import numpy as np

from argparse import ArgumentParser as AP
from matplotlib import pyplot as plt

def preprocess_data(data):
    for i in range(0, data.shape[1]):
        std = np.std(data[:, i])
        if std != 0:
            data[:, i] = (data[:, i] - np.mean(data[:, i])) / std
    return data

p = AP()
p.add_argument('--dataset', type=str, required=True, help='Dataset file in CSV format')
p.add_argument('--algorithm', type=str, choices=['EM', 'GA', 'both'], default='EM',
                              help='Algorithm to use for mixture model')
p.add_argument('--k', type=int, default=2, help='Number of clusters')
p.add_argument('--iters', type=int, default=100, help='Number of iterations')
p = p.parse_args()

data = np.genfromtxt(p.dataset, delimiter=',')

data = preprocess_data(data)  # normalizes the dataset N(0, 1)

if p.algorithm != 'both':
    if p.algorithm == 'EM':
        gmm = EGMM(X=data, k=p.k, seed=3141)
        init_ll = gmm._get_log_likelihood()
        lls = gmm.fit(iterations=p.iters)
        nlls = [-init_ll] + [-ll for ll in lls]

    else:
        gmm = GGMM(X=data, k=p.k, seed=3141)
        init_ll = gmm._get_log_likelihood(data, gmm.pi, gmm.mu, gmm.sigma)
        lls = gmm.fit(iterations=p.iters)
        nlls = [-init_ll] + [-ll for ll in lls]

    plt.plot(list(range(0, p.iters + 1)), nlls, 'b-', lw=3.0)
    plt.ylabel('Negative Log-likelihood')
    plt.xlabel('Iterations')
    plt.title(p.algorithm, fontsize=20)
    plt.show()

else:
    egmm = EGMM(X=data, k=p.k, seed=3141)
    init_ll = egmm._get_log_likelihood()
    lls = egmm.fit(iterations=p.iters)
    e_nlls = [-init_ll] + [-ll for ll in lls]

    ggmm = GGMM(X=data, k=p.k, seed=3141)
    init_ll = ggmm._get_log_likelihood(data, ggmm.pi, ggmm.mu, ggmm.sigma)
    lls = ggmm.fit(iterations=p.iters)
    g_nlls = [-init_ll] + [-ll for ll in lls]

    plt.plot(list(range(0, p.iters + 1)), e_nlls, c='orange', ls='-', lw=3.0,
             label='Expectation maximization')
    plt.plot(list(range(0, p.iters + 1)), g_nlls, c='green', ls='-', lw=3.0,
             label='Gradient ascent')
    plt.ylabel('Negative Log-likelihood')
    plt.xlabel('Iterations')
    plt.legend()
    plt.title('EM vs GA')
    plt.show()
