from EM import GMM as EGMM
from GA import GMM as GGMM

import numpy as np

from argparse import ArgumentParser as AP
from matplotlib import pyplot as plt

p = AP()
p.add_argument('--dataset', type=str, required=True, help='Dataset file in CSV format')
p.add_argument('--algorithm', type=str, choices=['EM', 'GA'], default='EM', help='Algorithm to use for mixture model')
p.add_argument('--k', type=int, default=2, help='Number of clusters')
p.add_argument('--iters', type=int, default=100, help='Number of iterations')
p = p.parse_args()

data = np.genfromtxt(p.dataset, delimiter=',')
if p.algorithm == 'EM':
    gmm = EGMM(X=data, k=p.k)
    lls = gmm.fit(iterations=p.iters)

else:
	gmm = GGMM(X=data, k=p.k)
	lls = gmm.fit(iterations=p.iters)

plt.plot(list(range(1, p.iters + 1)), lls, 'b-', marker='o', mfc='k', mec='k', ms=5.0, lw=3.0)
plt.show()