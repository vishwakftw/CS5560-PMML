import numpy as np

from collections import Counter
from argparse import ArgumentParser as AP

p = AP()
p.add_argument('--train_datasrc', type=str, required=True, help='Source for the training dataset (CSV)')
p.add_argument('--test_datasrc', type=str, default=None, help='Source for the testing dataset (CSV), if not entered\
                                                               testing is done on the training dataset')
p.add_argument('--model', type=str, required=True, choices=['bayes-clf', 'naive-bayes-clf'],
                          help='Model name: \"bayes-clf\" for Gaussian Discriminant Analysis (Bayes Classifier)\
                                \"naive-bayes-clf\" for Naive Bayes Classifier')
p = p.parse_args()

# Get the dataset
tr_data = np.genfromtxt(p.train_datasrc, delimiter=',')
# Assume last column are the outputs
tr_x, tr_y = np.hsplit(tr_data, [tr_data.shape[1] - 1])

# Build the models
# Bayes Classifier
if p.model == 'bayes-clf':
    # Estimate the pi parameter for the multinoulli
    pi = dict(Counter(tr_y))
    for k in pi.keys():
        pi[k] = pi[k] / tr_y.shape[0]

    # Estimate the means, determinant and inverse of covariances for the multivariate normal per class
    mean = []
    cov_det = []
    cov_inv = []
    for k in sorted(pi.keys()):
        tr_x_k = tr_x[tr_y == k]  # Isolates those examples belonging to a specific class
        mean.append(np.mean(tr_x_k, axis=0))
        tmp = np.outer(tr_x_k[0], tr_x_k[0])
        for i in range(1, tr_x_k.shape[0]):
            tmp += np.outer(tr_x_k[i] - mean[k], tr_x_k[i] - mean[k])
        tmp /= tr_x_k.shape[0]
        cov_det.append(np.sqrt(np.linalg.det(np.pi * 2 * tmp)))
        cov_inv.append(np.linalg.inv(tmp))

# Naive-Bayes Classifier
else:
    # Estimate the pi parameter for the multinoulli
    pi = dict(Counter(tr_y))
    for k in pi.keys():
        pi[k] = pi[k] / tr_y.shape[0]

    # Estimate the means, and variances for the univariate normal per dimension per class
    mean = dict()
    variances = dict()
    for k in sorted(pi.keys()):
        tr_x_k = tr_x[tr_y == k]  # Isolates those examples belonging to a specific class
        mean.append(np.mean(tr_x_k, tr_x_k[0]))
        cov_k_d = []
        for d in tr_x_k.shape[1]:
            cov_k_d.append(np.cov(tr_x_k[:, d], bias=True))
        variances.append(cov_k_d)
