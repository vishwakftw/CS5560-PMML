import numpy as np

from collections import Counter
from argparse import ArgumentParser as AP


def log_mvn_ll(inp_vec, mean_vec, cov_logdet, cov_inv):
    """
    Returns the log likelihood of the input vector `inp_vec` for a
    Multivariate Normal distribution
    """
    return -0.5 * (cov_logdet + np.dot(np.dot(cov_inv, inp_vec - mean_vec), inp_vec - mean_vec))

def log_uvn_ll(inp_vec, mean_list, var_log_list, var_inv_list):
    """
    Returns the log likelihood of the input vector against a set of univariate Gaussians
    per dimension (excludes a constant log 2 * pi term)
    """
    ll = 0.0
    for d in range(inp_vec.shape[0]):
        ll += ((inp_vec[d] - mean_list[d]) * var_inv_list[d] + var_log_list[d])
    return -0.5 * ll


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
del tr_data

# Build the models
# Bayes Classifier
if p.model == 'bayes-clf':
    # Estimate the pi parameter for the multinoulli
    pi = dict(Counter(tr_y))
    for k in pi.keys():
        pi[k] = pi[k] / tr_y.shape[0]

    # Estimate the means, determinant and inverse of covariances for the multivariate normal per class
    mean = dict()
    cov_det = dict()
    cov_inv = dict()
    for k in pi.keys():
        tr_x_k = tr_x[tr_y == k]  # Isolates those examples belonging to a specific class
        mean[k] = np.mean(tr_x_k, axis=0)
        tmp = np.outer(tr_x_k[0], tr_x_k[0])
        for i in range(1, tr_x_k.shape[0]):
            tmp += np.outer(tr_x_k[i] - mean[k], tr_x_k[i] - mean[k])
        tmp /= tr_x_k.shape[0]
        cov_logdet[k] = np.log(np.linalg.det(np.pi * 2 * tmp))
        cov_inv[k] = np.linalg.inv(tmp)
        pi[k] = np.log(pi[k])

# Naive-Bayes Classifier
else:
    # Estimate the pi parameter for the multinoulli
    pi = dict(Counter(tr_y))
    for k in pi.keys():
        pi[k] = pi[k] / tr_y.shape[0]

    # Estimate the means, and variances for the univariate normal per dimension per class
    mean = dict()
    var_inv = dict()
    var_log = dict()
    for k in sorted(pi.keys()):
        tr_x_k = tr_x[tr_y == k]  # Isolates those examples belonging to a specific class
        mean[k] = np.mean(tr_x_k, axis=0)
        var_inv_k = []
        var_log_k = []
        for d in tr_x_k.shape[1]:
            var = np.cov(tr_x_k[:, d], bias=True)
            var_inv_k.append(1 / var)
            var_log_k.append(np.log(var))
        var_inv[k] = var_inv_k
        var_log[k] = var_log_k
        pi[k] = np.log(pi[k])

# Test the model
if p.test_datasrc is None:
    tst_x = tr_x
    tst_y = tr_y
else:
    tst_data = np.genfromtxt(p.test_datasrc, ',')
    tst_x, tst_y = np.hsplit(tst_data, [tst_data.shape[1] - 1])
    del tst_data

if p.model == 'bayes-clf':
    preds = []
    max_k = -1, max_ll = -np.inf
    for inp in tst_x:
        for k in pi.keys():
            ll_k = pi[k] + log_mvn_ll(inp, mean[k], cov_logdet[k], cov_inv[k])
            if ll_k > max_ll:
                max_k, max_ll = k, ll_k
        preds.append(k)

else:
    preds = []
    max_k = -1, max_ll = -np.inf
    for inp in tst_x:
        for k in pi.keys():
            ll_k = pi[k] + log_uvn_ll(inp, mean[k], variances[k])
            if ll_k > max_ll:
                max_k, max_ll = k, ll_k
        preds.append(k)

print("Accuracy achieved: {} / {}".format((preds == tst_y), tst_y.shape[0]))
