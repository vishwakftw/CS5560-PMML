import numpy as np

from utils import *
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
        ll += (np.power(inp_vec[d] - mean_list[d], 2) * var_inv_list[d] + var_log_list[d])
    return -0.5 * ll


p = AP()
p.add_argument('--train_datasrc', type=str, required=True, help='Source for the training dataset')
p.add_argument('--test_datasrc', type=str, default=None, help='Source for the testing dataset,\
                                                               if not entered testing is done on the training dataset')
p.add_argument('--n_attributes', type=int, required=True, help='Number of attributes in the dataset')
p.add_argument('--format', type=str, default='csv', choices=['libsvm', 'csv'], help='Format specifier')
p.add_argument('--model', type=str, required=True, choices=['bayes-clf', 'naive-bayes-clf'],
                          help='Model name: \"bayes-clf\" for Gaussian Discriminant Analysis (Bayes Classifier)\
                                \"naive-bayes-clf\" for Naive Bayes Classifier')
p = p.parse_args()

# Get the dataset
if p.format == 'libsvm':
    tr_data = get_data_from_libsvm_format(p.train_datasrc, p.n_attributes)
else:
    tr_data = np.genfromtxt(p.train_datasrc, delimiter=',')
    get_data_in_libsvm_format(p.train_datasrc)
# Assume last column are the outputs
tr_x, tr_y = np.hsplit(tr_data, [tr_data.shape[1] - 1])
tr_y = tr_y.reshape(-1)
del tr_data

# Build the models
# Bayes Classifier
if p.model == 'bayes-clf':
    # Estimate the pi parameter for the multinoulli
    pi = Counter(list(tr_y))
    for k in pi.keys():
        pi[k] = pi[k] / tr_y.shape[0]

    # Estimate the means, determinant and inverse of covariances for the multivariate normal per class
    mean = dict()
    cov_logdet = dict()
    cov_inv = dict()
    for k in pi.keys():
        tr_x_k = tr_x[tr_y == k]  # Isolates those examples belonging to a specific class
        mean[k] = np.mean(tr_x_k, axis=0)
        tmp = np.outer(tr_x_k[0], tr_x_k[0])
        for i in range(1, tr_x_k.shape[0]):
            tmp += np.outer(tr_x_k[i] - mean[k], tr_x_k[i] - mean[k])
        tmp /= tr_x_k.shape[0]
        try:
            cov_inv[k] = np.linalg.inv(tmp)
        except np.linalg.linalg.LinAlgError:
            tmp = tmp + 1e-05 + np.eye(tmp.shape[0])
            cov_inv[k] = np.linalg.inv(tmp)
        cov_logdet[k] = np.log(np.linalg.det(np.pi * 2 * tmp))
        pi[k] = np.log(pi[k])

# Naive-Bayes Classifier
elif p.model == 'naive-bayes-clf':
    # Estimate the pi parameter for the multinoulli
    pi = Counter(list(tr_y))
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
        for d in range(tr_x_k.shape[1]):
            var = np.cov(tr_x_k[:, d], bias=True)
            if var == 0.0:
                var_inv_k.append(1 / (var + 1e-05))
                var_log_k.append(np.log(var + 1e-05))
            else:
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
    if p.format == 'libsvm':
        tst_data = get_data_from_libsvm_format(p.test_datasrc, p.n_attributes)
    else:
        tst_data = np.genfromtxt(p.test_datasrc, delimiter=',')
        get_data_in_libsvm_format(p.test_datasrc)
    tst_x, tst_y = np.hsplit(tst_data, [tst_data.shape[1] - 1])
    del tst_data

if p.model == 'bayes-clf':
    preds = []
    max_k = -1
    max_ll = -np.inf
    for inp in tst_x:
        for k in pi.keys():
            ll_k = pi[k] + log_mvn_ll(inp, mean[k], cov_logdet[k], cov_inv[k])
            if ll_k > max_ll:
                max_k, max_ll = k, ll_k
        preds.append(max_k)

elif p.model == 'naive-bayes-clf':
    preds = []
    max_k = -1
    max_ll = -np.inf
    for inp in tst_x:
        for k in pi.keys():
            ll_k = pi[k] + log_uvn_ll(inp, mean[k], var_log[k], var_inv[k])
            if ll_k > max_ll:
                max_k, max_ll = k, ll_k
        preds.append(max_k)

print("Accuracy achieved: {} / {} = {}".format(sum(preds == tst_y), tst_y.shape[0], sum(preds == tst_y) / tst_y.shape[0]))
if sum(preds == tst_y) != tst_y.shape[0]:
    misses = []
    for i in range(len(preds)):
        if preds[i] != tst_y[i]:
            misses.append(str(i + 1))
    print("Misclassification in: {}".format(', '.join(misses)))
