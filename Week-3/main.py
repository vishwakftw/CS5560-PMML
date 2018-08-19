import numpy as np

from numpy.linalg import inv
from argparse import ArgumentParser as AP

p = AP()
p.add_argument('--training_dataset', type=str, required=True,
                                     help='Source for the training dataset, in CSV format')
p.add_argument('--testing_dataset', type=str, default=None,
                                    help='Source for the testing dataset, in CSV format')
p.add_argument('--test_is_train', action='store_true',
                                  help='Toggle if you want to test on the training dataset')
p.add_argument('--input_dim', type=int, required=True,
                              help='Dimensionality of input data')
p.add_argument('--output_dim', type=int, required=True,
                               help='Dimensionality of output data')
p.add_argument('--shapecheck', action='store_true',
                               help='Check intermediate shapes for correctness')
p = p.parse_args()

tr_data = np.genfromtxt(p.training_dataset, delimiter=',')
if p.testing_dataset is not None:
    if p.test_is_train:
        raise ValueError('There should be exactly one source for testing dataset')
    else:
        te_data = np.genfromtxt(p.testing_dataset, delimiter=',')
else:
    if p.test_is_train:
        te_data = np.genfromtxt(p.training_dataset, delimiter=',')[:, :p.input_dim]
    else:
        raise ValueError('There should be exactly one source for testing dataset')

# Dataset checks
if tr_data.shape[1] != (p.input_dim + p.output_dim):
    raise ValueError('Input-output dimensions does not match the training dataset properties')
if te_data.shape[1] != p.input_dim:
    raise ValueError('Input dimensions does not match the testing dataset properties')

np.set_printoptions(precision=4)

# Parameter estimation
tr_data_mean = np.mean(tr_data, 0)
tr_data_cov = []
for i in range(0, tr_data.shape[0]):
    tmp = tr_data[i] - tr_data_mean
    tr_data_cov.append(np.outer(tmp, tmp))
tr_data_cov = np.mean(tr_data_cov, 0)
print("Estimated mean:\n{}".format(tr_data_mean))
print("Estimated covariance:\n{}".format(tr_data_cov))

# Building the parts of the function
tr_data_mean1, tr_data_mean2 = np.hsplit(tr_data_mean, [p.input_dim])
tr_data_cov1, tr_data_cov2 = np.vsplit(tr_data_cov, [p.input_dim])
tr_data_cov11, tr_data_cov12 = np.hsplit(tr_data_cov1, [p.input_dim])
tr_data_cov21, tr_data_cov22 = np.hsplit(tr_data_cov2, [p.input_dim])
print("Partitioned means and covariances")
print("mu_1:\n{}\nmu_2:\n{}".format(tr_data_mean1, tr_data_mean2))
print("cov_11:\n{}\ncov_12:\n{}".format(tr_data_cov11, tr_data_cov12))
print("cov_21:\n{}\ncov_22:\n{}".format(tr_data_cov21, tr_data_cov22))

# Size checks
if p.shapecheck:
    assert tr_data_mean1.shape == (p.input_dim,), "Mean component 1 shape incorrect"
    assert tr_data_mean2.shape == (p.output_dim,), "Mean component 2 shape incorrect"
    assert tr_data_cov11.shape == (p.input_dim, p.input_dim), "Covariance component 11 shape incorrect"
    assert tr_data_cov12.shape == (p.input_dim, p.output_dim), "Covariance component 12 shape incorrect"
    assert tr_data_cov21.shape == (p.output_dim, p.input_dim), "Covariance component 21 shape incorrect"
    assert tr_data_cov22.shape == (p.output_dim, p.output_dim), "Covariance component 22 shape incorrect"

# Building the function
fhat_x = lambda x: tr_data_mean2 - np.matmul(np.matmul(tr_data_cov21, inv(tr_data_cov11)), (x - tr_data_mean1))

# Getting the predictions
y_pred = []
for i in range(0, te_data.shape[0]):
    y_pred.append(fhat_x(te_data[i]))
y_pred = np.array(y_pred)

if p.shapecheck:
    assert y_pred.shape == (te_data.shape[0], p.output_dim), "Predicted outputs don't have correct shape"

np.savetxt('predictions.txt', y_pred)
if p.test_is_train:
    print("Mean Squared Loss:\n{}".format(np.power(y_pred - tr_data[:, p.input_dim:], 2).mean(0)))

print("Function coefficients are, assuming that the function form is a1 + a2 * (x - a3)")
print("a1:\n{}".format(tr_data_mean2))
print("a2:\n{}".format(np.matmul(tr_data_cov21, inv(tr_data_cov11))))
print("a3:\n{}".format(tr_data_mean1))
