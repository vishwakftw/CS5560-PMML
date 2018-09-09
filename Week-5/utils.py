import numpy as np

def _process_str(val_string):
    """
    Returns a dict corresponding to a line in a dataset in the LIBSVM format
    For example:
        1:0.25 2:0.5 3:0.75 --> {0:0.25, 1:0.5, 2:0.75}
    """
    index_dict = dict()
    for keyvalpair in val_string.split(','):
        key_val = keyvalpair.split(':')
        print(keyvalpair)
        if int(key_val[0]) == 0:
            print("bomb")
        index_dict[int(key_val[0]) - 1] = float(key_val[1])
    return index_dict

def get_data_from_libsvm_format(path_to_dataset, n_attr):
    """
    Get a NumPy dense version of a dataset in the LIBSVM format
    """
    outputs = []
    inputs = []
    with open(path_to_dataset) as f:
        for line in f:
            tmp = line.split(' ')
            tmp = tmp[:-1]
            outputs.append(int(tmp[0]))
            input_dict = _process_str(",".join(tmp[1:]))
            inputs.append(input_dict)

    input_data = np.zeros((len(outputs), n_attr))
    for idx, inp in enumerate(inputs):
        for keyval in inp.items():
            input_data[idx, keyval[0]] = keyval[1]
    return np.hstack([input_data, np.array(outputs).reshape(-1, 1)])

def get_data_in_libsvm_format(path_to_dataset):
    """
    Get the dataset in LIBSVM format. Saved to a file with suffix `LIBSVM`
    """
    data = np.genfromtxt(path_to_dataset, delimiter=',')
    x, y = np.hsplit(data, [data.shape[1] - 1])
    y = y.reshape(-1)
    del data
    with open(path_to_dataset[:-4] + '_LIBSVM' + path_to_dataset[-4:], 'w') as f:
        for i in range(len(y)):
            all_str = []
            all_str.append(str(int(y[i])))
            for j in range(len(x[i])):
                if x[i, j] != 0:
                    all_str.append('{}:{}'.format(j + 1, x[i, j]))
            f.write(' '.join(all_str) + '\n')
