import gzip
from six.moves import cPickle as pickle
import os
import platform
import numpy as np

# load pickle based on python version 2 or 3
def load_pickle(f):
    version = platform.python_version_tuple()
    if version[0] == '2':
        return pickle.load(f)
    elif version[0] == '3':
        return pickle.load(f, encoding='latin1')
    raise ValueError("invalid python version: {}".format(version))


def load_mnist_datasets(path='data/mnist.pkl.gz'):
    if not os.path.exists(path):
        raise Exception('Cannot find %s' % path)
    with gzip.open(path, 'rb') as f:
        train_set, val_set, test_set = load_pickle(f)
        return train_set, val_set, test_set

def load_monkey_datasets(path='monkey_dataset/'):
    path_train = os.path.join(path, 'trainsh_data.pkl')
    path_test = os.path.join(path, 'testsh_data.pkl')
    path_val = os.path.join(path, 'valsh_data.pkl')
    # val train_set, val_set, test_set
    if not os.path.exists(path_train):
        raise Exception('Cannot find %s' % path_train)
    with open(path_train, 'rb') as f:
        train_dict = load_pickle(f)
        train_set = [train_dict['data'],train_dict['labels']]
    with open(path_test, 'rb') as f:
        test_dict = load_pickle(f)
        test_set = [test_dict['data'],test_dict['labels']]
    with open(path_val, 'rb') as f:
        val_dict = load_pickle(f)
        val_set = [val_dict['data'],val_dict['labels']]
    return train_set, val_set, test_set