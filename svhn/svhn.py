import tensorflow.keras.utils as utils
import numpy as np
from scipy.io import loadmat

def load_data(path):
    """ Helper function for loading a MAT-File"""
    data = loadmat(path)
    return data['X'], data['y']

def get_data():
    x_train, y_train = load_data('./SVHN/train_32x32.mat')
    x_test, y_test = load_data('./SVHN/test_32x32.mat')
    x_train, y_train = x_train.transpose((3,0,1,2)), y_train[:,0]
    x_test, y_test = x_test.transpose((3,0,1,2)), y_test[:,0]
    y_train[y_train == 10] = 0
    y_test[y_test == 10] = 0
    x_train = x_train.astype('float32')/255.
    x_test = x_test.astype('float32')/255.
#     x_train_mean = np.mean(x_train, axis=0)
#     x_train -= x_train_mean
#     x_test -= x_train_mean
    y_train = utils.to_categorical(y_train, 10).astype('float32')
    y_test = utils.to_categorical(y_test, 10).astype('float32')
    index = [i for i in range(len(x_train))]
    np.random.shuffle(index)
    x_train = x_train[index]
    y_train = y_train[index]

    return x_train, y_train, x_test, y_test