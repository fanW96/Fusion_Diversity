import tensorflow.keras.datasets.cifar100 as cifar100
import tensorflow.keras.utils as utils
import numpy as np

def get_data():
    (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
    x_train = x_train.astype('float32')/255.
    x_test = x_test.astype('float32')/255.
    y_train = utils.to_categorical(y_train, 100).astype('float32')
    y_test = utils.to_categorical(y_test, 100).astype('float32')
    index = [i for i in range(len(x_train))]
    np.random.shuffle(index)
    x_train = x_train[index]
    y_train = y_train[index]

    return x_train, y_train, x_test, y_test