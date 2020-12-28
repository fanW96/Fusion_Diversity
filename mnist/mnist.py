import tensorflow.keras.datasets.mnist as mnist
import tensorflow.keras.utils as utils
import numpy as np

def get_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)
    x_train = x_train.astype('float32')/255.
    x_test = x_test.astype('float32')/255.
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean
    y_train = utils.to_categorical(y_train, 10).astype('float32')
    y_test = utils.to_categorical(y_test, 10).astype('float32')
    index = [i for i in range(len(x_train))]
    np.random.shuffle(index)
    x_train = x_train[index]
    y_train = y_train[index]

    return x_train, y_train, x_test, y_test