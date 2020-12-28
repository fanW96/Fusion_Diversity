import keras
from keras.layers import AveragePooling2D, Input, Flatten
from keras.models import Model, load_model
from keras.datasets import mnist
import tensorflow as tf
import cleverhans.attacks as attacks
from cleverhans.utils_tf import model_eval
import os
from keras_wraper import KerasModelWrapper_fusion
from cleverhans.utils_keras import KerasModelWrapper
import numpy as np


def attack_optimization(sess, isbaseline, model, dataset, num_classes, attack_method, iconst):
    # Subtracting pixel mean improves accuracy
    subtract_pixel_mean = False

    # Load the data.
    if dataset == 'mnist':
        (x_train, y_train), (x_test, y_test_index) = mnist.load_data()

    # input_shape = x_train.shape[1:]

    # Normalize data.
    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # If subtract pixel mean is enabled
    clip_min = 0.0
    clip_max = 1.0
    if subtract_pixel_mean:
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean
        clip_min -= x_train_mean
        clip_max -= x_train_mean

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test_index, num_classes)


    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, num_classes))
    # y_target = tf.placeholder(tf.float32, shape=(None, num_classes))

    if isbaseline:
        wrap_model = KerasModelWrapper(model)
    else:
        wrap_model = KerasModelWrapper_fusion(model)

    # Initialize the attack method
    if attack_method == 'CarliniWagnerL2':
        num_samples = 500
        eval_par = {'batch_size': 100}
        att = attacks.CarliniWagnerL2(wrap_model, sess=sess)
        att_params = {
            'batch_size': 100,
            'confidence': 0.1,
            'learning_rate': 0.01,
            'binary_search_steps': 1,
            'max_iterations': 1000,
            'initial_const': iconst,
            'clip_min': clip_min,
            'clip_max': clip_max
        }
        adv_x = att.generate(x, **att_params)
    elif attack_method == 'ElasticNetMethod':
        num_samples = 2000
        eval_par = {'batch_size': 100}
        att = attacks.ElasticNetMethod(wrap_model, sess=sess)
        att_params = {
            'batch_size': 100,
            'confidence': 1,
            'learning_rate': 0.01,
            'binary_search_steps': 1,
            'max_iterations': 1000,
            'initial_const': iconst,
            'beta': 1e-2,
            'fista': True,
            'decision_rule': 'EN',
            'clip_min': clip_min,
            'clip_max': clip_max
        }
        adv_x = att.generate(x, **att_params)

    preds = wrap_model.get_probs(x)
    preds_adv = wrap_model.get_probs(adv_x)
    acc = model_eval(sess, x, y, preds, x_test[:num_samples], y_test[:num_samples], args=eval_par)
    acc_adv = model_eval(sess, x, y, preds_adv, x_test[:num_samples], y_test[:num_samples], args=eval_par)
    return acc, acc_adv