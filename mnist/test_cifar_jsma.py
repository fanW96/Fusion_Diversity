import keras
from keras.layers import AveragePooling2D, Input, Flatten
from keras.models import Model, load_model
import mnist
import tensorflow as tf
import cleverhans.attacks as attacks
from cleverhans.utils_tf import model_eval
import os
from keras_wraper import KerasModelWrapper_fusion
from cleverhans.utils_keras import KerasModelWrapper
import numpy as np
from jsma import jsma_impl_loop
import time

def attack_jsma(sess, isbaseline, model, dataset, num_classes, attack_method, theta_, gamma_):
    # Subtracting pixel mean improves accuracy
    subtract_pixel_mean = False

    # Load the data.
    if dataset == 'mnist':
        x_train, y_train, x_test, y_test = mnist.get_data()


    # input_shape = x_train.shape[1:]

    # If subtract pixel mean is enabled
    clip_min = 0.0
    clip_max = 1.0
    if subtract_pixel_mean:
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean
        clip_min -= x_train_mean
        clip_max -= x_train_mean


    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
    y = tf.placeholder(tf.float32, shape=(None, num_classes))
    y_adv = tf.placeholder(tf.float32, shape=(None, num_classes))

    if isbaseline:
        wrap_model = KerasModelWrapper(model)
    else:
        wrap_model = KerasModelWrapper_fusion(model)
    
    eval_par = {'batch_size': 1}

    num_samples = 500
    print('num_samples:', num_samples)
    att = attacks.SaliencyMapMethod(wrap_model, sess=sess)
    att_params = {
        'gamma': gamma_,
        'theta': theta_,
        'symbolic_impl': False,
        'clip_min': clip_min,
        'clip_max': clip_max
    }
    adv_x = att.generate(x, **att_params)
    preds = wrap_model.get_probs(x)
    preds_adv = wrap_model.get_probs(adv_x)
    acc = model_eval(sess, x, y, preds, x_test[:num_samples], y_test[:num_samples], args=eval_par)

    starttime = time.time()
    # adv_data = np.zeros(x_test.shape)
    # for i in range(num_samples):
    #     tmp = jsma_impl_loop(sess, x_test[i:i+1], y_test[i:i+1], wrap_model, x, y, gamma=gamma_, eps=theta_, clip_min=clip_min, clip_max=clip_max, increase=False)
    #     adv_data[i] = tmp[0]
    #     print("\r{}/100".format(i), end="")
    # print("\n", time.time()-starttime)

    acc_adv = model_eval(sess, x, y, preds_adv, x_test[:num_samples], y_test[:num_samples], args=eval_par)
    # acc_adv = model_eval(sess, x, y, preds, adv_data[:num_samples], y_test[:num_samples], args=eval_par)
    print(time.time()-starttime)

    return acc, acc_adv

