import keras
from keras.layers import AveragePooling2D, Input, Flatten
from keras.models import Model, load_model
import cifar10
import tensorflow as tf
import cleverhans.attacks as attacks
from cleverhans.utils_tf import model_eval
import os
from resnet_keras_feature_fusion_v2 import ResNet_v2
from resnet_keras_feature_fusion import ResNet
from keras_wraper import KerasModelWrapper_fusion
from cleverhans.utils_keras import KerasModelWrapper
import numpy as np
from jsma import jsma_impl_loop


def attack_jsma(sess, isbaseline, model, dataset, num_classes, attack_method, theta_, gamma_):
    # Subtracting pixel mean improves accuracy
    subtract_pixel_mean = False

    # Load the data.
    if dataset == 'cifar10':
        x_train, y_train, x_test, y_test = cifar10.get_data()
    # elif dataset == 'cifar100':
    #     (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')


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
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))
    y = tf.placeholder(tf.float32, shape=(None, num_classes))
    y_adv = tf.placeholder(tf.float32, shape=(None, num_classes))

    if isbaseline:
        wrap_model = KerasModelWrapper(model)
    else:
        wrap_model = KerasModelWrapper_fusion(model)
    
    eval_par = {'batch_size': 1}

    num_samples = 4000
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

    # adv_data = np.zeros(x_test.shape)
    # for i in range(num_samples):
    #     tmp = jsma_impl_loop(sess, x_test[i:i+1], y_test[i:i+1], wrap_model, x, y, gamma=gamma_, eps=theta_, clip_min=clip_min, clip_max=clip_max, increase=False)
    #     adv_data[i] = tmp[0]

    acc_adv = model_eval(sess, x, y, preds_adv, x_test[:num_samples], y_test[:num_samples], args=eval_par)
    # acc_adv = model_eval(sess, x, y, preds, adv_data[:num_samples], y_test[:num_samples], args=eval_par)
    return acc, acc_adv

