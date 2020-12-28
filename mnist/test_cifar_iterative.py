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



def attack_iterative(sess, model, dataset, num_classes, attack_method, eps_, nb_iter_):
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

    wrap_model = KerasModelWrapper_fusion(model)

    # Initialize the attack method
    if attack_method == 'MadryEtAl':
        att = attacks.MadryEtAl(wrap_model)
    elif attack_method == 'FastGradientMethod':
        att = attacks.FastGradientMethod(wrap_model)
    elif attack_method == 'MomentumIterativeMethod':
        att = attacks.MomentumIterativeMethod(wrap_model)
    elif attack_method == 'BasicIterativeMethod':
        att = attacks.BasicIterativeMethod(wrap_model)

    # consider the attack to be constant
    eval_par = {'batch_size': 128}
    if attack_method == 'FastGradientMethod':
        att_params = {'eps': eps_,
                    'clip_min': clip_min,
                    'clip_max': clip_max}
    else:
        att_params = {'eps': eps_,
                    'eps_iter': eps_*1.0/nb_iter_,
                    'clip_min': clip_min,
                    'clip_max': clip_max,
                    'nb_iter': nb_iter_}

    adv_x = tf.stop_gradient(att.generate(x, **att_params))
    preds = model(adv_x)
    acc = model_eval(sess, x, y, preds, x_test, y_test, args=eval_par)
    return acc
    