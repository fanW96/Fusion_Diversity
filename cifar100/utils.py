import numpy as np
import tensorflow as tf
import keras as keras

def eval_acc(model, x, y):
  predictions = model.predict(x)
  return 100*np.sum(np.argmax(predictions, 1) == np.argmax(y, 1))/y.shape[0]