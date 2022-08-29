!pip install tensorflow_io

import keras.backend as K
import tensorflow_io as tfio

def my_mse_prewitt(y_true, y_pred):

  #convert Tensorflow tensor to numpy
  y_true = K.cast( y_true, 'float32')
  y_pred = K.cast( y_pred, 'float32')

  #source: https://www.tensorflow.org/io/api_docs/python/tfio/experimental/filter/
  y_true = tfio.experimental.filter.prewitt(y_true) #input and output is a Tensor
  y_pred = tfio.experimental.filter.prewitt(y_pred)

  y_true = K.flatten(y_true)
  y_pred = K.flatten(y_pred)

  return K.mean(K.square(y_pred - y_true), axis=-1)
