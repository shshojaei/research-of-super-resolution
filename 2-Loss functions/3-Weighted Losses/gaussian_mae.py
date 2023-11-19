!pip install tensorflow-addons

import tensorflow_addons as tfa
import keras.backend as K

def gaussian_mae(y_true, y_pred):

  #convert Tensorflow tensor to numpy
  y_true = K.cast( y_true, 'float32')
  y_pred = K.cast( y_pred, 'float32')

  #source: https://www.tensorflow.org/addons/api_docs/python/tfa/image/gaussian_filter2d
  y_true = tfa.image.gaussian_filter2d(y_true, filter_shape=(11,11), sigma=1.5) #input and output is a Tensor
  y_pred = tfa.image.gaussian_filter2d(y_pred, filter_shape=(11,11), sigma=1.5)

  return K.mean(K.abs(y_pred - y_true), axis=-1)
