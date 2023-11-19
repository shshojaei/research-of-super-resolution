!pip install tensorflow_io

import tensorflow_io as tfio
import keras.backend as K

def sobel_mae(y_true, y_pred):

  #convert Tensorflow tensor to numpy
  y_true = K.cast( y_true, 'float32')
  y_pred = K.cast( y_pred, 'float32')

  y_true = tf.image.rgb_to_grayscale(y_true)
  y_pred = tf.image.rgb_to_grayscale(y_pred)

  #source: https://www.tensorflow.org/io/api_docs/python/tfio/experimental/filter/
  y_true = tfio.experimental.filter.sobel(y_true) #input and output is a Tensor
  y_pred = tfio.experimental.filter.sobel(y_pred)

  return K.mean(K.abs(y_pred - y_true), axis=-1)
