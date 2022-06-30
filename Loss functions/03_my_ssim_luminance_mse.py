import keras.backend as K
import tensorflow_addons as tfa

def my_ssim_luminance_mse(y_true, y_pred):

  #source: https://www.tensorflow.org/addons/api_docs/python/tfa/image/gaussian_filter2d
  y_true = tfa.image.gaussian_filter2d(y_true, filter_shape=(11,11), sigma=1.5) #input and output is a Tensor
  y_pred = tfa.image.gaussian_filter2d(y_pred, filter_shape=(11,11), sigma=1.5)

  y_true = K.flatten(y_true)
  y_pred = K.flatten(y_pred)

  #convert Tensorflow tensor to numpy
  y_true = K.cast( y_true, 'float64')
  y_pred = K.cast( y_pred, 'float64')

  return K.mean(K.square(y_pred - y_true), axis=-1)
