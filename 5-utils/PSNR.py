#https://www.tensorflow.org/api_docs/python/tf/image/psnr

def PSNR1(y_pred, y_true):
  #Compute the peak signal-to-noise ratio, measures quality of image.
  # Max value of pixel is 255
  psnr_value = tf.image.psnr(y_true, y_pred, max_val=255)
  return psnr_value

##############################

#tensorflow psnr github: https://github.com/tensorflow/tensorflow/blob/v2.9.1/tensorflow/python/ops/image_ops_impl.py#L4129-L4181

import keras.backend as K

def PSNR2(y_pred, y_true):

  y_true = K.flatten(y_true)
  y_pred = K.flatten(y_pred)

  #convert Tensorflow tensor to numpy
  y_true = K.cast( y_true, 'float32')
  y_pred = K.cast( y_pred, 'float32')

  mse = K.mean(K.square(y_pred - y_true), axis=-1)

  max_val = 255.0

  psnr_val = (20.0 * K.log(max_val) / K.log(10.0)) - ((10.0 / K.log(10.0)) * K.log(mse)) 

  return psnr_val

##############################

import keras.backend as K

def PSNR3(y_pred, y_true):
  y_true = K.flatten(y_true)
  y_pred = K.flatten(y_pred)

  #convert Tensorflow tensor to numpy
  y_true = K.cast( y_true, 'float32')
  y_pred = K.cast( y_pred, 'float32')

  mse = K.mean(K.square(y_pred - y_true), axis=-1)
  max_pixel = 255.0

  psnr_value = (10.0 * K.log(max_pixel **2 / mse)) / 2.303
  
  return psnr_value

#print(K.log(2.303))
##############################

import keras.backend as K

def PSNR4(y_pred, y_true):
  y_true = K.flatten(y_true)
  y_pred = K.flatten(y_pred)

  #convert Tensorflow tensor to numpy
  y_true = K.cast( y_true, 'float32')
  y_pred = K.cast( y_pred, 'float32')

  #https://stackoverflow.com/questions/55844618/how-to-calculate-psnr-metric-in-keras
  #divide 2.303 Because natural log is bigger than base-10 log.
  #sometimes psnr can be inf, maybe y_pred == y_true. if you don't want to see it, just square (y_pred - y_true + 1e-8)

  mse = K.mean(K.square(y_pred - y_true + 1e-8), axis=-1)
  max_pixel = 255.0

  psnr_value = (10.0 * K.log(max_pixel **2 / mse)) / 2.303
  
  return psnr_value
