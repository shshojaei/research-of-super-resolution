!pip install tensorflow-addons

import keras.backend as K
import tensorflow_addons as tfa
import math

def my_dssim_main_luminance(y_true, y_pred):
  #source: https://www.tensorflow.org/addons/api_docs/python/tfa/image/gaussian_filter2d
  #Means obtained by Gaussian filtering of inputs => source: https://www.programcreek.com/python/?CodeExample=compute+ssim
  #Sigma is a standard deviation of the gaussian

  #source for set sigma: http://www.adeveloperdiary.com/data-science/computer-vision/applying-gaussian-smoothing-to-an-image-using-python-from-scratch/
  kernel_size = 13
  sigma_value = math.sqrt(kernel_size)
  #y_true = tfa.image.gaussian_filter2d(y_true, filter_shape=(11,11), sigma=1.5) #input and output is a Tensor
  #y_pred = tfa.image.gaussian_filter2d(y_pred, filter_shape=(11,11), sigma=1.5)

  y_true = tfa.image.gaussian_filter2d(y_true, filter_shape=(13,13), sigma=sigma_value) #input and output is a Tensor
  y_pred = tfa.image.gaussian_filter2d(y_pred, filter_shape=(13,13), sigma=sigma_value)

  y_true = K.flatten(y_true)
  y_pred = K.flatten(y_pred)

  #convert Tensorflow tensor to numpy
  y_true = K.cast( y_true, 'float32')
  y_pred = K.cast( y_pred, 'float32')

  # k1 & c1 depend on L (width of color map)
  l = 255
  k_1 = 0.01
  c_1 = (k_1 * l)**2

  # Squares of means
  mu_1_sq = y_true**2
  mu_2_sq = y_pred**2
  mu_1_mu_2 = y_true * y_pred

  ssim_map = (2 * mu_1_mu_2 + c_1) / (mu_1_sq + mu_2_sq + c_1)

  # return MSSIM
  index = K.mean(1-ssim_map)

  return index 
