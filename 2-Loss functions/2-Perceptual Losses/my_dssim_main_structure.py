!pip install tensorflow-addons

import keras.backend as K
import tensorflow_addons as tfa

def my_dssim_main_structure(y_true, y_pred):
  #source: https://www.tensorflow.org/addons/api_docs/python/tfa/image/gaussian_filter2d
  #Means obtained by Gaussian filtering of inputs => source: https://www.programcreek.com/python/?CodeExample=compute+ssim
  #Sigma is a standard deviation of the gaussian

  y_true_mean = tfa.image.gaussian_filter2d(y_true, filter_shape=(11,11), sigma=1.5) #input and output is a Tensor
  y_pred_mean = tfa.image.gaussian_filter2d(y_pred, filter_shape=(11,11), sigma=1.5)

  #convert Tensorflow tensor to numpy
  y_true_mean = K.cast( y_true_mean, 'float32')
  y_pred_mean = K.cast( y_pred_mean, 'float32')

  y_true = K.cast( y_true, 'float32')
  y_pred = K.cast( y_pred, 'float32')

  #source: https://github.com/mubeta06/python/blob/master/signal_processing/sp/ssim.py
  # k1 & c1 depend on L (width of color map)
  l = 255
  k_1 = 0.01
  k_2 = 0.03
  c_1 = (k_1 * l)**2
  c_2 = (k_2 * l)**2
  c_3 = c_2/2

  # Squares of means
  mu_1_sq = y_true_mean**2
  mu_2_sq = y_pred_mean**2
  mu_1_mu_2 = y_true_mean * y_pred_mean

  sigma1_sq = (y_true - y_true_mean)**2
  sigma1_sq_mean = (tfa.image.mean_filter2d((y_true - y_true_mean)**2))**1/2
  sigma2_sq = (y_pred - y_pred_mean)**2 
  sigma2_sq_mean = (tfa.image.mean_filter2d((y_pred - y_pred_mean)**2))**1/2
  sigma12 = tfa.image.mean_filter2d((y_true - y_true_mean) * (y_pred - y_pred_mean))

  ssim_map = (sigma12 + c_3) / ((sigma1_sq_mean * sigma2_sq_mean) + c_3)
 
  # return MSSIM
  index = K.mean(1-ssim_map)

  return index
