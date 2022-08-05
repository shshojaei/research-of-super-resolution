import keras.backend as K

def my_mae(y_true, y_pred):
  
  y_true = K.flatten(y_true)
  y_pred = K.flatten(y_pred)

  #convert Tensorflow tensor to numpy
  y_true = K.cast( y_true, 'float32')
  y_pred = K.cast( y_pred, 'float32')

  return K.mean(K.abs(y_pred - y_true), axis=-1)
