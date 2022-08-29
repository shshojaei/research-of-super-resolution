import keras.backend as K

def logistic(y_true, y_pred):

  y_true = K.flatten(y_true)
  y_pred = K.flatten(y_pred)

  #convert Tensorflow tensor to numpy
  y_true = K.cast( y_true, 'float32')
  y_pred = K.cast( y_pred, 'float32')
    
  res = y_true - y_pred 
  C=255.0
  yt = (C**2) * K.log(tf.cosh(res/C))
 
  return K.mean(yt) 
