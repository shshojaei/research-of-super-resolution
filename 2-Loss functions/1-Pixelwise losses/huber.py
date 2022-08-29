import keras.backend as K

def huber(y_true, y_pred):

  y_true = K.flatten(y_true)
  y_pred = K.flatten(y_pred)

  #convert Tensorflow tensor to numpy
  y_true = K.cast( y_true, 'float32')
  y_pred = K.cast( y_pred, 'float32')

  C=255.0
  res = y_true - y_pred
  cond  = K.abs(res) < C
    
  squared_loss = 0.5 * (res**2)
  linear_loss  = C * (K.abs(res) - 0.5 * C)
    
  return K.mean(tf.where(cond, squared_loss, linear_loss))
