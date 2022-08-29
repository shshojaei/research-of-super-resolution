import keras.backend as K

def geman(y_true, y_pred):

  y_true = K.flatten(y_true)
  y_pred = K.flatten(y_pred)

  #convert Tensorflow tensor to numpy
  y_true = K.cast( y_true, 'float32')
  y_pred = K.cast( y_pred, 'float32')

  C=255.0
  res = y_true - y_pred
  yt = ((res**2) /2)/((C**2) +  (res**2))
       
  return K.mean(yt)
