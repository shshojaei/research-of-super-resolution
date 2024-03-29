import keras.backend as K

def charbonnier(y_true, y_pred):

  #y_true = K.flatten(y_true)
  #y_pred = K.flatten(y_pred)

  #convert Tensorflow tensor to numpy
  y_true = K.cast( y_true, 'float32')
  y_pred = K.cast( y_pred, 'float32')
  
  epsilon = 1e-3
  res = y_true - y_pred
  yt = K.sqrt((res**2) + (epsilon**2))
        
  return K.mean(yt)
