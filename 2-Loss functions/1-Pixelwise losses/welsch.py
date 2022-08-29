import keras.backend as K

def welsch(y_true, y_pred):

  y_true = K.flatten(y_true)
  y_pred = K.flatten(y_pred)

  #convert Tensorflow tensor to numpy
  y_true = K.cast( y_true, 'float32')
  y_pred = K.cast( y_pred, 'float32')
       
  res = y_true - y_pred 
  C=255.0
  yt = 0.5 * (C**2) * (1-tf.exp(-(res/C)**2)) 
            
  return K.mean(yt)
