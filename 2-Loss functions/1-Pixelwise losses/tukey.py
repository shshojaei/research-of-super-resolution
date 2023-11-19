import keras.backend as K

def tukey(y_true, y_pred):

  #y_true = K.flatten(y_true)
  #y_pred = K.flatten(y_pred)

  #convert Tensorflow tensor to numpy
  y_true = K.cast( y_true, 'float32')
  y_pred = K.cast( y_pred, 'float32')
        
  res = y_true - y_pred 
        
  C=255.0
  scale = (C**2) / 6 ;
  yt = scale * (1 - (1 - (res / C)**2)**3) ;
        
  linearRegion = tf.greater(tf.abs(res), C)
  yt =  tf.where(linearRegion, scale*tf.ones_like(yt), yt) 
    
  return K.mean(yt)
