import keras.backend as K

def talwar(y_true, y_pred):

  #y_true = K.flatten(y_true)
  #y_pred = K.flatten(y_pred)

  #convert Tensorflow tensor to numpy
  y_true = K.cast( y_true, 'float32')
  y_pred = K.cast( y_pred, 'float32')

  res = y_true - y_pred 
  absRes = K.abs(res) 
  C=255.0
        
  linearRegion = tf.greater_equal(absRes, C)
  yt =  tf.where (linearRegion, (C**2)*tf.ones_like(absRes), absRes**2) 
            
  return K.mean(yt)
