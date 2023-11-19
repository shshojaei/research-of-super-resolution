import keras.backend as K

def HR_mean_mae(y_true, y_pred):

  #convert Tensorflow tensor to numpy
  y_true = K.cast( y_true, 'float32')
  y_pred = K.cast( y_pred, 'float32')

  # Apply average pooling with factor 2
  pooling_result = tf.keras.layers.AveragePooling2D(pool_size=(2, 2))(y_true)

  # Upsample the result with factor 2
  upsampling_result = tf.keras.layers.UpSampling2D(size=(2, 2))(pooling_result)

  y_true_details = K.abs(y_true - upsampling_result)

  result = y_true_details * K.abs(y_pred - y_true)

  return K.mean(result, axis=-1)
