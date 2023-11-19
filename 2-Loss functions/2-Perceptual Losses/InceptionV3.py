from tensorflow.keras.applications import InceptionV3
import keras.backend as K
from tensorflow.keras.models import Model

#layer_names = [ mixed3, mixed6, mixed10]
layer_name = "mixed10"

inceptionV3 = InceptionV3(include_top=False, weights='imagenet', pooling='avg')
inceptionV3.trainable = False
output = inceptionV3.get_layer(layer_name).output
model_inceptionV3 = Model(inceptionV3.input, output)

#model_inceptionV3.summary()

#for layer in model_inceptionV3.layers:
#  print(layer.name)

@tf.function
def my_inceptionV3(y_true , y_pred):

  y_true = K.cast( y_true, 'float32')
  y_pred = K.cast( y_pred, 'float32')

  #Note: each Keras Application expects a specific kind of input preprocessing.
  #For InceptionV3, call tf.keras.applications.inception_v3.preprocess_input on your inputs before passing them to the model.
  #inception_v3.preprocess_input will scale input pixels between -1 and 1.
  y_true = tf.keras.applications.inception_v3.preprocess_input(y_true)
  y_pred = tf.keras.applications.inception_v3.preprocess_input(y_pred)

  y_true = model_inceptionV3(y_true)
  y_pred = model_inceptionV3(y_pred)

  return K.mean(K.square(y_pred - y_true), axis=-1)
