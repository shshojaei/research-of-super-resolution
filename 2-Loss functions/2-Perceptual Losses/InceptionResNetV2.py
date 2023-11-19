from tensorflow.keras.applications import InceptionResNetV2
import keras.backend as K
from tensorflow.keras.models import Model

#layer_names = [block35_1_conv, block17_5, block8_10_conv] #
layer_name = "block8_10_conv"

inceptionResnet = InceptionResNetV2(include_top=False, weights='imagenet', pooling='avg')

inceptionResnet.trainable = False
output = inceptionResnet.get_layer(layer_name).output
model_inceptionResnet = Model(inceptionResnet.input, output)

model_inceptionResnet.summary()

#for layer in model_inceptionResnet.layers:
#  print(layer.name)

@tf.function
def my_inceptionResnet(y_true , y_pred):

  #convert Tensorflow tensor to numpy
  y_true = K.cast( y_true, 'float32')
  y_pred = K.cast( y_pred, 'float32')

  #Note: each Keras Application expects a specific kind of input preprocessing.
  #For InceptionV3, call tf.keras.applications.inception_v3.preprocess_input on your inputs before passing them to the model.
  #inception_v3.preprocess_input will scale input pixels between -1 and 1.
  y_true = tf.keras.applications.inception_resnet_v2.preprocess_input(y_true)
  y_pred = tf.keras.applications.inception_resnet_v2.preprocess_input(y_pred)

  y_true = model_inceptionResnet(y_true)
  y_pred = model_inceptionResnet(y_pred)

  return K.mean(K.square(y_pred - y_true), axis=-1)
