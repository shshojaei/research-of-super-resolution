#https://keras.io/api/applications

from tensorflow.keras.applications.resnet50 import ResNet50
import keras.backend as K
from tensorflow.keras.models import Model

#layer_names after activation= [conv1_relu , conv2_block3_2_relu , conv3_block4_2_relu , conv4_block6_2_relu , conv5_block3_2_relu ]
#layer_names before activation= [conv1_conv , conv2_block3_2_conv , conv3_block4_2_conv , conv4_block6_2_conv , conv5_block3_2_conv ]
layer_name = "conv1_relu"

resnet = ResNet50(include_top=False, weights='imagenet', pooling='avg')
resnet.trainable = False
output = resnet.get_layer(layer_name).output
model_resnet = Model(resnet.input, output)

#model_resnet.summary()

#for layer in model_resnet.layers:
#  print(layer.name)

@tf.function
def my_resnet(y_true , y_pred):

  #convert Tensorflow tensor to numpy
  y_true = K.cast( y_true, 'float32')
  y_pred = K.cast( y_pred, 'float32')

  y_true = model_resnet(y_true)
  y_pred = model_resnet(y_pred)

  return K.mean(K.square(y_pred - y_true), axis=-1)
