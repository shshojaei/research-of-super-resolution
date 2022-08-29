ResNet152V2().summary()
########################

#https://keras.io/api/applications

from tensorflow.keras.applications import ResNet152V2
from tensorflow.keras.models import Model
import keras.backend as K

#layer_names = [conv1_conv, conv2_block3_out, conv3_block8_out, conv4_block36_out,conv5_block3_out]
layer_name = "conv5_block3_out"

resnet = ResNet152V2(include_top=False, weights='imagenet', pooling='avg')
resnet.trainable = False
output = resnet.get_layer(layer_name).output
model_resnet = Model(resnet.input, output)

@tf.function
def my_resnet152(y_true , y_pred):
    y_true = model_resnet(y_true)
    y_pred = model_resnet(y_pred)

    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    return K.mean(K.square(y_pred - y_true), axis=-1)
