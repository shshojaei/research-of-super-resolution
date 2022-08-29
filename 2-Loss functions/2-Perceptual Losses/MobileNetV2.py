MobileNetV2().summary()
########################

#https://keras.io/api/applications

from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
import keras.backend as K

#layer_names = [block_4_add, block_8_add, block_12_add, block_16_project_BN ] #block4,8,12,16
layer_name = "block_4_add"

mobilenet = MobileNetV2(include_top=False, weights='imagenet', pooling='avg')
mobilenet.trainable = False
output = mobilenet.get_layer(layer_name).output
model_mobilenet = Model(mobilenet.input, output)

@tf.function
def my_mobilenet(y_true , y_pred):
    y_true = model_mobilenet(y_true)
    y_pred = model_mobilenet(y_pred)

    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    return K.mean(K.square(y_pred - y_true), axis=-1)
