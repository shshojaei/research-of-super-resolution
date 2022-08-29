Xception().summary()
#####################

#https://keras.io/api/applications

from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Model
import keras.backend as K

#layer_names = [block3_sepconv2, block6_sepconv3, block9_sepconv3, block12_sepconv3] #block3,6,9,12
layer_name = "block12_sepconv3"

xception = Xception(include_top=False, weights='imagenet', pooling='avg')
xception.trainable = False
output = xception.get_layer(layer_name).output
model_xception = Model(xception.input, output)

@tf.function
def my_Xception(y_true , y_pred):
    y_true = model_xception(y_true)
    y_pred = model_xception(y_pred)

    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    return K.mean(K.square(y_pred - y_true), axis=-1)
