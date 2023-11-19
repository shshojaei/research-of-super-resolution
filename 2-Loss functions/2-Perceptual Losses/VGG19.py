#source: https://stackoverflow.com/questions/65484420/define-custom-loss-perceptual-loss-in-cnn-autoencoder-with-pre-train-vgg19-ten
#https://keras.io/api/applications

from tensorflow.keras.models import Model
from keras.applications.vgg19 import VGG19
import keras.backend as K

#layer_name = ["block1_conv1","block1_conv2",
#              "block2_conv1","block2_conv2",
#              "block3_conv1","block3_conv2","block3_conv3","block3_conv4",
#              "block4_conv1","block4_conv2","block4_conv3","block4_conv4",
#              "block5_conv1","block5_conv2","block5_conv3","block5_conv4"]

layer_name = "block5_conv4"

vgg = VGG19(weights='imagenet', include_top=False)
vgg.trainable = False
output = vgg.get_layer(layer_name).output
model_vgg = Model(vgg.input, output)

@tf.function
def my_vgg(y_true , y_pred):
    y_true = model_vgg(y_true)
    y_pred = model_vgg(y_pred)

    #y_true = K.flatten(y_true)
    #y_pred = K.flatten(y_pred)

    return K.mean(K.square(y_pred - y_true), axis=-1)
