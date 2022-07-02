#source: https://stackoverflow.com/questions/65484420/define-custom-loss-perceptual-loss-in-cnn-autoencoder-with-pre-train-vgg19-ten

from tensorflow.keras.models import Model
import keras.backend as K

#layer_name = ["block1_conv1","block1_conv2",
#              "block2_conv1","block2_conv2",
#              "block3_conv1","block3_conv2","block3_conv3","block3_conv4",
#              "block4_conv1","block4_conv2","block4_conv3","block4_conv4",
#              "block5_conv1","block5_conv2","block5_conv3","block5_conv4"]

layer_name = "block4_conv1"

vgg = VGG19(weights='imagenet', include_top=False)
vgg.trainable = False
output = vgg.get_layer(layer_name).output
model_vgg = Model(vgg.input, output)

@tf.function
def perceptual_vgg_loss(y_true , y_pred):
    y_true = model_vgg(y_true)
    y_pred = model_vgg(y_pred)

    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    #convert Tensorflow tensor to numpy
    #y_true = K.cast( y_true, 'float64')
    #y_pred = K.cast( y_pred, 'float64')

    return K.mean(K.square(y_pred - y_true), axis=-1)
