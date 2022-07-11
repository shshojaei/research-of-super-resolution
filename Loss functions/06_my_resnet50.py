from tensorflow.keras.models import Model
import keras.backend as K

layer_name = "pool1_pool"

resnet = ResNet50(include_top=False, weights=None, pooling='avg')
resnet.trainable = False
output = resnet.get_layer(layer_name).output
model_resnet = Model(resnet.input, output)

@tf.function
def perceptual_resnet_loss(y_true , y_pred):
    y_true = model_resnet(y_true)
    y_pred = model_resnet(y_pred)

    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)

    #convert Tensorflow tensor to numpy
    #y_true = K.cast( y_true, 'float64')
    #y_pred = K.cast( y_pred, 'float64')

    return K.mean(K.square(y_pred - y_true), axis=-1)
