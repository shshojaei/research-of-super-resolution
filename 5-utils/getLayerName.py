inceptionResnet = InceptionResNetV2(include_top=False, weights=None, pooling='avg')

for layer in inceptionResnet.layers:
  print(layer.name)

###############################################################################
from tensorflow.keras.applications.resnet50 import ResNet50
import keras.backend as K
from tensorflow.keras.models import Model

resnet = ResNet50(include_top=False, weights='imagenet', pooling='avg')
resnet.trainable = False
layer_names = [layer.name for layer in resnet.layers]
print(layer_names)
