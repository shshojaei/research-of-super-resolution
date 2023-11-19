import keras.backend as K
from tensorflow.keras.applications.resnet50 import ResNet50

# Load ResNet50
base_model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
base_model.trainable = False

#layer_names after activation= [conv1_relu , conv2_block3_2_relu , conv3_block4_2_relu , conv4_block6_2_relu , conv5_block3_2_relu ]
#layer_names before activation= [conv1_conv , conv2_block3_2_conv , conv3_block4_2_conv , conv4_block6_2_conv , conv5_block3_2_conv ]
# Select layer
layer_name = 'conv5_block3_2_relu'

# Loop through layers up to selected layer
# Append layer outputs to list
layer_outputs = []
for layer in base_model.layers:
  layer_outputs.append(layer.output)
  if layer.name == layer_name:
    break

feature_model = tf.keras.Model(inputs=base_model.input, outputs=layer_outputs)

@tf.function
def resnet50_mean_loss(y_true, y_pred):

  y_true_features = feature_model(y_true)
  y_pred_features = feature_model(y_pred)

  layer_losses = []
  for true_feat, pred_feat in zip(y_true_features, y_pred_features):
     loss = K.mean(K.abs(true_feat - pred_feat))
     layer_losses.append(loss)

  total_loss = K.mean(K.stack(layer_losses))

  return total_loss
