inceptionResnet = InceptionResNetV2(include_top=False, weights=None, pooling='avg')

for layer in inceptionResnet.layers:
  print(layer.name)
