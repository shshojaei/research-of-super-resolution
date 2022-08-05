for lowres, highres in train.take(1):

  #lowres = tf.image.random_crop(lowres, (150,150,3))
  lowres_pixels = np.asarray(lowres)
  # confirm pixel range is 0-255
  print('\nlowres Data Type: %s' % lowres_pixels.dtype)
  print('\nMin: %.3f, Max: %.3f' % (lowres_pixels.min(), lowres_pixels.max()))

  preds = model.predict_step(lowres)
  preds_pixels = np.asarray(preds)
  # confirm pixel range is 0-255
  print('\nprediction Data Type: %s' % preds_pixels.dtype)
  print('\nMin: %.3f, Max: %.3f' % (preds_pixels.min(), preds_pixels.max()))
