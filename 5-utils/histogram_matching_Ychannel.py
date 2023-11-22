  #-----------------histogram matching------------
  #from skimage import exposure
  #preds = K.cast( preds, 'float32')
  #HR_image = K.cast( HR_image, 'float32')
  #preds = preds.numpy()
  #HR_image = HR_image.numpy()
  #matched = exposure.match_histograms(preds, HR_image)

  #matched = matched.astype(np.uint8)
  #HR_image = HR_image.astype(np.uint8)
  #-----------------------------------------------

  #---------------- preds + Y --------------------

  #convert Tensorflow tensor to numpy
  #preds = preds.numpy()
  #HR_image = HR_image.numpy()

  #convert the image to YCrCb color space
  #img_ycc = cv2.cvtColor(HR_image , cv2.COLOR_RGB2BGR)
  #img_ycc = cv2.cvtColor(HR_image , cv2.COLOR_BGR2YCR_CB)

  #pred_ycc = cv2.cvtColor(preds , cv2.COLOR_RGB2BGR)
  #pred_ycc = cv2.cvtColor(preds , cv2.COLOR_BGR2YCR_CB)

  #split the channels
  #y, cr, cb = cv2.split(img_ycc)

  #replace y channel from hr_image to preds
  #pred_ycc = cv2.merge((y , pred_ycc[...,1] , pred_ycc[... , 2]))

  #convert back
  #pred_rgb = cv2.cvtColor(pred_ycc , cv2.COLOR_YCR_CB2RGB)
  #pred_rgb = pred_rgb.astype(np.uint8)

  #-----------------------------------------------
