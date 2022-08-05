def plot_results(i,highres, lowres, preds):

  #highres = highres.numpy()
  #preds = preds.numpy()

  plt.figure(figsize=(24, 14))
  plt.subplot(1,3,1), plt.imshow(highres), plt.title("high resolution") #plt.imsave('drive//My Drive/Colab Notebooks/EDSR/my_mse/HR/{}.jpg'.format(i), highres)
  plt.subplot(1,3,2), plt.imshow(lowres), plt.title("low resolution")
  plt.subplot(1,3,3), plt.imshow(preds), plt.title("prediction(super resolution)"), plt.xlabel('PSNR:{}\n SSIM:{}'.format(PSNR(preds,highres),tf.image.ssim(preds, highres , max_val=255))) #plt.imsave('drive//My Drive/Colab Notebooks/EDSR/my_mse/SR/{}.jpg'.format(i), preds)
  plt.show()

#image quality calculations
PSNR_scores = []
SSIM_scores = []

i=1

for lowres, highres in val.take(10):

  #lowres = tf.image.random_crop(lowres, (150,150,3))
  preds = model.predict_step(lowres)

  PSNR_scores.append(PSNR(preds, highres))
  SSIM_scores.append(tf.image.ssim(preds, highres , max_val=255))

  plot_results(i,highres,lowres,preds)
  i = i + 1

# Python program to get average of a list
def Average(input_list):
    return sum(input_list) / len(input_list)

print("\nTotal PSNR: {}".format(Average(PSNR_scores)))
print("\nTotal SSIM: {}".format(Average(SSIM_scores)))
