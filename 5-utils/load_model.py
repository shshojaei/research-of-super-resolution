model = tf.keras.models.load_model('drive/My Drive/Colab Notebooks/EDSR/models/model.h5', custom_objects={'EDSRModel':EDSRModel, 'PSNR':PSNR })
