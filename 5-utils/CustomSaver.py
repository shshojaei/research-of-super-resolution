#https://www.tensorflow.org/guide/keras/custom_callback

import keras
import os.path

class CustomSaver(keras.callbacks.Callback):

  def on_train_begin(self, logs={}):
    self.val_loss = []
    self.val_PSNR = []   
    
  def on_epoch_end(self, epoch, logs={}):
    self.val_loss.append(logs.get('val_loss'))
    self.val_PSNR.append(logs.get('val_PSNR'))

    if ((self.val_loss[epoch-1] < self.val_loss[epoch]) and (self.val_PSNR[epoch-1] < self.val_PSNR[epoch]) or (self.val_loss[epoch-1] > self.val_loss[epoch]) and (self.val_PSNR[epoch-1] > self.val_PSNR[epoch])):
      self.model.save("drive/MyDrive/Colab Notebooks/EDSR/my_mse/model_epoch{}.h5".format(epoch+1))#second epoch (epoch index starts from 0)
      #self.model.stop_training=True


saver = CustomSaver()

model.compile(optimizer=optimizer_ , loss='mse', metrics=[PSNR])
history_my_mse = model.fit(train_ds, epochs=100, steps_per_epoch=200, validation_data=val_ds, callbacks=[saver])
