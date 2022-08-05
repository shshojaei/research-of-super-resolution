import matplotlib.pyplot as plt

loss = history_EDSR_tensorflow_mae_50_epoch.history['loss']
val_loss = history_EDSR_tensorflow_mae_50_epoch.history['val_loss']

epochs = range(len(loss))

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.figure()

plt.show()

###################################################

import matplotlib.pyplot as plt

PSNR = history_EDSR_tensorflow_mae_50_epoch.history['PSNR']
val_PSNR = history_EDSR_tensorflow_mae_50_epoch.history['val_PSNR']

epochs = range(len(PSNR))

plt.plot(epochs, PSNR, 'bo', label='Training PSNR')
plt.plot(epochs, val_PSNR, 'b', label='Validation PSNR')
plt.title('Training and validation PSNR')
plt.legend()

plt.figure()

plt.show()
