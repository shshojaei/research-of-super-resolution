#load history
import pickle

history_tf_mae = pickle.load(open('drive/MyDrive/Colab Notebooks/EDSR/histories/history_EDSR_tensorflow_mae_50_epoch', "rb"))
history_my_mae = pickle.load(open('drive/MyDrive/Colab Notebooks/EDSR/histories/history_EDSR_my_mae_50_epoch', "rb"))

import matplotlib.pyplot as plt

epochs = range(2,len(history_my_mae["val_PSNR"]))

fig, ax = plt.subplots(figsize=(10,5))

ax.plot(epochs, history_tf_mae["val_PSNR"][2:], label='tf_mae')  
ax.plot(epochs, history_my_mae["val_PSNR"][2:], label='my_mae')   

ax.set_xlabel('epochs')  
ax.set_ylabel('validation PSNR')  
ax.set_title("tf_mae vs. my_mae") 
ax.legend()
