#load history
import pickle
import matplotlib.pyplot as plt

history_tf_mae = pickle.load(open('./drive/MyDrive/ColabNotebooks/EDSR-Tensorflow/history_mae_x4', "rb"))
history_my_mae = pickle.load(open('./drive/MyDrive/ColabNotebooks/EDSR-Tensorflow/history_my_mae_x4', "rb"))
history_tf_mse = pickle.load(open('./drive/MyDrive/ColabNotebooks/EDSR-Tensorflow/history_mse_x4', "rb"))
history_my_mse = pickle.load(open('./drive/MyDrive/ColabNotebooks/EDSR-Tensorflow/history_my_mse_x4', "rb"))
history_cauchy = pickle.load(open('./drive/MyDrive/ColabNotebooks/EDSR-Tensorflow/history_cauchy_x4_again', "rb"))
history_fair = pickle.load(open('./drive/MyDrive/ColabNotebooks/EDSR-Tensorflow/history_fair_x4', "rb"))
history_geman = pickle.load(open('./drive/MyDrive/ColabNotebooks/EDSR-Tensorflow/history_geman_x4', "rb"))
history_huber = pickle.load(open('./drive/MyDrive/ColabNotebooks/EDSR-Tensorflow/history_huber_x4', "rb"))
history_logistic = pickle.load(open('./drive/MyDrive/ColabNotebooks/EDSR-Tensorflow/history_logistic_x4', "rb"))
history_phuber = pickle.load(open('./drive/MyDrive/ColabNotebooks/EDSR-Tensorflow/history_phuber_x4', "rb"))
history_talwar = pickle.load(open('./drive/MyDrive/ColabNotebooks/EDSR-Tensorflow/history_talwar_x4', "rb"))
history_tukey = pickle.load(open('./drive/MyDrive/ColabNotebooks/EDSR-Tensorflow/history_tukey_x4', "rb"))
history_welsch = pickle.load(open('./drive/MyDrive/ColabNotebooks/EDSR-Tensorflow/history_welsch_x4', "rb"))

epochs = range(2,len(history_my_mae["val_PSNR"]))

fig, ax = plt.subplots(figsize=(10,5))

ax.plot(epochs, smooth_curve(history_tf_mae["val_PSNR"][2:]), '+', label='tf_mae')
ax.plot(epochs, smooth_curve(history_my_mae["val_PSNR"][2:]), 'x', label='my_mae' )
ax.plot(epochs, smooth_curve(history_tf_mse["val_PSNR"][2:]), '_', label='tf_mse')
ax.plot(epochs, smooth_curve(history_my_mse["val_PSNR"][2:]), '.', label='my_mse')
ax.plot(epochs, smooth_curve(history_cauchy["val_PSNR"][2:]), '^', label='cauchy')
ax.plot(epochs, smooth_curve(history_fair["val_PSNR"][2:]), '8', label='fair' )
ax.plot(epochs, smooth_curve(history_geman["val_PSNR"][2:]), 's', label='geman')
ax.plot(epochs, smooth_curve(history_huber["val_PSNR"][2:]), '*', label='huber')
ax.plot(epochs, smooth_curve(history_logistic["val_PSNR"][2:]), 'X', label='logistic')
ax.plot(epochs, smooth_curve(history_phuber["val_PSNR"][2:]), 'P', label='phuber')
ax.plot(epochs, smooth_curve(history_talwar["val_PSNR"][2:]), 'd', label='talwar')
ax.plot(epochs, smooth_curve(history_tukey["val_PSNR"][2:]), 'o', label='tukey')
ax.plot(epochs, smooth_curve(history_welsch["val_PSNR"][2:]), 'H', label='welsch')

ax.set_xlabel('epochs')
ax.set_ylabel('validation PSNR')
ax.set_title("compare PSNR of pixelwise losses")
ax.legend()
