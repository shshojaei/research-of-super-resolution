#source: https://github.com/fchollet/deep-learning-with-python-notebooks/blob/master/first_edition/5.3-using-a-pretrained-convnet.ipynb

def smooth_curve(points, factor=0.8):
  smoothed_points = []
  for point in points:
    if smoothed_points:
      previous = smoothed_points[-1]
      smoothed_points.append(previous * factor + point * (1 - factor))
    else:
      smoothed_points.append(point)
  return smoothed_points

########################################

import matplotlib.pyplot as plt

epochs = range(2,len(history_my_mae["val_PSNR"]))

fig, ax = plt.subplots(figsize=(10,5))

ax.plot(epochs, smooth_curve(history_tf_mae["val_PSNR"][2:]), label='tf_mae')  
ax.plot(epochs, smooth_curve(history_my_mae["val_PSNR"][2:]), label='my_mae')   

ax.set_xlabel('epochs')  
ax.set_ylabel('validation PSNR')  
ax.set_title("tf_mae vs. my_mae") 
ax.legend()
