#source: https://stackoverflow.com/questions/41061457/keras-how-to-save-the-training-history-attribute-of-the-history-object

import pickle

with open('drive/MyDrive/Colab Notebooks/EDSR/my_mae/history_my_mae', 'wb') as file_pi:
  pickle.dump(history_my_mae.history, file_pi)
