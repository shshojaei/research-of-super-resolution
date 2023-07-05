#pip install pyiqa

import pyiqa
import torch
import cv2
import os
from glob import glob

# create metric function
pieapp_metric = pyiqa.create_metric('pieapp').cuda()

pieApp_score = pieapp_metric('./hr_img.png', './sr_img.png')
print(pieApp_score)

# check if lower better or higher better
print(pieapp_metric.lower_better)

##just added for eagertensor to numpy error#
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

pieApp_scores = []

# create metric function
pieapp_metric = pyiqa.create_metric('pieapp').cuda()

#############################################################

# define the paths to the dataset
BASE_DATA_PATH = './results'
# [Set5 , Set14 , BSDS100 , Urban100 , div2k]
HR_test_path = os.path.join(BASE_DATA_PATH , 'hr/Set5')

# create HR image lists
HR_images = glob(HR_test_path + '/*.png')

for i in range(len(HR_images)):

  pieApp_scores.append(torch.abs(pieapp_metric('./hr/Set5/hr{}.png'.format(i) , './mae/Set5/sr{}.png'.format(i))))
  #print(pieApp_scores[i])

pieApp_scores = torch.tensor(pieApp_scores).cpu()

print("\nmean pieApp: {}".format(pieApp_scores.mean()))
