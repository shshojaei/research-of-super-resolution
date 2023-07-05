#pip install piq

import torch
from piq import PieAPP
import cv2
import os
from glob import glob

pieApp_scores = []

pieApp = PieAPP().cuda()

# define the paths to the dataset
# [Set5 , Set14 , BSDS100 , Urban100 , DIV2K/DIV2K_valid_HR]
BASE_DATA_PATH = './results'
HR_test_path = os.path.join(BASE_DATA_PATH , 'hr/Set5')
SR_test_path = os.path.join(BASE_DATA_PATH, 'mae/Set5')

# create LR and HR image lists
HR_images = glob(HR_test_path + '/*.png')
SR_images = glob(SR_test_path + '/*.png')

# sort the lists
HR_images.sort()
SR_images.sort()

for i in range(len(HR_images)):

  #preprocess images
  HR_image = cv2.cvtColor(cv2.imread(HR_images[i]), cv2.COLOR_BGR2RGB)/255
  SR_image = cv2.cvtColor(cv2.imread(SR_images[i]), cv2.COLOR_BGR2RGB)/255

  HR_image = np.transpose(HR_image , (2,0,1))
  HR_image = torch.from_numpy(HR_image).unsqueeze(0)
  HR_image = HR_image.float()

  SR_image = np.transpose(SR_image , (2,0,1))
  SR_image = torch.from_numpy(SR_image).unsqueeze(0)
  SR_image = SR_image.float()

  pieApp_scores.append(torch.abs(pieApp(HR_image,SR_image)))
  #print("\npieApp for img{}:".format(i), pieApp_scores[i])

pieApp_scores = torch.tensor(pieApp_scores).cpu()

print("\nmean pieApp: {}".format(pieApp_scores.mean()))
