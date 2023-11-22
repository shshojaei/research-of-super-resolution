!pip install torch-dct

import torch
import torch.nn.functional as F
import torch_dct as dct

# Quantization matrix
Q = torch.tensor([[16, 11, 10, 16, 24, 40, 51, 61],
                 [12, 12, 14, 19, 26, 58, 60, 55],
                 [14, 13, 16, 24, 40, 57, 69, 56],
                 [14, 17, 22, 29, 51, 87, 80, 62],
                 [18, 22, 37, 56, 68, 109, 103, 77],
                 [24, 35, 55, 64, 81, 104, 113, 92],
                 [49, 64, 78, 87, 103, 121, 120, 101],
                 [72, 92, 95, 98, 112, 100, 103, 99]])

# Reshape to match dimensions
Q = Q.repeat(24, 24)
Q = Q.reshape(1, 192, 192, 1).repeat(1, 1, 1, 3).cuda()

def dct_q_loss(y_true, y_pred):

  y_true = y_true.float()
  y_pred = y_pred.float()

  y_true_dct = dct.dct_2d(y_true)
  y_pred_dct = dct.dct_2d(y_pred)

  #print(y_true_dct.shape)
  #print(y_pred_dct.shape)

  y_true_dct = y_true_dct.reshape(8, 192, 192 , 3)
  y_pred_dct = y_pred_dct.reshape(8, 192, 192 , 3)

  y_true_quant = y_true_dct / Q
  y_pred_quant = y_pred_dct / Q

  loss = torch.mean(torch.abs(y_true_quant - y_pred_quant))

  if torch.cuda.is_available():
    loss.cuda()

  return loss

###### Train part of ESRT code(https://github.com/luissen/ESRT) ######

dct_q_criterion = nn.L1loss()

if cuda:
    model = model.to(device)
    #l1_criterion = l1_criterion.to(device)
    dct_q_criterion = dct_q_loss

# . . . #

loss_dct = dct_q_criterion(sr_tensor, hr_tensor)
