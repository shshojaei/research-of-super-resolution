import torch
import torchvision
from torchvision.models import resnet50

model = resnet50(pretrained=True)
model.eval()

# Freeze the convolutional layers
for param in model.parameters():
    param.requires_grad = False

# Extract features from layer4 (before avg pool)
layer_name = 'layer1'
resnet50_model = torch.nn.Sequential(*list(model.children())[:-2])

if torch.cuda.is_available():
    resnet50_model.cuda()

def perceptual_loss(y_true, y_pred):
  y_true_features = resnet50_model(y_true)
  y_pred_features = resnet50_model(y_pred)
  return torch.mean(torch.abs(y_true_features - y_pred_features))
  
