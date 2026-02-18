import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50

def initializeResnetModel(model, num_class):
  if model == 'resnet18':
    model = resnet18(weights=True)
  elif model == 'resnet34':
    model = resnet34(weights=True)
  elif model == 'resnet50':
    model = resnet50(weights=True)
  else:
     raise ValueError(f'Invalid model {model}.')
  
  model.fc = nn.Linear(model.fc.in_features, num_class)
  model = model.to("cpu").eval()
  return model