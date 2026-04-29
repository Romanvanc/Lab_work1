import torch.nn as nn
import torch.optim as optim
from torchvision import models

def get_model(num_classes=120):
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    
    for name, param in model.named_parameters():
        if 'classifier.2' not in name:
            param.requires_grad = False
    
    return model

def get_optimizer(model):
    return optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
