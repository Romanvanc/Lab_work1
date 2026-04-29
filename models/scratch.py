import torch.nn as nn
import torch.optim as optim
from torchvision import models

def get_model(num_classes=120):
    model = models.convnext_tiny(weights=None)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    return model

def get_optimizer(model):
    return optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
