import torch.nn as nn
from torchvision import models

def get_convnext_tiny(pretrained=True, num_classes=120):
    if pretrained:
        model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
    else:
        model = models.convnext_tiny(weights=None)
    
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, num_classes)
    
    return model

def freeze_backbone(model, unfreeze_last_only=True):
    for name, param in model.named_parameters():
        if 'classifier.2' not in name:
            param.requires_grad = False
    
    return model
