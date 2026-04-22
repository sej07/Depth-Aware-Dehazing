import torch
import torch.nn as nn
from torchvision import models


class PerceptualLoss(nn.Module):
    def __init__(self, layers=None):
        super().__init__()

        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.features = vgg.features

        for param in self.features.parameters():
            param.requires_grad = False
        if layers is None:
            self.layer_indices = [4, 9, 16]
        else:
            self.layer_indices = layers
        
        self.criterion = nn.L1Loss()
        
    def normalize(self, x):
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        return (x - mean) / std
    
    def extract_features(self, x):
        x = self.normalize(x)
        features = []
        
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in self.layer_indices:
                features.append(x)
                
            if len(features) == len(self.layer_indices):
                break
                
        return features
        
    def forward(self, pred, target):
        self.features = self.features.to(pred.device)
        
        pred_features = self.extract_features(pred)
        target_features = self.extract_features(target)
        
        loss = 0.0
        for pf, tf in zip(pred_features, target_features):
            loss += self.criterion(pf, tf)
            
        return loss / len(self.layer_indices)