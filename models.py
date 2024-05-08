import torch
from torch import nn
import torchvision.models as models

class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.num_classes = num_classes
        self.weights = models.ResNet18_Weights.DEFAULT
        self.model = models.resnet18(weights=self.weights)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(512, self.num_classes)
    
    def forward(self, x):
        return self.model(x)
    
class ViT_B_16(nn.Module):
    def __init__(self, num_classes):
        super(ViT_B_16, self).__init__()
        self.num_classes = num_classes
        self.weights = models.ViT_B_16_Weights.DEFAULT
        self.model = models.vit_b_16(weights=self.weights)
        self.model.heads.head = nn.Linear(self.model.heads.head.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)
