import torch
from torch import nn
import torchvision.models as models

from torchvision.models.vision_transformer import MLPBlock
from opacus.layers.dp_multihead_attention import DPMultiheadAttention
from functools import partial
from typing import Callable
from collections import OrderedDict

class MyEncoderBlock(nn.Module):
    def __init__(self, num_heads, hidden_dim, mlp_dim, dropout, attention_dropout,
        norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()
        self.num_heads = num_heads

        # Attention block
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = DPMultiheadAttention(hidden_dim, num_heads, dropout=attention_dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

        # MLP block
        self.ln_2 = norm_layer(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout)

    def forward(self, input: torch.Tensor):
        torch._assert(input.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {input.shape}")
        x = self.ln_1(input)
        x, _ = self.self_attention(x, x, x, need_weights=False)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y


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
    def __init__(self, num_classes, num_layers = 12):
        super(ViT_B_16, self).__init__()
        self.num_classes = num_classes
        self.weights = models.ViT_B_16_Weights.DEFAULT
        self.model = models.vit_b_16(weights=self.weights)
        
        # Modify the encoder layers
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = MyEncoderBlock(
                num_heads=12,
                hidden_dim=768,
                mlp_dim=3072,
                dropout=0.0,
                attention_dropout=0.0,
                norm_layer=partial(nn.LayerNorm, eps=0.000001),
            )
        self.model.encoder.layers = nn.Sequential(layers)
        
        self.model.heads.head = nn.Linear(self.model.heads.head.in_features, num_classes)
    
    def forward(self, x):
        return self.model(x)
