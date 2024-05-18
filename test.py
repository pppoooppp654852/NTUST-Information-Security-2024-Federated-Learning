import torch
import torchvision.models as models
from torchinfo import summary

weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights)
model.fc = torch.nn.Linear(512, 12)

summary(model=model,
        input_size=(16, 3, 224, 224), # (batch_size, color_channels, height, width)
        # col_names=["input_size"], # uncomment for smaller output
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"]
)

