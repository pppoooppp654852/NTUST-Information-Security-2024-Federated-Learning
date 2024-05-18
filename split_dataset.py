import os
from torchvision import datasets, transforms
from torch.utils.data import random_split
import torch

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to the expected 224 x 224 for ViT
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for pre-trained model
])

# Create ImageFolder dataset
root_dir = "dataset/virus_share_177"
dataset = datasets.ImageFolder(root=root_dir, transform=transform)
num_classes = len(dataset.classes)

# Define the ratio for splitting
train_ratio = 0.8
val_ratio = 1 - train_ratio

# Calculate the lengths of training and validation sets based on the ratios
num_data = len(dataset)
train_size = int(train_ratio * num_data)
val_size = num_data - train_size

# Use random_split to split the dataset into training and validation sets
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Calculate sizes for split
total_size = len(train_dataset)
part_size = total_size // 3

# Split dataset
part1, part2, remaining = random_split(train_dataset, [part_size, part_size, total_size - 2 * part_size])
part3, _ = random_split(remaining, [part_size, len(remaining) - part_size])
part3 = part3.dataset

# Create directories for each part
os.makedirs('./data_split', exist_ok=True)

# Save each part
torch.save(part1, './data_split/part1.pt')
torch.save(part2, './data_split/part2.pt')
torch.save(part3, './data_split/part3.pt')
torch.save(val_dataset, './data_split/val.pt')