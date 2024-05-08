import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
import json
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from models import ViT_B_16
from utils import EarlyStopping, train_step, test_step, calculate_metrics

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to the expected 224 x 224 for ViT
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for pre-trained model
])

# Create ImageFolder dataset
root_dir = "dataset"
dataset = datasets.ImageFolder(root=root_dir, transform=transform)

# Define the ratio for splitting
train_ratio = 0.9
val_ratio = 1 - train_ratio

# Calculate the lengths of training and validation sets based on the ratios
num_data = len(dataset)
train_size = int(train_ratio * num_data)
val_size = num_data - train_size

# Use random_split to split the dataset into training and validation sets
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
num_classes = len(dataset.classes)

# Create DataLoader for training and validation sets
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"number of classes: {num_classes}")
print(f"number of train data: {len(train_dataset)}")
print(f"number of validation data: {len(val_dataset)}")

# create model
vit_b_16 = ViT_B_16(num_classes)
vit_b_16.to(device)


# training configurations
initial_lr = 1e-7
peak_lr = 0.0005
num_epochs = 50
num_warmup_epochs = 5
num_batches_per_epoch = len(train_loader)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vit_b_16.parameters(), lr=initial_lr)
early_stopping = EarlyStopping(patience=20, verbose=True, path='models/deep_learning.pt')

# Setup schedulers
scheduler_cosine = CosineAnnealingLR(optimizer, T_max=num_epochs - num_warmup_epochs, eta_min=0)


# training loop
results = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [],
           "train_precision": [], "train_recall": [], "train_f1": [],
           "val_precision": [], "val_recall": [], "val_f1": [],
           "learning_rate": []}
# Loop through training and testing steps for a number of epochs
for epoch in tqdm(range(num_epochs)):
    train_loss, train_acc, train_preds, train_labels = train_step(
        model=vit_b_16,
        dataloader=train_loader,
        loss_fn=criterion,
        optimizer=optimizer,
        device=device,
    )
    
    val_loss, val_acc, val_preds, val_labels = test_step(
        model=vit_b_16,
        dataloader=val_loader,
        loss_fn=criterion,
        device=device
    )
    
    # Warmup handling
    if epoch < num_warmup_epochs:
        lr_scale = (peak_lr - initial_lr) / num_warmup_epochs
        lr = initial_lr + lr_scale * epoch
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
    else:
        scheduler_cosine.step()
    
    # Calculate additional metrics for train and validation sets
    train_precision, train_recall, train_f1 = calculate_metrics(train_labels, train_preds)
    val_precision, val_recall, val_f1 = calculate_metrics(val_labels, val_preds)
    current_lr = optimizer.param_groups[0]['lr']
        
    # Print out what's happening
    print(
        f"Epoch: {epoch+1} | "
        f"train_loss: {train_loss:.4f} | "
        f"train_acc: {train_acc:.4f} | "
        f"test_loss: {val_loss:.4f} | "
        f"test_acc: {val_acc:.4f} | "
        f"learning_rate: {current_lr}"
    )
    
    early_stopping(val_loss, vit_b_16)
    if early_stopping.early_stop:
        print("Early stopping")
        break

    # Store metrics
    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["train_precision"].append(train_precision)
    results["train_recall"].append(train_recall)
    results["train_f1"].append(train_f1)

    results["val_loss"].append(val_loss)
    results["val_acc"].append(val_acc)
    results["val_precision"].append(val_precision)
    results["val_recall"].append(val_recall)
    results["val_f1"].append(val_f1)

    results["learning_rate"].append(current_lr)

save_path = "results/deep_learning.json"
with open(save_path, "w") as file:
    json.dump(results, file)
print("Results have been saved as", save_path)