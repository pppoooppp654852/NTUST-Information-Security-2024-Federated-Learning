import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
import json
from tqdm.auto import tqdm
from torch.optim.lr_scheduler import CosineAnnealingLR
from models import ViT_B_16
from sklearn.metrics import precision_score, recall_score, f1_score
from pathlib import Path
import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path=None, save=True):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.delta = delta
        if not Path(path).exists():
            Path(path).mkdir()
        self.path = path
        self.save = save
        

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            if self.save:
                self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.save:
                self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        checkpoint_path = Path(self.path) / 'deep_learning.pt'
        torch.save(model.state_dict(), str(checkpoint_path))
        self.val_loss_min = val_loss
    
def train_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
):
    model.train()
    train_loss, train_acc = 0, 0
    all_preds, all_labels = [], []
    
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Training")
    
    for batch, (X, y) in progress_bar:
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)

        all_preds.extend(y_pred_class.detach().cpu().numpy())
        all_labels.extend(y.detach().cpu().numpy())
        
        # Update progress bar description
        progress_bar.set_description(f"Training - Loss: {train_loss/(batch+1):.4f}, Acc: {train_acc/(batch+1):.4f}")

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc, torch.tensor(all_preds), torch.tensor(all_labels)

def test_step(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_fn: torch.nn.Module,
    device: torch.device,
):
    model.eval()
    test_loss, test_acc = 0, 0
    all_preds, all_labels = [], []
    
    with torch.inference_mode():
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc="Testing")
        
        for batch, (X, y) in progress_bar:
            X, y = X.to(device), y.to(device)
            y_pred = model(X)
            loss = loss_fn(y_pred, y)
            test_loss += loss.item()

            y_pred_class = y_pred.argmax(dim=1)
            test_acc += (y_pred_class == y).sum().item() / len(y_pred_class)

            all_preds.extend(y_pred_class.detach().cpu().numpy())
            all_labels.extend(y.detach().cpu().numpy())
            
            # Update progress bar description
            progress_bar.set_description(f"Testing - Loss: {test_loss/(batch+1):.4f}, Acc: {test_acc/(batch+1):.4f}")

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc, torch.tensor(all_preds), torch.tensor(all_labels)

def calculate_metrics(y_true, y_pred):
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().numpy()
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    return precision, recall, f1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to the expected 224 x 224 for ViT
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for pre-trained model
])

# Create ImageFolder dataset
dataset_name = 'virus_share_177'
dataset_dir = Path("dataset") / dataset_name
dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)

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
initial_lr = 1e-8
peak_lr = 0.0005
num_epochs = 50
num_warmup_epochs = 10
num_batches_per_epoch = len(train_loader)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vit_b_16.parameters(), lr=initial_lr, weight_decay=0.01)
early_stopping = EarlyStopping(patience=10, verbose=True, path='models/', save=False)

# Setup schedulers
scheduler_cosine = CosineAnnealingLR(optimizer, T_max=num_epochs - num_warmup_epochs, eta_min=0)


# Loop through training and testing steps for a number of epochs
results = {"loss": [], "accuracy": []}
for epoch in range(num_epochs):
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
    if epoch <= num_warmup_epochs:
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
        f"learning_rate: {current_lr:.6f}"
    )
    
    early_stopping(val_loss, vit_b_16)
    if early_stopping.early_stop:
        print("Early stopping")
        break

    # Store metrics
    results["loss"].append(val_loss)
    results["accuracy"].append(val_acc)

save_path = f"results/{dataset_name}.json"
with open(save_path, "w") as file:
    json.dump(results, file)
print("Results have been saved as", save_path)