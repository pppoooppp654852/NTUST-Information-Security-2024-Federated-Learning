import flwr as fl
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import sys
from collections import OrderedDict
from models import ViT_B_16
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
from opacus import PrivacyEngine
from typing import List

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images to the expected 224 x 224 for ViT
    transforms.ToTensor(),  # Convert images to PyTorch tensors
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize for pre-trained model
])

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, data_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset = torch.load(data_path)
        self.train_loader = DataLoader(self.dataset, batch_size=16, shuffle=True)
        self.test_loader = DataLoader(torch.load("data_split/val.pt"), batch_size=16, shuffle=False)
        print(type(self.test_loader))
        self.num_classes = len(self.test_loader.dataset.dataset.classes)
        self.model = ViT_B_16(self.num_classes).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0005)
        self.criterion = nn.CrossEntropyLoss()
        
    def apply_differential_privacy(self, epsilon: float, epochs: int, delta: float | None = None, max_grad_norm: float | List[float] = 1.0) -> None:
        if delta is None:
            delta = 1 / len(self.dataset)
        diff_pri_eng = PrivacyEngine()
        self.model, self.optimizer, self.train_loader = diff_pri_eng.make_private_with_epsilon(
            module=self.model,
            optimizer=self.optimizer,
            data_loader=self.train_loader,
            target_epsilon=epsilon,
            target_delta=delta,
            epochs=epochs,
            max_grad_norm=max_grad_norm
        )
        self.model.to(self.device)

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        for X, y in self.train_loader:
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(X)
            loss = self.criterion(output, y)
            loss.backward()
            self.optimizer.step()
        return self.get_parameters(config={}), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        total_loss, total_correct, total_samples = 0, 0, 0
        with torch.no_grad():
            for X, y in self.test_loader:
                X, y = X.to(self.device), y.to(self.device)
                output = self.model(X)
                loss = self.criterion(output, y)
                total_loss += loss.item() * X.size(0)
                total_correct += (output.argmax(1) == y).type(torch.float).sum().item()
                total_samples += X.size(0)
        average_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        return average_loss, total_samples, {"accuracy": accuracy, "loss": average_loss, "num_examples": total_samples}
    

if __name__ == "__main__":
    data_path = sys.argv[1]  # Pass the data path as the first command line argument
    epsilon = float(sys.argv[2]) # Pass epsilon as the second command line argument
    client = FlowerClient(data_path)
    client.apply_differential_privacy(epsilon, 40)
    fl.client.start_numpy_client(server_address="localhost:8080", client=client.to_client())