import argparse
import warnings
import flwr as fl
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Select device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class NeuralNetClassifier(nn.Module):
    def __init__(self, input_dim, hidden_units=[128, 64, 32], dropout_rate=0.3):
        super(NeuralNetClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_units[0])
        self.bn1 = nn.BatchNorm1d(hidden_units[0])
        self.fc2 = nn.Linear(hidden_units[0], hidden_units[1])
        self.bn2 = nn.BatchNorm1d(hidden_units[1])
        self.fc3 = nn.Linear(hidden_units[1], hidden_units[2])
        self.bn3 = nn.BatchNorm1d(hidden_units[2])
        self.dropout = nn.Dropout(dropout_rate)
        self.out = nn.Linear(hidden_units[2], 1)  # Single output (logit)

    def forward(self, x):
        x = self.dropout(torch.relu(self.bn1(self.fc1(x))))
        x = self.dropout(torch.relu(self.bn2(self.fc2(x))))
        x = self.dropout(torch.relu(self.bn3(self.fc3(x))))
        x = self.out(x)
        return x

def load_data(dataset_path: str):
    df = pd.read_csv(dataset_path)
    for col in ["Unnamed: 0", "pid"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    if "status" in df.columns:
        X = df.drop(columns=["status"]).values
        y = df["status"].values
    else:
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    full_dataset = TensorDataset(X_tensor, y_tensor)
    
    # 80/20 split.
    n_total = len(full_dataset)
    n_train = int(0.8 * n_total)
    n_test = n_total - n_train
    train_dataset, test_dataset = random_split(full_dataset, [n_train, n_test])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    return train_loader, test_loader, X_scaled.shape[1]

# Flower client
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        new_state_dict = {k: torch.tensor(param) for k, param in zip(state_dict.keys(), parameters)}
        self.model.load_state_dict(new_state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        criterion = nn.BCEWithLogitsLoss()
        epochs = config.get("local_epochs", 1)
        for _ in range(epochs):
            for inputs, labels in self.train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = self.model(inputs).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        criterion = nn.BCEWithLogitsLoss()
        total_loss = 0.0
        all_labels = []
        all_preds = []
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs).squeeze()
                loss = criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
                preds = (torch.sigmoid(outputs) >= 0.5).float()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
        avg_loss = total_loss / len(self.test_loader.dataset)
        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(all_labels, all_preds)
        return float(avg_loss), len(self.test_loader.dataset), {"accuracy": float(accuracy)}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument("--dataset_path", type=str, default="dataset/BreastCancerDataRoystonAltman_subset_A.csv",
                        help="Path to the local client dataset (CSV file).")
    parser.add_argument("--server_address", type=str, default="0.0.0.0:8080",
                        help="Address of the Flower server.")
    args = parser.parse_args()

    train_loader, test_loader, input_dim = load_data(args.dataset_path)
    model = NeuralNetClassifier(input_dim=input_dim, hidden_units=[128, 64, 32], dropout_rate=0.3)
    client = FlowerClient(model, train_loader, test_loader)
    fl.client.start_client(server_address=args.server_address, client=client)