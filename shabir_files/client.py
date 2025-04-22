import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score  # Added metric imports
import flwr as fl

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
        self.out = nn.Linear(hidden_units[2], 1)

    def forward(self, x):
        x = self.dropout(torch.relu(self.bn1(self.fc1(x))))
        x = self.dropout(torch.relu(self.bn2(self.fc2(x))))
        x = self.dropout(torch.relu(self.bn3(self.fc3(x))))
        x = self.out(x)
        return x

def train_model_early_stop(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=50, patience=10):
    best_val_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
            print("Improved, saving model state!")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"No improvement for {patience} epochs. Early stopping.")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model

# Flower client implementation.
class TorchFLClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader

    # Return model parameters as a list of NumPy arrays
    def get_parameters(self, config: dict = None):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    # Update model parameters
    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        new_state_dict = {key: torch.tensor(param) for key, param in zip(state_dict.keys(), parameters)}
        self.model.load_state_dict(new_state_dict, strict=True)

    def fit(self, parameters, config: dict = None):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = optim.AdamW(self.model.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
        self.model = train_model_early_stop(
            self.model,
            self.train_loader,
            self.val_loader,
            criterion=nn.BCEWithLogitsLoss(),
            optimizer=optimizer,
            scheduler=scheduler,
            epochs=50,
            patience=10,
        )
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config: dict = None):
        self.set_parameters(parameters)
        self.model.eval()
        val_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []  # To store probabilities for AUC computation

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs).squeeze()
                loss = nn.BCEWithLogitsLoss()(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        val_loss /= len(self.val_loader.dataset)
        acc = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)

        # Print the metrics
        print("Evaluation Metrics:")
        print(f"Loss: {val_loss:.4f}, Accuracy: {acc:.4f}, F1 Score: {f1:.4f}, Recall: {recall:.4f}, AUC: {auc:.4f}")

        return float(val_loss), len(self.val_loader.dataset), {
            "accuracy": float(acc),
            "f1": float(f1),
            "recall": float(recall),
            "auc": float(auc)
        }

def init_main():
    parser = argparse.ArgumentParser(description="Flower Client")
    parser.add_argument("--dataset_path", type=str, default="./dataset/BreastCancerDataRoystonAltman_subset_A.csv",
                        help="Path to the local dataset")
    parser.add_argument("--server_address", type=str, default="0.0.0.0:8080",
                        help="Address of the Flower server")
    args = parser.parse_args()

    df = pd.read_csv(args.dataset_path)
    for col in ["Unnamed: 0", "pid"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    if 'status' in df.columns:
        X = df.drop(columns=["status"]).values
        y = df["status"].values
    else:
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    # Split data into train and validation sets.
    full_dataset = TensorDataset(X_tensor, y_tensor)
    n_total = len(full_dataset)
    n_train = int(0.8 * n_total)
    n_val = n_total - n_train
    train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    input_dim = X_scaled.shape[1]
    model = NeuralNetClassifier(input_dim=input_dim, hidden_units=[128, 64, 32], dropout_rate=0.3).to(device)

    print(f"Starting Flower client with dataset at: {args.dataset_path} connecting to server {args.server_address}")
    fl_client = TorchFLClient(model, train_loader, val_loader)
    fl.client.start_client(server_address=args.server_address, client=fl_client)

if __name__ == "__main__":
    init_main()
