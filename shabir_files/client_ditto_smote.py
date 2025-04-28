import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import flwr as fl
import copy

# Pick the best available device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
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

def train_model_ditto(model, train_loader, val_loader, criterion, optimizer, scheduler,
                      global_state_dict, epochs=50, patience=10, lambda_reg=0.1):
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
            # Regularization: keep local model close to global
            reg_loss = sum(
                torch.norm(param - global_state_dict[name].to(device))**2
                for name, param in model.named_parameters()
            )
            loss = loss + (lambda_reg / 2.0) * reg_loss
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
                val_loss += criterion(outputs, labels).item() * inputs.size(0)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = copy.deepcopy(model.state_dict())
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

class TorchFLClientDITTO(fl.client.NumPyClient):
    def __init__(self, model, train_loader, val_loader, lambda_reg=0.1):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lambda_reg = lambda_reg

    def get_parameters(self, config=None):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = self.model.state_dict()
        new_state = {
            key: torch.tensor(param) 
            for key, param in zip(state_dict.keys(), parameters)
        }
        self.model.load_state_dict(new_state, strict=True)
    
    def fit(self, parameters, config=None):
        # Load global params
        self.set_parameters(parameters)
        global_state = copy.deepcopy(self.model.state_dict())

        # Optimizer, scheduler, loss
        optimizer = optim.AdamW(self.model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        criterion = nn.BCEWithLogitsLoss()

        # Local training with Ditto regularization
        self.model = train_model_ditto(
            self.model, self.train_loader, self.val_loader,
            criterion, optimizer, scheduler,
            global_state,
            epochs=50, patience=10, lambda_reg=self.lambda_reg
        )
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config=None):
        self.set_parameters(parameters)
        self.model.eval()
        criterion = nn.BCEWithLogitsLoss()
        total_loss, all_labels, all_preds, all_probs = 0.0, [], [], []
        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = self.model(inputs).squeeze()
                total_loss += criterion(outputs, labels).item() * inputs.size(0)
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        n = len(self.val_loader.dataset)
        metrics = {
            "loss": total_loss / n,
            "accuracy": accuracy_score(all_labels, all_preds),
            "f1":         f1_score(all_labels, all_preds),
            "recall":     recall_score(all_labels, all_preds),
            "auc":        roc_auc_score(all_labels, all_probs),
        }
        print("Local evaluation:", metrics)
        return float(metrics["loss"]), n, metrics

def main():
    parser = argparse.ArgumentParser(description="Flower Ditto Client with SMOTE")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="./dataset/BreastCancer_subset_A_younger.csv",
        help="Path to local CSV"
    )
    parser.add_argument(
        "--server_address",
        type=str,
        default="0.0.0.0:8080",
        help="Flower server address"
    )
    parser.add_argument(
        "--lambda_reg",
        type=float,
        default=0.1,
        help="Ditto regularization coefficient"
    )
    args = parser.parse_args()

    df = pd.read_csv(args.dataset_path)
    for col in ["Unnamed: 0", "pid"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Split features/target
    X = df.drop(columns=["status"]).values
    y = df["status"].values

    # 1) Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2) Stratified train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, stratify=y, random_state=42
    )

    # 3) SMOTE on training data
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # 4) Build DataLoaders
    train_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_train_res, dtype=torch.float32),
            torch.tensor(y_train_res, dtype=torch.float32),
        ),
        batch_size=32, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32),
        ),
        batch_size=32, shuffle=False
    )

    # Model & FL client
    input_dim = X_scaled.shape[1]
    model = NeuralNetClassifier(input_dim=input_dim).to(device)

    print(f"Starting Flower Ditto client with dataset: {args.dataset_path}")
    client = TorchFLClientDITTO(model, train_loader, val_loader, lambda_reg=args.lambda_reg)
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client
    )

if __name__ == "__main__":
    main()
