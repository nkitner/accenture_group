import os
import argparse
import flwr as fl
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from flwr.common import parameters_to_ndarrays
import copy

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

def get_test_data(dataset_path):
    df = pd.read_csv(dataset_path)
    for col in ["Unnamed: 0", "pid"]:
        if col in df.columns:
            df = df.drop(columns=[col])
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
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    return loader, X_scaled.shape[1]

# Custom FedAvg strategy that evaluates the aggregated global model
class FedAvgLocalVsGlobalEval(fl.server.strategy.FedAvg):
    def __init__(self, global_model, test_loader, **kwargs):
        super().__init__(**kwargs)
        self.global_model = global_model
        self.test_loader = test_loader

    def evaluate_model(self, model):
        model.eval()
        criterion = nn.BCEWithLogitsLoss()
        total_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []
        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                loss = criterion(outputs, labels)
                total_loss += loss.item() * inputs.size(0)
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.5).float()
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        n = len(self.test_loader.dataset)
        metrics = {
            "loss": total_loss / n,
            "accuracy": accuracy_score(all_labels, all_preds),
            "f1": f1_score(all_labels, all_preds),
            "recall": recall_score(all_labels, all_preds),
            "auc": roc_auc_score(all_labels, all_probs)
        }
        return metrics

    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)
        if aggregated_parameters is not None:
            ndarrays = parameters_to_ndarrays(aggregated_parameters)
            state_dict = self.global_model.state_dict()
            new_state_dict = {key: torch.tensor(param) for key, param in zip(state_dict.keys(), ndarrays)}
            self.global_model.load_state_dict(new_state_dict, strict=True)
            
            global_metrics = self.evaluate_model(self.global_model)
            print(f"After round {rnd}: Global Model Evaluation on gbsb.csv:")
            print(f"Loss: {global_metrics['loss']:.4f}, "
                  f"Accuracy: {global_metrics['accuracy']:.4f}, "
                  f"F1: {global_metrics['f1']:.4f}, "
                  f"Recall: {global_metrics['recall']:.4f}, "
                  f"AUC: {global_metrics['auc']:.4f}")
        return aggregated_parameters, aggregated_metrics

def main():
    parser = argparse.ArgumentParser(
        description="Flower Server evaluating Global Model on gbsb.csv"
    )
    parser.add_argument("--server_address", type=str, default="0.0.0.0:8080",
                        help="Address on which the server listens.")
    parser.add_argument("--num_rounds", type=int, default=5,
                        help="Number of federated learning rounds.")
    parser.add_argument("--eval_dataset", type=str, default="dataset/gbsg.csv",
                        help="Path to evaluation dataset (gbsb.csv).")
    args = parser.parse_args()

    test_loader, input_dim = get_test_data(args.eval_dataset)

    global_model = NeuralNetClassifier(
        input_dim=input_dim,
        hidden_units=[128, 64, 32],
        dropout_rate=0.3
    ).to(device)

    strategy = FedAvgLocalVsGlobalEval(
        global_model=global_model,
        test_loader=test_loader,
        fraction_fit=1.0,
        min_fit_clients=2,
        min_available_clients=2,
    )

    print(f"Starting Flower server at {args.server_address} for {args.num_rounds} rounds.")
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
