import argparse
import yaml
import logging
from pathlib import Path
from typing import Tuple, List, Dict

import flwr as fl
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from flwr.common.parameter import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.typing import Scalar

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

# Data Loading Function 
def get_data(dataset_path: str) -> Tuple[DataLoader, int]:
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        print(f"Error loading dataset from {dataset_path}: {e}")
        raise

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
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    input_dim = X_scaled.shape[1]
    return loader, input_dim

# Client Evaluation Configuration Functions
def on_evaluate_config_fn(server_round: int) -> dict:
    return {"local_epochs": 1, "batch_size": 32}

def evaluate_metrics_aggregation_fn(
    evaluate_results: List[Tuple[float, int, Dict[str, Scalar]]]
) -> Tuple[float, Dict[str, Scalar]]:
    total_examples = sum(num for _, num, _ in evaluate_results)
    aggregated_loss = sum(loss * num for loss, num, _ in evaluate_results) / total_examples
    aggregated_accuracy = sum(metrics["accuracy"] * num for _, num, metrics in evaluate_results) / total_examples
    return aggregated_loss, {"accuracy": aggregated_accuracy}

# Final Evaluation Function (Server)
def final_evaluation(model: nn.Module, test_loader: DataLoader):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            loss = criterion(outputs, labels)
            total_loss += loss.item() * inputs.size(0)
            preds = (torch.sigmoid(outputs) >= 0.5).float()
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    avg_loss = total_loss / len(test_loader.dataset)
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"final server evaluation: Test Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")

# YAML configuration
def load_config(config_path: str) -> dict:
    with open(config_path, "r") as file:
        return yaml.safe_load(file)

# main
def main():
    parser = argparse.ArgumentParser(
        description="Flower Server: Client Evaluation Every Round and Final Server Evaluation"
    )
    parser.add_argument("--server_address", type=str, default="0.0.0.0:8080",
                        help="Server address.")
    parser.add_argument("--client_dataset_path", type=str, default="dataset/BreastCancerDataRoystonAltman_subset_A.csv",
                        help="Path to dataset used by clients for local evaluation.")
    parser.add_argument("--final_dataset_path", type=str, default="dataset/gbsg.csv",
                        help="Path to dataset used by the server for final evaluation.")
    parser.add_argument("--num_rounds", type=int, default=5,
                        help="Number of federated learning rounds.")
    parser.add_argument("--config", type=str, default=None,
                        help="Optional path to YAML configuration file.")
    args = parser.parse_args()

    # Default FL parameters.
    default_config = {
        "server_address": "0.0.0.0:8080",
        "num_rounds": 5,
        "min_fit_clients": 2,
        "min_available_clients": 2,
        "fraction_fit": 1.0,
    }
    # Load YAML config if provided.
    if args.config:
        config_yaml = load_config(args.config)
        config = {**default_config, **config_yaml}
    else:
        config = default_config

    server_address = config["server_address"]
    num_rounds = config["num_rounds"]
    min_fit_clients = config["min_fit_clients"]
    min_available_clients = config["min_available_clients"]
    fraction_fit = config["fraction_fit"]

    # client dataset
    client_loader, input_dim_client = get_data(args.client_dataset_path)
    # gbsb dataset
    final_loader, input_dim_final = get_data(args.final_dataset_path)
    input_dim = input_dim_client

    # Instantiate the global model.
    global_model = NeuralNetClassifier(
        input_dim=input_dim,
        hidden_units=[128, 64, 32],
        dropout_rate=0.3
    ).to(device)
    initial_parameters = ndarrays_to_parameters(
        [val.cpu().numpy() for _, val in global_model.state_dict().items()]
    )

    # FedAvg strategy - client-side evaluation
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=fraction_fit,
        min_fit_clients=min_fit_clients,
        min_available_clients=min_available_clients,
        on_evaluate_config_fn=on_evaluate_config_fn,
        evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        initial_parameters=initial_parameters,
    )

    print(f"Starting server at {server_address} for {num_rounds} rounds.")
    fl.server.start_server(
        server_address=server_address,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

    # final evaluation on the server.
    print("Federated training rounds completed. Running final server evaluation...")
    final_evaluation(global_model, final_loader)

if __name__ == "__main__":
    main()
