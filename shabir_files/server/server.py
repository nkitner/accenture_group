import argparse
import yaml
import flwr as fl
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, recall_score, roc_auc_score
from flwr.common import parameters_to_ndarrays  # helper to convert Parameters to list[np.ndarray]

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
        self.out = nn.Linear(hidden_units[2], 1)  # Single output neuron (logit)

    def forward(self, x):
        x = self.dropout(torch.relu(self.bn1(self.fc1(x))))
        x = self.dropout(torch.relu(self.bn2(self.fc2(x))))
        x = self.dropout(torch.relu(self.bn3(self.fc3(x))))
        x = self.out(x)
        return x

# load and preprocess test data for evaluation
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
    full_dataset = TensorDataset(X_tensor, y_tensor)
    test_loader = DataLoader(full_dataset, batch_size=32, shuffle=False)
    return test_loader, X_scaled.shape[1]

# strategy - intermediate evaluation
class FedAvgWithIntermediateEval(fl.server.strategy.FedAvg):
    def __init__(self, global_model, test_loader, **kwargs):
        super().__init__(**kwargs)
        self.global_model = global_model
        self.test_loader = test_loader

    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(rnd, results, failures)
        if aggregated_parameters is not None:
            # Parameters to list of array conversion
            ndarrays = parameters_to_ndarrays(aggregated_parameters)
            state_dict = self.global_model.state_dict()
            new_state_dict = {
                key: torch.tensor(param) for key, param in zip(state_dict.keys(), ndarrays)
            }
            self.global_model.load_state_dict(new_state_dict, strict=True)
            self.global_model.eval()

            # evaluation on the test set
            criterion = nn.BCEWithLogitsLoss()
            total_loss = 0.0
            all_labels = []
            all_preds = []
            all_probs = []  # AUC calculation

            with torch.no_grad():
                for inputs, labels in self.test_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = self.global_model(inputs).squeeze()
                    loss = criterion(outputs, labels)
                    total_loss += loss.item() * inputs.size(0)
                    
                    # sigmoid, thresholds at 0.5
                    probs = torch.sigmoid(outputs)
                    preds = (probs >= 0.5).float()
                    
                    all_labels.extend(labels.cpu().numpy())
                    all_preds.extend(preds.cpu().numpy())
                    all_probs.extend(probs.cpu().numpy())

            avg_loss = total_loss / len(self.test_loader.dataset)
            accuracy = accuracy_score(all_labels, all_preds)
            f1 = f1_score(all_labels, all_preds)
            recall = recall_score(all_labels, all_preds)
            auc = roc_auc_score(all_labels, all_probs)

            print(f"After round {rnd}: Test Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}, "
                  f"F1 Score = {f1:.4f}, Recall = {recall:.4f}, AUC = {auc:.4f}")
        return aggregated_parameters, aggregated_metrics

def main():
    parser = argparse.ArgumentParser(
        description="Flower Server with Intermediate Evaluation (Custom Strategy)"
    )
    parser.add_argument("--server_address", type=str, default="0.0.0.0:8080",
                        help="Address on which the server listens.")
    parser.add_argument("--dataset_path", type=str, default="dataset/gbsg.csv",
                        help="Path to the test evaluation dataset (held-out or same as training).")
    parser.add_argument("--num_rounds", type=int, default=5,
                        help="Number of federated learning rounds.")
    parser.add_argument("--config", type=str, default=None,
                        help="Optional path to a YAML configuration file.")
    args = parser.parse_args()

    # Load YAML config and override defaults if provided.
    if args.config:
        with open(args.config, "r") as file:
            config_yaml = yaml.safe_load(file)
        if "server_address" in config_yaml:
            args.server_address = config_yaml["server_address"]
        if "dataset_path" in config_yaml:
            args.dataset_path = config_yaml["dataset_path"]
        if "num_rounds" in config_yaml:
            args.num_rounds = config_yaml["num_rounds"]

    # Load test data and determine input dimensions.
    test_loader, input_dim = get_test_data(args.dataset_path)

    # Instantiate the global model.
    global_model = NeuralNetClassifier(
        input_dim=input_dim,
        hidden_units=[128, 64, 32],
        dropout_rate=0.3
    ).to(device)

    # Create the custom FedAvg strategy with intermediate evaluation.
    strategy = FedAvgWithIntermediateEval(
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
