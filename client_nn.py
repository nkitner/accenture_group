import argparse
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MaxAbsScaler
from flwr.common.logger import log
from flwr.common.typing import Config
from fl4health.clients.tabular_data_client import TabularDataClient
from fl4health.utils.metrics import Accuracy
import flwr as fl

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
        self.out = nn.Linear(hidden_units[2], 1)  # Single output neuron for binary classification

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.out(x)
        return x

class MyTabularDataClient(TabularDataClient):
    def get_data_frame(self, config: Config) -> pd.DataFrame:
        df = pd.read_csv(self.data_path)
        df.drop(columns=["Unnamed: 0"], inplace=True)
        return df

    def get_model(self, config: Config) -> nn.Module:
        model = NeuralNetClassifier(self.input_dimension, hidden_units=[128, 64, 32], dropout_rate=0.3)
        model.to(self.device)
        return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FL Client Main")
    parser.add_argument(
        "--dataset_path",
        action="store",
        type=str,
        help="Path to the local dataset",
        default="/BreastCancerDataRoystonAltman_subset_A.csv"
    )
    parser.add_argument(
        "--server_address",
        action="store",
        type=str,
        help="Server Address for the clients to communicate with the server through",
        default="0.0.0.0:8080",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(INFO, f"Device to be used: {device}")
    log(INFO, f"Server Address: {args.server_address}")

    data_path = Path(args.dataset_path)
    
    client = MyTabularDataClient(data_path, [Accuracy("accuracy")], device, "pid", ["status"])
    fl.client.start_client(server_address=args.server_address, client=client.to_client())
