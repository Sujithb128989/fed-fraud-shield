import os
import sys
import grpc
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp
import pickle
import time
import logging

# Add project root and proto dir to path for proto imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'proto'))
import proto.federated_pb2 as federated_pb2
import proto.federated_pb2_grpc as federated_pb2_grpc
from client.models import Autoencoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FederatedClient:
    def __init__(self, client_id, server_address="localhost:50051", data_dir="data/clients"):
        self.client_id = client_id
        self.server_address = server_address
        self.data_dir = os.path.join(data_dir, client_id)

        self.channel = grpc.insecure_channel(self.server_address)
        self.stub = federated_pb2_grpc.FederatedLearningStub(self.channel)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler = StandardScaler()
        self.load_data()

        self.input_dim = self.X_train.shape[1]
        self.autoencoder = Autoencoder(self.input_dim).to(self.device)
        self.isolation_forest = IsolationForest(n_estimators=100, contamination=0.01, random_state=42)

        self.round_number = 0

    def load_data(self):
        train_df = pd.read_csv(f"{self.data_dir}/train.csv")
        test_df = pd.read_csv(f"{self.data_dir}/test.csv")

        # Isolation Forest is unsupervised, Autoencoder is trained on normal data ideally,
        # but for this example we train on all and use reconstruction error
        self.X_train = train_df.drop(['Class', 'Time'], axis=1).values
        self.y_train = train_df['Class'].values

        self.X_test = test_df.drop(['Class', 'Time'], axis=1).values
        self.y_test = test_df['Class'].values

        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)

        self.train_tensor = torch.tensor(self.X_train, dtype=torch.float32).to(self.device)
        self.test_tensor = torch.tensor(self.X_test, dtype=torch.float32).to(self.device)

    def register(self):
        logging.info(f"Registering client {self.client_id} with server...")
        try:
            req = federated_pb2.RegisterRequest(client_id=self.client_id)
            resp = self.stub.RegisterClient(req)
            if resp.success:
                self.round_number = resp.round_number
                logging.info(f"Successfully registered. Current round: {self.round_number}")
                return True
            return False
        except Exception as e:
            logging.error(f"Failed to register: {e}")
            return False

    def get_global_model(self):
        logging.info(f"Requesting global model...")
        try:
            req = federated_pb2.GetModelRequest(client_id=self.client_id)
            resp = self.stub.GetGlobalModel(req)
            self.round_number = resp.round_number
            if resp.weights:
                weights = pickle.loads(resp.weights)
                self.autoencoder.set_weights(weights)
                logging.info(f"Loaded global model for round {self.round_number}")
            return True
        except Exception as e:
            logging.error(f"Failed to get global model: {e}")
            return False

    def train_autoencoder(self, epochs=5, batch_size=64):
        logging.info(f"Training Autoencoder locally for round {self.round_number}...")
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=0.001)

        dataset = torch.utils.data.TensorDataset(self.train_tensor, self.train_tensor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        self.autoencoder.train()
        for epoch in range(epochs):
            total_loss = 0
            for batch_x, _ in dataloader:
                optimizer.zero_grad()
                output = self.autoencoder(batch_x)
                loss = criterion(output, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            if (epoch+1) % max(1, epochs//2) == 0:
                logging.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")

        return total_loss / len(dataloader)

    def train_isolation_forest(self):
        logging.info("Training local Isolation Forest...")
        self.isolation_forest.fit(self.X_train)

    def send_local_model(self):
        logging.info("Sending local model weights to server...")
        try:
            weights = self.autoencoder.get_weights()
            serialized_weights = pickle.dumps(weights)

            req = federated_pb2.LocalModelRequest(
                client_id=self.client_id,
                round_number=self.round_number,
                weights=serialized_weights,
                num_samples=len(self.X_train)
            )
            resp = self.stub.SendLocalModel(req)
            return resp.success
        except Exception as e:
            logging.error(f"Failed to send local model: {e}")
            return False

    def evaluate(self, train_loss):
        logging.info("Evaluating ensemble model...")
        self.autoencoder.eval()
        with torch.no_grad():
            reconstructions = self.autoencoder(self.test_tensor)
            mse = torch.mean(torch.pow(self.test_tensor - reconstructions, 2), dim=1).cpu().numpy()

        # Normalize MSE
        mse_norm = (mse - mse.min()) / (mse.max() - mse.min())

        # IF predictions (1 for inliers, -1 for outliers -> map to 0 and 1)
        if_preds = self.isolation_forest.predict(self.X_test)
        if_scores = -self.isolation_forest.score_samples(self.X_test) # Higher is more anomalous
        if_scores_norm = (if_scores - if_scores.min()) / (if_scores.max() - if_scores.min())

        # Ensemble score: average of normalized autoencoder MSE and normalized IF anomaly score
        ensemble_scores = 0.5 * mse_norm + 0.5 * if_scores_norm

        # Threshold for evaluation (can be tuned)
        preds = (ensemble_scores > 0.5).astype(int)

        roc_auc = roc_auc_score(self.y_test, ensemble_scores)
        precision, recall, f1, _ = precision_recall_fscore_support(self.y_test, preds, average='binary', zero_division=0)

        # Calculate drift score (Kolmogorov-Smirnov test on a feature, e.g., first feature or MSE distributions)
        # Here we compare train and test reconstruction errors as a proxy for drift
        with torch.no_grad():
            train_reconstructions = self.autoencoder(self.train_tensor)
            train_mse = torch.mean(torch.pow(self.train_tensor - train_reconstructions, 2), dim=1).cpu().numpy()

        statistic, pvalue = ks_2samp(train_mse, mse)
        drift_score = statistic # Use KS statistic as drift score

        logging.info(f"Evaluation metrics - ROC AUC: {roc_auc:.4f}, F1: {f1:.4f}, Drift: {drift_score:.4f}")

        # Explain anomalies
        self.explain_anomalies(preds, ensemble_scores)

        return train_loss, roc_auc, precision, recall, f1, drift_score

    def explain_anomalies(self, preds, scores):
        # Find indices of anomalies
        anomaly_indices = np.where(preds == 1)[0]
        if len(anomaly_indices) == 0:
            return

        # Limit to top 5 anomalies for explainability to save time
        top_indices = anomaly_indices[np.argsort(scores[anomaly_indices])[-5:]]

        background = self.X_train[:100]
        # Use a wrapper for Autoencoder MSE to use with SHAP
        def ae_mse_wrapper(X):
            x_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                rec = self.autoencoder(x_tensor)
                err = torch.mean(torch.pow(x_tensor - rec, 2), dim=1).cpu().numpy()
            return err

        explainer = shap.KernelExplainer(ae_mse_wrapper, background)
        shap_values = explainer.shap_values(self.X_test[top_indices])

        # Save explanations (e.g., to disk for dashboard)
        os.makedirs("data/explanations", exist_ok=True)
        with open(f"data/explanations/{self.client_id}_round_{self.round_number}_shap.pkl", "wb") as f:
            pickle.dump({'indices': top_indices, 'shap_values': shap_values, 'data': self.X_test[top_indices]}, f)

    def send_metrics(self, loss, roc_auc, precision, recall, f1, drift_score):
        try:
            req = federated_pb2.MetricsRequest(
                client_id=self.client_id,
                round_number=self.round_number,
                loss=loss,
                roc_auc=roc_auc,
                precision=precision,
                recall=recall,
                f1_score=f1,
                drift_score=drift_score
            )
            resp = self.stub.SendMetrics(req)
            return resp.success
        except Exception as e:
            logging.error(f"Failed to send metrics: {e}")
            return False

    def run_federated_loop(self, num_rounds=5):
        # Train IF once per client locally (non-federated part)
        self.train_isolation_forest()

        for r in range(num_rounds):
            logging.info(f"\n--- Starting Round {r+1}/{num_rounds} ---")
            # Wait a bit before registering to ensure server is ready or processing previous rounds
            time.sleep(2)

            while not self.register():
                time.sleep(5)

            self.get_global_model()

            loss = self.train_autoencoder(epochs=3)

            self.send_local_model()

            loss, roc_auc, precision, recall, f1, drift = self.evaluate(loss)

            self.send_metrics(loss, roc_auc, precision, recall, f1, drift)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--client_id", type=str, required=True)
    parser.add_argument("--server", type=str, default="localhost:50051")
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()

    client = FederatedClient(args.client_id, args.server)
    client.run_federated_loop(args.rounds)
