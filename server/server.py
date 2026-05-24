import os
import sys
import grpc
from concurrent import futures
import pickle
import numpy as np
import threading
import logging
from sqlalchemy.orm import Session

# Add project root and proto dir to path for proto imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'proto'))
import proto.federated_pb2 as federated_pb2
import proto.federated_pb2_grpc as federated_pb2_grpc
from client.models import Autoencoder
from server.db import init_db, Metrics, RoundInfo

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FederatedServer(federated_pb2_grpc.FederatedLearningServicer):
    def __init__(self, num_clients=3):
        self.num_clients = num_clients
        self.current_round = 1

        self.registered_clients = set()
        self.client_weights = {}
        self.client_samples = {}

        # Initialize a global model
        # Using the same input_dim as client (29 features after dropping Time and Class)
        self.global_model = Autoencoder(input_dim=29)
        self.global_weights = self.global_model.get_weights()

        self.lock = threading.Lock()

        # Init DB
        try:
            self.db: Session = init_db()
            logging.info("Database connected successfully.")
        except Exception as e:
            logging.error(f"Failed to connect to DB: {e}")
            self.db = None

    def RegisterClient(self, request, context):
        with self.lock:
            self.registered_clients.add(request.client_id)
            logging.info(f"Client {request.client_id} registered. Total: {len(self.registered_clients)}")

        return federated_pb2.RegisterResponse(
            success=True,
            round_number=self.current_round
        )

    def GetGlobalModel(self, request, context):
        serialized_weights = pickle.dumps(self.global_weights)
        return federated_pb2.ModelResponse(
            round_number=self.current_round,
            weights=serialized_weights
        )

    def SendLocalModel(self, request, context):
        with self.lock:
            if request.round_number != self.current_round:
                logging.warning(f"Received model for round {request.round_number} from {request.client_id}, but current round is {self.current_round}")
                return federated_pb2.UpdateResponse(success=False)

            weights = pickle.loads(request.weights)
            self.client_weights[request.client_id] = weights
            self.client_samples[request.client_id] = request.num_samples

            logging.info(f"Received local model from {request.client_id} for round {self.current_round}")

            # Check if all clients have submitted
            if len(self.client_weights) >= self.num_clients:
                self.aggregate_weights()

        return federated_pb2.UpdateResponse(success=True)

    def aggregate_weights(self):
        logging.info(f"Aggregating weights for round {self.current_round}...")

        total_samples = sum(self.client_samples.values())

        new_global_weights = {}
        for client_id, weights in self.client_weights.items():
            weight_factor = self.client_samples[client_id] / total_samples
            for name, layer_weights in weights.items():
                if name not in new_global_weights:
                    new_global_weights[name] = np.zeros_like(layer_weights)
                new_global_weights[name] += layer_weights * weight_factor

        self.global_weights = new_global_weights

        # Log round info to DB
        if self.db:
            try:
                round_info = RoundInfo(
                    round_number=self.current_round,
                    clients_participated=len(self.client_weights)
                )
                self.db.add(round_info)
                self.db.commit()
            except Exception as e:
                logging.error(f"Failed to log round info: {e}")
                self.db.rollback()

        # Reset for next round
        self.client_weights = {}
        self.client_samples = {}
        self.current_round += 1

        logging.info(f"Moved to round {self.current_round}")

    def SendMetrics(self, request, context):
        logging.info(f"Received metrics from {request.client_id} for round {request.round_number}")

        if self.db:
            try:
                metrics = Metrics(
                    client_id=request.client_id,
                    round_number=request.round_number,
                    loss=request.loss,
                    roc_auc=request.roc_auc,
                    precision=request.precision,
                    recall=request.recall,
                    f1_score=request.f1_score,
                    drift_score=request.drift_score
                )
                self.db.add(metrics)
                self.db.commit()
            except Exception as e:
                logging.error(f"Failed to log metrics: {e}")
                self.db.rollback()

        return federated_pb2.UpdateResponse(success=True)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    # We expect 3 clients based on our data generation
    federated_pb2_grpc.add_FederatedLearningServicer_to_server(FederatedServer(num_clients=3), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    logging.info("Server started on port 50051")
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
