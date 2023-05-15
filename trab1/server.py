from concurrent import futures
from typing import List, Union, Tuple

import fedlearn_grpc_pb2
import fedlearn_grpc_pb2_grpc
import grpc
import numpy as np
import random
import tensorflow as tf
import threading
import time


class Client:
    def __init__(
        self, ip: str, port: int, client_id: str, server_address: str, epochs: int
    ):
        self.ip: str = ip
        self.port: int = port
        self.client_id: str = client_id
        self.start: int = 0
        self.end: int = 0
        self.epochs: int = epochs
        self.server_address: str = server_address
        self.stub = None
        self.thread = threading.Thread(target=self.start_stub)
        self.connect()

    def start_stub(self):
        channel = grpc.insecure_channel(self.ip + ":" + str(self.port))
        self.stub = fedlearn_grpc_pb2_grpc.clientStub(channel)

    def trainingStart(self):
        request = self.stub.TrainingStart(
            fedlearn_grpc_pb2.TrainingStartRequest(
                start=self.start,
                end=self.end,
                epochs=self.epochs,
            )
        )
        response = self.stub.TrainingStart(request)
        weights_bytes_list = response.weights

        received_weights_list = []
        for weights_bytes in weights_bytes_list:
            weights_array = np.frombuffer(weights_bytes, dtype=np.float32)
            received_weights_list.append(weights_array)

        return received_weights_list, response.num_samples

    def modelEvaluation(self, aggregated_weights):
        try:
            aggregated_weights_bytes = [weight.flatten().tobytes() for weight in aggregated_weights]

            request = self.stub.ModelEvaluation(
                fedlearn_grpc_pb2.ModelEvaluationRequest(
                    aggregated_weights=aggregated_weights_bytes
                )
            )
            response = self.stub.ModelEvaluation(request)
            return response.accuracy
        except Exception as e:
            print(e)

    def connect(self):
        self.thread.start()


class FederatedLearningServer(fedlearn_grpc_pb2_grpc.apiServicer):
    def __init__(
        self,
        epochs: int,
        min_clients: int,
        max_clients: int,
        per_round_clients: int,
        max_rounds: int,
        accuracy_threshold: float,
    ) -> None:
        if max_clients < min_clients:
            raise ValueError(
                "max_clients must be greater than or equal to min_clients."
            )

        self.epochs: int = epochs
        self.min_clients: int = min_clients
        self.max_clients: int = max_clients
        self.per_round_clients: int = per_round_clients
        self.max_rounds: int = max_rounds
        self.accuracy_threshold: float = accuracy_threshold

        self.clients: List[Client] = []
        self.dataset: tuple = tf.keras.datasets.mnist.load_data()
        self.current_round: int = 0
        self.rpc = fedlearn_grpc_pb2
        self.last_aggregated_weights = np.array([])

        self.main_thread = threading.Thread(target=self.run)
        self.main_thread.start()

    def run(self) -> None:
        while True:
            if self.has_sufficient_clients():
                round_clients = self.choose_clients()
                self.split_dataset(self.clients)

                threads = []
                results = self.clients_work_now(round_clients)
                x, y = zip(*results)

                print(y)

                self.last_aggregated_weights = self.federated_average(x, y)

                results = self.clients_evaluate_now(
                    self.clients, self.last_aggregated_weights
                )

                average_accuracy = np.mean(results)

                if average_accuracy >= self.accuracy_threshold:
                    print("Accuracy threshold reached.")
                    break

                self.current_round += 1

                if self.current_round >= self.max_rounds:
                    print("Max rounds reached.")
                    break

            time.sleep(0.1)

    def clients_evaluate_now(self, clients, weights) -> List[int]:
        threads = []
        results = []

        for client in clients:
            thread = threading.Thread(
                target=self.execute_model_evaluation, args=(client, weights, results)
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        return results

    def clients_work_now(self, clients):
        threads = []
        results = []

        for client in clients:
            thread = threading.Thread(
                target=self.execute_training_start, args=(client, results)
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        print("All clients finished training.")
        print(results)
        return results

    def execute_training_start(self, client: Client, results: List):
        weights, num_samples = client.trainingStart()
        print(f"Client {client.client_id} finished training.")
        results.append((weights, num_samples))

    def execute_model_evaluation(
        self, client: Client, weights: np.ndarray, results: List
    ):
        accuracy = client.modelEvaluation(weights)
        results.append(accuracy)

    def federated_average(self, weights, num_samples):
        return weights[0]

    def has_sufficient_clients(self) -> bool:
        return len(self.clients) >= self.min_clients

    def ClientRegister(self, request, context):
        ip: str = request.ip
        port: int = request.port
        client_id: str = request.client_id

        confirmation_code: int = self.add_client(ip, port, client_id)
        current_round: int = self.current_round

        return self.rpc.ClientRegisterResponse(
            confirmation_code=confirmation_code, current_round=current_round
        )

    def choose_clients(self) -> List[Client]:
        if len(self.clients) < self.per_round_clients:
            raise ValueError("There are not enough clients to choose from.")

        chosen_clients = random.sample(self.clients, self.per_round_clients)
        return chosen_clients

    def add_client(self, ip: str, port: int, client_id: str) -> int:
        if len(self.clients) >= self.max_clients:
            return -1

        client = Client(ip, port, client_id, "localhost:8080", self.epochs)
        self.clients.append(client)
        print(f"Added client {client_id}.")
        return 0

    def split_dataset(self, clients) -> None:
        (x_train, y_train), (x_test, y_test) = self.dataset
        num_clients: int = len(clients)

        chunk_size = len(x_train) // num_clients

        for i, client in enumerate(clients):
            start = i * chunk_size
            end = (i + 1) * chunk_size

            client.start = start
            client.end = end


def serve():
    options = [
        ("grpc.max_receive_message_length", 500 * 1024 * 1024),  # 100 MB
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4), options=options)
    fedlearn_grpc_pb2_grpc.add_apiServicer_to_server(
        FederatedLearningServer(
            epochs=1,
            min_clients=2,
            max_clients=4,
            per_round_clients=2,
            max_rounds=5,
            accuracy_threshold=90,
        ),
        server,
    )
    server.add_insecure_port("[::]:8080")
    server.start()
    server.wait_for_termination()


serve()
