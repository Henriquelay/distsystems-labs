from concurrent import futures
from typing import List, Union, Tuple
import pickle
import grpc
import numpy as np
import random
import tensorflow as tf
import threading
import time
import matplotlib.pyplot as plt
import os
import warnings

import fedlearn_grpc_binds
import fedlearn_grpc

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
tf.get_logger().setLevel("ERROR")
warnings.filterwarnings("ignore")

accuracies = []


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
        self.stub = fedlearn_grpc_binds.clientStub(channel)

    def serialize_weights(self, weights):
        serialized_weights = pickle.dumps(weights)
        return serialized_weights

    def deserialize_weights(self, serialized_weights):
        weights = pickle.loads(serialized_weights)
        return weights

    def trainingStart(self):
        request = self.stub.TrainingStart(
            fedlearn_grpc.TrainingStartRequest(
                start=self.start,
                end=self.end,
                epochs=self.epochs,
            )
        )
        response = self.stub.TrainingStart(request)
        weights_bytes_list = response.weights

        received_weights_list = self.deserialize_weights(weights_bytes_list[0])

        return received_weights_list, response.num_samples

    def modelEvaluation(self, weights: List[np.ndarray]):
        try:
            aggregated_weights_bytes = [self.serialize_weights(weights)]

            print("Sending evaluation request to client: ", self.client_id)

            request = fedlearn_grpc.ModelEvaluationRequest(
                aggregated_weights=aggregated_weights_bytes
            )

            print("Received evaluation response from client: ", self.client_id)

            response = self.stub.ModelEvaluation(request)

            return response.accuracy
        except Exception as e:
            exit()

    def connect(self):
        self.thread.start()


class FederatedLearningServer(fedlearn_grpc_binds.apiServicer):
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
        self.rpc = fedlearn_grpc
        self.last_aggregated_weights = np.array([])

        self.main_thread = threading.Thread(target=self.train)

    def train(self) -> None:
        while True:
            if self.has_sufficient_clients():
                print(f"Current round: {self.current_round + 1}")
                self.split_dataset(self.clients)

                results = self.clients_work_now()
                x, y = zip(*results)

                self.last_aggregated_weights = self.federated_average(x, y)

                results = self.clients_evaluate_now()
                print("Evaluation results: ", results)

                average_accuracy = np.mean(results)
                print(f"Average accuracy: {average_accuracy}")

                # append the accuracy to global accuracy list
                accuracies.append(average_accuracy)

                if average_accuracy >= self.accuracy_threshold:
                    print("Accuracy threshold reached.")
                    return

                self.current_round += 1

                if self.current_round >= self.max_rounds:
                    print("Max rounds reached.")
                    return

            time.sleep(0.1)

    def clients_evaluate_now(self) -> List[int]:
        threads = []
        results = []

        for client in self.clients:
            thread = threading.Thread(
                target=self.execute_model_evaluation,
                args=(client, results, self.last_aggregated_weights),
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        return results

    def clients_work_now(self):
        threads = []
        results = []

        for client in self.choose_clients():
            thread = threading.Thread(
                target=self.execute_training_start, args=(client, results)
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        print("All clients finished training.")

        return results

    def execute_training_start(self, client: Client, results: List):
        weights, num_samples = client.trainingStart()
        print(f"Client {client.client_id} finished training.")
        results.append((weights, num_samples))

    def execute_model_evaluation(self, client: Client, results: List, weights):
        accuracy = client.modelEvaluation(weights)
        results.append(accuracy)

    def federated_average(self, weights_list, num_samples_list):
        weighted_average = np.average(
            weights_list, axis=0, weights=num_samples_list
        ).astype(object)

        return weighted_average

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
        print(f"Chosen clients: {[client.client_id for client in chosen_clients]}")
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
    aggregator = FederatedLearningServer(
            epochs=1,
            min_clients=4,
            max_clients=5,
            per_round_clients=2,
            max_rounds=5,
            accuracy_threshold=0.99,
        )
    fedlearn_grpc_binds.add_apiServicer_to_server(
        aggregator,
        server,
    )
    server.add_insecure_port("[::]:8080")
    aggregator.main_thread.start()
    server.start()
    aggregator.main_thread.join()
    server.stop(3.0)

    # plot global accuracy list
    plt.plot(accuracies)
    plt.xlabel("Round")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Round using Federated Averaging")
    plt.show()


if __name__ == "__main__":
    serve()
