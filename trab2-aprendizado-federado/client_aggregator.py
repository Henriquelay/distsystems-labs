import random
import numpy as np
from typing import List, Union, Tuple
from serialization import serialize_weights, deserialize_weights
import tensorflow as tf

class ClientAggregator:
    def __init__(self, epochs: int, min_clients: int, max_clients: int, per_round_clients: int, max_rounds: int, accuracy_threshold: float):
        self.epochs: int = epochs
        self.min_clients: int = min_clients
        self.max_clients: int = max_clients
        self.per_round_clients: int = per_round_clients
        self.max_rounds: int = max_rounds
        self.accuracy_threshold: float = accuracy_threshold

        self.clients: List = []
        self.dataset: tuple = tf.keras.datasets.mnist.load_data()

    def load_clients(self, clients: List):
        self.clients = clients
    
    def has_sufficient_clients(self) -> bool:
        return len(self.clients) >= self.min_clients

    def federated_average(self, weights_list, num_samples_list):
        weighted_average = np.average(
            weights_list, axis=0, weights=num_samples_list
        ).astype(object)

        return weighted_average

    def choose_clients(self):
        if len(self.clients) < self.per_round_clients:
            raise ValueError("Not enough clients to choose from.")

        chosen = random.sample(self.clients, self.per_round_clients)
        print(f"Chosen clients: {[client for client in chosen]}")
        return chosen
    
    def split_dataset(self, clients):
        (x_train, y_train), (x_test, y_test) = self.dataset
        num_clients: int = len(clients)

        chunk_size = len(x_train) // num_clients

        edges = []

        for i, client in enumerate(clients):
            left = i * chunk_size
            right = (i + 1) * chunk_size

            edges.append((left, right))

        return edges
    