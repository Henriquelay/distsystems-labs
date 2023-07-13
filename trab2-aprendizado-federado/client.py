import argparse
import os

import random
import threading
import time
import json
import uuid

from concurrent import futures
from collections import Counter
from typing import List, Union, Tuple
import paho.mqtt.client as mqtt

from client_trainer import ClientTrainer
from client_aggregator import ClientAggregator
from serialization import (
    serialize_weights,
    deserialize_weights,
    encode_base64,
    decode_base64,
)

from enum import Enum

import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime


class State(Enum):
    WAITING_CLIENTS = 1
    WAITING_ELECTION = 2
    AGGREGATOR = 3
    TRAINER = 4
    IDLE = 5
    EVALUATING = 6
    EXITING = 7


class Primordial:
    def __init__(
        self,
        mqtt_server: str,
        client_id,
        epochs,
        min_clients,
        max_clients,
        per_round_clients,
        max_rounds,
        accuracy_threshold,
    ):
        self.clients = [client_id]
        self.votes = []
        self.elected = None
        self.state = State.WAITING_CLIENTS

        self.client_id = client_id
        self.epochs = epochs
        self.min_clients = min_clients
        self.max_clients = max_clients
        self.per_round_clients = per_round_clients
        self.max_rounds = max_rounds
        self.accuracy_threshold = accuracy_threshold

        self.trainer = ClientTrainer(self.client_id)
        self.aggregator = ClientAggregator(
            self.epochs,
            self.min_clients,
            self.max_clients,
            self.per_round_clients,
            self.max_rounds,
            self.accuracy_threshold,
        )

        self.weigths_list = []
        self.num_samples_list = []
        self.chosen = []

        self.accuracy_list = []

        self.accuracy_history = []

        self.setup_broker_connection(mqtt_server)
        self.mqtt_client.loop_forever()

    def setup_broker_connection(self, mqtt_server):
        self.mqtt_client = mqtt.Client()
        broker_address, broker_port = mqtt_server.split(":")
        self.mqtt_client.connect(broker_address, int(broker_port), 3600)
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message

    def subscribe(self, topic):
        self.mqtt_client.subscribe(topic)
        print(f"Subscribed to topic '{topic}'")

    def elect_leader(self):
        random_client = random.choice(self.clients)
        self.mqtt_client.publish("fl/ElectionMsg", json.dumps({"id": random_client}))

    def handle_init_msg(self, msg):
        if msg["id"] == self.client_id:
            return

        if msg["id"] not in self.clients:
            self.clients.append(msg["id"])
            self.mqtt_client.publish("fl/InitMsg", json.dumps({"id": self.client_id}))

        if (
            len(self.clients) == self.min_clients
            and self.state == State.WAITING_CLIENTS
        ):
            self.state = State.WAITING_ELECTION
            self.elect_leader()

    def handle_election_msg(self, msg):
        self.votes.append(msg["id"])
        if self.state == State.WAITING_ELECTION:
            if len(self.votes) == len(self.clients):
                print(f"Votes: {self.votes}")

                self.elected = self.find_elected()
                self.votes = []

                if self.elected == self.client_id:
                    self.state = State.AGGREGATOR
                    print("I am the aggregator")
                    clients = self.clients.copy()
                    clients.remove(self.client_id)
                    self.aggregator.load_clients(clients)
                    self.aggregator_job()
                else:
                    self.state = State.IDLE
                    print(f"I am idle, waiting for {self.elected}")

    def aggregator_job(self):
        self.chosen = self.aggregator.choose_clients()
        self.aggregator.new_round()
        self.mqtt_client.publish("fl/TrainingMsg", json.dumps({"clients": self.chosen}))

    def trainer_job(self, left, right, epochs, actual_round):
        weights, num_samples = self.trainer.train(left, right, epochs, actual_round)
        print("Trained, sending weights to aggregator")

        serialized_weights = serialize_weights(weights)

        encoded_weights = encode_base64(serialized_weights)

        self.mqtt_client.publish(
            "fl/RoundMsg",
            json.dumps({"weights": encoded_weights, "samples": num_samples}),
        )

        print("Sent, waiting for aggregation to evaluate")

        self.state = State.IDLE

    def handle_round_msg(self, msg):
        if self.state == State.AGGREGATOR:
            encoded_weights = msg["weights"]
            serialized_weights = decode_base64(encoded_weights)
            num_samples = msg["samples"]
            desserialized_weights = deserialize_weights(serialized_weights)

            self.weigths_list.append(desserialized_weights)
            self.num_samples_list.append(num_samples)
            print(
                f"Weights received, current status: {len(self.weigths_list)}/{len(self.chosen)}"
            )

            if len(self.weigths_list) == len(self.chosen):
                weighted_average = self.aggregator.federated_average(
                    self.weigths_list, self.num_samples_list
                )
                serialized_weigthted_average = serialize_weights(weighted_average)
                encoded_weigthted_average = encode_base64(serialized_weigthted_average)

                self.mqtt_client.publish(
                    "fl/AggregationMsg",
                    json.dumps({"weights": encoded_weigthted_average}),
                )
                self.weigths_list = []
                self.num_samples_list = []
        else:
            print("Ignoring round message, I am idle")

    def handle_training_msg(self, msg):
        if self.state == State.IDLE:
            clients = msg["clients"]
            if self.client_id in clients:
                self.state = State.TRAINER
                index = clients.index(self.client_id)
                print("I am a trainer at index", index)
                edges = self.aggregator.split_dataset(clients)[index]
                print(f"I will train on edges {edges}")
                actual_round = self.aggregator.get_round()
                self.trainer_job(edges[0], edges[1], self.epochs, actual_round)
            else:
                print("I am idle, but not a trainer, waiting for evaluation call")

    def handle_aggregation_msg(self, msg):
        if self.state == State.IDLE:
            print("I am evaluator, evaluating")
            self.state = State.EVALUATING
            encoded_weights = msg["weights"]
            serialized_weights = decode_base64(encoded_weights)
            self.evaluation_job(serialized_weights)

    def handle_evaluation_msg(self, msg):
        if self.state == State.AGGREGATOR:
            accuracy = msg["accuracy"]
            self.accuracy_list.append(accuracy)
            if len(self.accuracy_list) == len(self.clients) - 1:
                mean_accuracy = sum(self.accuracy_list) / len(self.accuracy_list)

                print(f"Mean accuracy: {mean_accuracy}")

                self.accuracy_history.append(mean_accuracy)

                if self.aggregator.has_reached_max_rounds():
                    self.mqtt_client.publish(
                        "fl/FinishMsg", json.dumps({"id": self.client_id})
                    )
                    print("Reached max rounds, stopping")
                    return

                if self.aggregator.has_reached_accuracy_threshold(mean_accuracy):
                    self.mqtt_client.publish(
                        "fl/FinishMsg", json.dumps({"id": self.client_id})
                    )
                    print("Reached accuracy threshold, stopping")
                    return

                clients = self.clients.copy()
                clients.remove(self.client_id)
                self.aggregator.load_clients(clients)
                self.accuracy_list = []
                self.aggregator_job()

    def handle_finish_msg(self, msg):
        if self.state == State.AGGREGATOR:
            df = pd.DataFrame(
                {
                    "Acurácia": self.accuracy_history,
                    "Round": range(1, len(self.accuracy_history) + 1),
                }
            )

            df.plot(x="Round", y="Acurácia", marker="o")
            plt.xlabel("Round")
            plt.ylabel("Acurácia")
            plt.title(
                f"Acurácia por round - {self.aggregator.get_round()} rounds - {self.accuracy_threshold * 100}% de threshold de acurácia"
            )

            current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M")
            nome_arquivo = f"grafico_{current_datetime}.png"
            plt.savefig(nome_arquivo)

        id = msg["id"]
        print(f"Client {id} sent finish message")
        self.state = State.EXITING
        print("Exiting")
        exit(0)

    def evaluation_job(self, serialized_weights):
        accuracy = self.trainer.evaluate_aggregated(serialized_weights)
        print(f"Evaluated, accuracy: {accuracy}")
        self.mqtt_client.publish("fl/EvaluationMsg", json.dumps({"accuracy": accuracy}))
        self.state = State.IDLE
        print(f"I am idle, waiting for {self.elected}")

    def find_elected(self):
        counts = Counter(self.votes)
        max_freq = max(counts.values())
        most_frequent = [
            element for element, count in counts.items() if count == max_freq
        ]
        most_frequent.sort()
        return most_frequent[-1]

    def on_message(self, client, userdata, msg):
        desserialized_msg = json.loads(msg.payload)
        if msg.topic == "fl/InitMsg":
            self.handle_init_msg(desserialized_msg)
        if msg.topic == "fl/ElectionMsg":
            self.handle_election_msg(desserialized_msg)
        if msg.topic == "fl/TrainingMsg":
            self.handle_training_msg(desserialized_msg)
        if msg.topic == "fl/RoundMsg":
            self.handle_round_msg(desserialized_msg)
        if msg.topic == "fl/AggregationMsg":
            self.handle_aggregation_msg(desserialized_msg)
        if msg.topic == "fl/EvaluationMsg":
            self.handle_evaluation_msg(desserialized_msg)
        if msg.topic == "fl/FinishMsg":
            self.handle_finish_msg(desserialized_msg)

    def on_connect(self, client, userdata, flags, rc):
        if State.WAITING_CLIENTS:
            self.subscribe("fl/InitMsg")
            self.subscribe("fl/ElectionMsg")
            self.subscribe("fl/TrainingMsg")
            self.subscribe("fl/RoundMsg")
            self.subscribe("fl/AggregationMsg")
            self.subscribe("fl/EvaluationMsg")
            self.subscribe("fl/FinishMsg")
            self.mqtt_client.publish("fl/InitMsg", json.dumps({"id": self.client_id}))


def main() -> None:
    uuid1 = uuid.uuid1()
    client_id: str = str(uuid1.int >> 96)

    parser = argparse.ArgumentParser()

    parser.add_argument("--mqttserver", type=str, default="localhost:1883")
    args = parser.parse_args()

    mqtt_server = args.mqttserver
    epochs = 1
    per_round_clients = 3
    min_clients = 5
    max_clients = 8
    max_rounds = 30
    accuracy_threshold = 0.995

    primordial = Primordial(
        mqtt_server,
        client_id,
        epochs,
        min_clients,
        max_clients,
        per_round_clients,
        max_rounds,
        accuracy_threshold,
    )


if __name__ == "__main__":
    main()
