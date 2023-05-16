from concurrent import futures
from typing import List, Union, Tuple

import argparse
import fedlearn_grcp
import fedlearn_grpc_binds
import grpc
import numpy as np
import random
import tensorflow as tf
import uuid

uuid1 = uuid.uuid1()
client_id = str(uuid1.int >> 96)

parser = argparse.ArgumentParser()
parser.add_argument("--server", type=str, default="127.0.0.1:8080")
parser.add_argument("--address", type=str, default="127.0.0.1")
parser.add_argument("--port", type=str, default="50051")
args = parser.parse_args()


class FederatedLearningClient(fedlearn_grpc_binds.clientServicer):
    def __init__(self, ip: str, port: int, client_id: str):
        self.ip = ip
        self.port = port
        self.client_id = client_id

        self.model = self.define_model((28, 28, 1), 10)
        self.x_train = np.array([])
        self.y_train = np.array([])
        self.x_test = np.array([])
        self.y_test = np.array([])
        self.epochs = 0

        self.current_round = 0
        self.weights = np.array([])

        self.original_weight_shape = None

        self.dataset: tuple = tf.keras.datasets.mnist.load_data()
        self.x_train, self.y_train = self.dataset[0]
        self.x_test, self.y_test = self.dataset[1]
        self.x_train, self.y_train, self.x_test, self.y_test = self.normalize_dataset(
            self.x_train, self.y_train, self.x_test, self.y_test
        )
        self.start = 0
        self.end = 0

    def normalize_dataset(
        self,
        x_train,
        y_train,
        x_test,
        y_test,
    ) -> tuple:
        x_tr = (
            x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
            / 255.0
        )
        x_te = (
            x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1) / 255.0
        )

        y_tr = tf.one_hot(y_train.astype(np.int32), depth=10)
        y_te = tf.one_hot(y_test.astype(np.int32), depth=10)

        return x_tr, y_tr, x_te, y_te

    def TrainingStart(self, request, context):
        self.start = request.start
        self.end = request.end

        self.epochs = request.epochs

        self.fit()

        weights_list = self.model.get_weights()

        print(f"Client {self.client_id} - Weights: {weights_list}")

        self.original_weight_shape = weights_list[0].shape

        weights_bytes_list = [np.array(weights).tobytes() for weights in weights_list]

        num_samples = self.x_train[self.start : self.end].shape[0]

        return fedlearn_grcp.TrainingStartResponse(
            weights=weights_bytes_list, num_samples=num_samples
        )

    def ModelEvaluation(self, request, context):
        print("Received Evaluation Request")
        aggregated_weights = request.aggregated_weights

        original_shape = self.original_weight_shape

        received_weights_list = []
        for weights_bytes in aggregated_weights:
            weights_array = np.frombuffer(weights_bytes, dtype=np.float32)
            reshaped_weights = weights_array.reshape(original_shape)
            received_weights_list.append(reshaped_weights)
        

        accuracy = self.evaluate(received_weights_list)
        print(f"Client {self.client_id} - Accuracy: {accuracy}")
        return fedlearn_grcp.ModelEvaluationResponse(accuracy=accuracy)

    def fit(self):
        self.model.fit(
            self.x_train[self.start : self.end],
            self.y_train[self.start : self.end],
            epochs=self.epochs,
        )

    def evaluate(self, aggregated_weights):
        self.model.set_weights(aggregated_weights)
        return self.model.evaluate(self.x_test, self.y_test)[1]

    def define_model(self, input_shape, num_classes):
        model = tf.keras.models.Sequential()
        model.add(
            tf.keras.layers.Conv2D(
                32,
                kernel_size=(3, 3),
                activation="relu",
                kernel_initializer="he_uniform",
                input_shape=input_shape,
            )
        )
        model.add(tf.keras.layers.MaxPool2D((2, 2)))
        model.add(tf.keras.layers.Flatten())
        model.add(
            tf.keras.layers.Dense(
                100, activation="relu", kernel_initializer="he_uniform"
            )
        )
        model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))
        opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
        model.compile(
            optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"]
        )
        return model


def main() -> None:
    server = args.server
    address = args.address
    port = args.port

    with grpc.insecure_channel(server) as channel:
        stub = fedlearn_grpc_binds.apiStub(channel)
        print(f"Client {client_id} connected to server {server}")

        client = FederatedLearningClient(address, port, client_id)

        options = [
            ("grpc.max_receive_message_length", 500 * 1024 * 1024),  # 100 MB
        ]

        clientGrpc = grpc.server(
            futures.ThreadPoolExecutor(max_workers=2), options=options
        )
        fedlearn_grpc_binds.add_clientServicer_to_server(client, clientGrpc)
        clientGrpc.add_insecure_port(f"{address}:{port}")
        clientGrpc.start()

        response = stub.ClientRegister(
            fedlearn_grcp.ClientRegisterRequest(
                ip=client.ip, port=client.port, client_id=client.client_id
            )
        )

        confirmation_code = response.confirmation_code
        current_round = response.current_round

        client.current_round = current_round

        if confirmation_code != 0:
            return

        print(
            f"Client {client_id} registered with confirmation code {confirmation_code}"
        )

        print(f"Client {client_id} listening on {address}:{port}")
        clientGrpc.wait_for_termination()


if __name__ == "__main__":
    main()
