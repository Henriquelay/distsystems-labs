import numpy as np
import tensorflow as tf
import random

from serialization import serialize_weights, deserialize_weights


class ClientTrainer:
    def __init__(self, client_id: str):
        self.client_id: str = client_id
        self.setup_defaults()
        self.actual_round: int = 0
        # self.thread = threading.Thread(target=self.start)

    def setup_defaults(self) -> None:
        self.model = tf.keras.models.Sequential()
        self.model.add(
            tf.keras.layers.Conv2D(
                32,
                kernel_size=(3, 3),
                activation="relu",
                kernel_initializer="he_uniform",
                input_shape=(28, 28, 1),
            )
        )
        self.model.add(tf.keras.layers.MaxPool2D((2, 2)))
        self.model.add(tf.keras.layers.Flatten())
        self.model.add(
            tf.keras.layers.Dense(
                100, activation="relu", kernel_initializer="he_uniform"
            )
        )
        self.model.add(tf.keras.layers.Dense(10, activation="softmax"))
        opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
        self.model.compile(
            optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"]
        )
        self.weights = np.array([])
        self.epochs = 0
        self.dataset: tuple = tf.keras.datasets.mnist.load_data()
        self.x_train: np.ndarray = self.dataset[0][0]
        self.y_train: np.ndarray = self.dataset[0][1]
        self.x_test: np.ndarray = self.dataset[1][0]
        self.y_test: np.ndarray = self.dataset[1][1]

        self.x_train = (
            self.x_train.reshape(
                self.x_train.shape[0], self.x_train.shape[1], self.x_train.shape[2], 1
            )
            / 255.0
        )
        self.x_test = (
            self.x_test.reshape(
                self.x_test.shape[0], self.x_test.shape[1], self.x_test.shape[2], 1
            )
            / 255.0
        )

        self.y_train = tf.one_hot(self.y_train.astype(np.int32), depth=10)
        self.y_test = tf.one_hot(self.y_test.astype(np.int32), depth=10)
        self.start = 0
        self.end = len(self.x_train)

    def evaluate_aggregated(self, aggregated_weights):
        deserialized_weights = deserialize_weights(aggregated_weights)
        accuracy = self.evaluate(deserialized_weights)
        return accuracy

    def fit(self):
        self.model.fit(
            self.x_train[self.start : self.end],
            self.y_train[self.start : self.end],
            epochs=self.epochs,
        )

    def evaluate(self, weights):
        self.model.set_weights(weights)
        dataset_size = len(self.x_test)
        sample_size = int(0.6 * dataset_size)
        left = random.randint(0, dataset_size - sample_size)
        right = left + sample_size
        _, accuracy = self.model.evaluate(self.x_test[left:right], self.y_test[left:right])
        return accuracy

    def train(self, left, right, epochs, round):
        self.start = left
        self.end = right
        self.epochs = epochs
        self.actual_round = round
        self.fit()
        weights = self.model.get_weights()
        num_samples = right - left
        return weights, num_samples
