# FLWR (Flower) Federated Learning

This is a simple example of federated learning using FLWR (Flower), a Python framework for Federated Learning.

The project consists of a client and a server using the MNIST dataset.

The [client.py] file defines a machine learning model for handwritten digit classification using the MNIST dataset. This model is used as the basis for implementing the federated learning client, which is a subclass of the `NumPyClient` class from the FLWR library. The client implements the necessary methods for communicating with the federated learning server, including retrieving and updating model parameters, fitting the model on local data, and evaluating the model's performance.

The [server.py] file defines the federated learning server, which coordinates the communication between the clients and manages the aggregation of model parameters. It uses the FLWR FedAvg strategy to perform federated averaging of the model weights across clients, and it also defines a custom aggregation function to compute a weighted average of client accuracies. The server runs for a fixed number of rounds and records the history of the training process.

## Requirements

- Python >=3.10,<3.11
- Poetry (for package management)

## Installation

1. Install the dependencies using Poetry

```sh
poetry install
```

## Usage

1. Start the server

```sh
poetry run python server.py
```

2. Start up to 8 clients (in another terminal window)

```sh
poetry run python client.py
```

3. The server will run for 40 rounds of training and evaluation. You can monitor the progress in the server console.

4. After the training process is complete, the server will output the final metrics.
