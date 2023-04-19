import flwr as fl


def weighted_average(metrics):
    # Multiply accuracy of each client by number of examples used
    acc = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    results = {"accuracy": sum(acc) / sum(examples)}
    return results

if __name__ == "__main__":
    num_clients = 6
    strategy=fl.server.strategy.FedAvg(
                fraction_fit=0.9,
                fraction_evaluate=1,
                min_fit_clients=5,
                min_evaluate_clients=5,
                min_available_clients=int(
                    num_clients * 0.9
                ),
                evaluate_metrics_aggregation_fn=weighted_average,
            )
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=3),
        strategy=strategy,
    )

