"""Solving the challenge of a transaction"""

import random
import hashlib
import time
import json

from transactions import transaction_has_winner, my_client_id

from paho.mqtt.client import Client as mqttClient

def try_generate_solution(challenge: int) -> str:
    num1 = random.randint(1, 1000 * challenge)
    num2 = random.randint(1, 1000 * challenge)

    hash = hashlib.sha1()
    hash.update(str(num1 + num2).encode("utf-8"))
    return hash.hexdigest()


def mine(client: mqttClient, transaction_id: int, challenge: int) -> None:
    print(f"Desafio da transação {transaction_id}: challenge {challenge}")
    print("Minerando...")
    while not transaction_has_winner(transaction_id):
        time.sleep(0.01)
        solution = try_generate_solution(challenge)
        client.publish(
            "sd/solution",
            json.dumps(
                {
                    "clientId": my_client_id,
                    "transactionId": transaction_id,
                    "solution": solution,
                }
            ),
        )
