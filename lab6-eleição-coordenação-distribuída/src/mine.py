"""Solving the challenge of a transaction"""

from random import randint
from hashlib import sha1
from time import sleep
from json import dumps

from paho.mqtt.client import Client as mqttClient

from transactions import ChallengeTransaction

def try_generate_solution(challenge: int) -> str:
    num1 = randint(1, 1000 * challenge)
    num2 = randint(1, 1000 * challenge)

    hash = sha1()
    hash.update(str(num1 + num2).encode("utf-8"))
    return hash.hexdigest()


def mine(client, challenge: ChallengeTransaction) -> None:
    print(f"Desafio da transação {challenge.transaction_id}: challenge {challenge.challenge}")
    print("Minerando...")
    while not transaction_has_winner(challenge.transaction_id):
        sleep(0.01)
        solution = try_generate_solution(challenge.challenge)
        client.publish(
            "sd/solution",
            dumps(
                {
                    "clientId": my_client_id,
                    "transactionId": transaction_id,
                    "solution": solution,
                }
            ),
        )
