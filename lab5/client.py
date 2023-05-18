import random
import hashlib
import uuid
import json
import threading
import time

uuid1 = uuid.uuid1()
client_id = (uuid1.int >> 96) & 0xFFFF

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--server", default="localhost:1883")
args = parser.parse_args()

import paho.mqtt.client as mqtt


def get_int_input(prompt: str) -> int:
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Entrada inválida, tente novamente.")


def try_generate_solution(challenge) -> str:
    num1 = random.randint(1, 1000 * challenge)
    num2 = random.randint(1, 1000 * challenge)

    hash = hashlib.sha1()
    hash.update(str(num1 + num2).encode("utf-8"))
    return hash.hexdigest()


def mine(client, transaction_id, challenge) -> None:
    print(f"Desafio da transação {transaction_id}: challenge {challenge}")
    print("Minerando...")
    while not transaction_has_winner(transaction_id):
        time.sleep(0.01)
        solution = try_generate_solution(challenge)
        client.publish(
            "sd/solution",
            json.dumps(
                {
                    "clientId": client_id,
                    "transactionId": transaction_id,
                    "solution": solution,
                }
            ),
        )


transactions = []


def add_transaction(id, challenge, solution, winner):
    transactions.append(
        {
            "transactionId": id,
            "challenge": challenge,
            "solution": solution,
            "winner": winner,
        }
    )


def transaction_has_winner(transaction_id) -> bool:
    transaction = next(t for t in transactions if t["transactionId"] == transaction_id)
    return transaction["winner"] != -1


def on_connect(client, userdata, flags, rc):
    print("Connected with result code " + str(rc))
    client.subscribe("sd/challenge")
    client.subscribe("sd/result")
    print("Subscribed to topic 'sd/challenge'")
    print("Subscribed to topic 'sd/result'")


def handle_challenge(client, userdata, msg):
    print(f"challenge received: {msg.payload}")
    transaction = json.loads(msg.payload)
    add_transaction(transaction["transactionId"], transaction["challenge"], "", -1)
    mining_thread = threading.Thread(
        target=mine,
        args=(client, transaction["transactionId"], transaction["challenge"]),
    )
    mining_thread.start()


def handle_result(client, userdata, msg):
    result = json.loads(msg.payload)
    if result["result"] == 0:
        transaction = next(
            t for t in transactions if t["transactionId"] == result["transactionId"]
        )
        transaction["winner"] = result["clientId"]
        transaction["solution"] = result["solution"]
        if result["clientId"] == client_id:
            print("Received result from sd/result, I won!")
        else:
            print("Received result from sd/result, I lost!")


def on_message(client, userdata, msg):
    if msg.topic == "sd/challenge":
        challenge_thread = threading.Thread(
            target=handle_challenge, args=(client, userdata, msg)
        )
        challenge_thread.start()
    elif msg.topic == "sd/result":
        result_thread = threading.Thread(
            target=handle_result, args=(client, userdata, msg)
        )
        result_thread.start()


def main() -> None:
    server_address = args.server
    broker_address, broker_port = server_address.split(":")
    broker_port = int(broker_port)
    client = mqtt.Client()
    client.connect(broker_address, broker_port, 60)
    client.on_connect = on_connect
    client.on_message = on_message
    client.loop_forever()


main()
