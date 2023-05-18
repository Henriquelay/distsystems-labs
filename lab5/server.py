import hashlib
import random
import threading
import time

import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--server", default="localhost:1883")
args = parser.parse_args()

import paho.mqtt.client as mqtt


class Transaction:
    transactions = 0

    @classmethod
    def get_last_transaction(cls):
        return cls.transactions - 1

    def __init__(self):
        self.transaction_id = Transaction.transactions
        Transaction.transactions += 1
        self.challenge = random.randint(1, 6)
        self.solution = self.create_solution()
        self.winner = -1

    def create_solution(self):
        num1 = random.randint(1, 1000 * self.challenge)
        num2 = random.randint(1, 1000 * self.challenge)

        hash = hashlib.sha1()
        hash.update(str(num1 + num2).encode("utf-8"))
        return hash.hexdigest()


class LodgeCoinServicer:
    def __init__(self, client, broker_address, broker_port):
        self.transactions = []
        self.lock = threading.Lock()

        self.client = client
        self.client.on_connect = self.on_connect
        self.client.on_message = self.on_message

        self.client.connect(broker_address, broker_port, 60)
        self.mqtt_thread = threading.Thread(target=self.client.loop_forever)
        self.mqtt_thread.start()

    def on_connect(self, client, userdata, flags, rc):
        print("Connected with result code " + str(rc))
        client.subscribe("sd/solution")
        print("Subscribed to topic 'sd/solution'")
        self.new_challenge()

    def on_message(self, client, userdata, msg):
        if msg.topic == "sd/solution":
            self.handle_solution(client, userdata, msg)

    def handle_solution(self, client, userdata, msg):
        payload = msg.payload.decode("utf-8")
        solution = json.loads(payload)
        # print(f"Received solution for transaction {solution['transactionId']}: {solution['solution']}")

        result = self.submitChallenge(solution)

        if result == -1:
            return

        solution_to_send = solution["solution"] if result == 0 else ""

        result_data = {
            "clientId": solution["clientId"],
            "transactionId": solution["transactionId"],
            "solution": solution_to_send,
            "result": result,
        }
        result_message = json.dumps(result_data)
        self.client.publish("sd/result", result_message)
        if result == 0:
            print(
                f"Published result winner for transaction {solution['transactionId']}: {result_message}"
            )

    def new_challenge(self):
        with self.lock:
            transaction = Transaction()
            self.transactions.append(transaction)
            self.client.publish(
                "sd/challenge",
                json.dumps(
                    {
                        "transactionId": transaction.transaction_id,
                        "challenge": transaction.challenge,
                    }
                ),
            )
            print(
                f"Published new challenge with id {transaction.transaction_id} and challenge {transaction.challenge}"
            )

    def submitChallenge(self, request):
        if request["transactionId"] >= len(self.transactions):
            return -1

        transaction = self.transactions[request["transactionId"]]

        with self.lock:
            if transaction.winner != -1:
                return -1
            if transaction.solution == request["solution"]:
                transaction.winner = request["clientId"]
                print(
                    f"Transaction {transaction.transaction_id} has a winner: {transaction.winner}"
                )
                return 0
            else:
                return 1


def serve():
    client = mqtt.Client()
    broker_address = args.server.split(":")[0]
    broker_port = int(args.server.split(":")[1])
    lodgecoin = LodgeCoinServicer(client, broker_address, broker_port)

    while True:
        print("Menu:")
        print("1. New challenge")
        print("2. Exit controller")
        choice = input("Enter your choice: ")

        if choice == "1":
            lodgecoin.new_challenge()
        elif choice == "2":
            exit(0)
        else:
            print("Invalid choice")


serve()
