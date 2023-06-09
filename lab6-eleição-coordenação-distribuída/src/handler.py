import json
import threading

from paho.mqtt.client import Client as mqttClient, MQTTMessage

from typing import Final

from transactions import add_transaction, transactions, my_client_id
from mine import mine

challenges_topic: Final[str] = "sd/challenge"
results_topic: Final[str] = "sd/result"


def on_connect(client: mqttClient, userdata, flags, rc: int):
    print("Connected with result code " + str(rc))
    client.subscribe(challenges_topic)
    client.subscribe(results_topic)
    print("Subscribed to topic 'sd/challenge'")
    print("Subscribed to topic 'sd/result'")


def on_message(client: mqttClient, userdata, msg: MQTTMessage):
    if msg.topic == challenges_topic:
        challenge_thread = threading.Thread(
            target=handle_challenge, args=(client, userdata, msg)
        )
        challenge_thread.start()
    elif msg.topic == results_topic:
        result_thread = threading.Thread(
            target=handle_result, args=(client, userdata, msg)
        )
        result_thread.start()


def handle_challenge(client: mqttClient, userdata, msg: MQTTMessage):
    print(f"challenge received: {msg.payload}")
    transaction = json.loads(msg.payload)
    add_transaction(transaction["transactionId"], transaction["challenge"], "", -1)
    mining_thread = threading.Thread(
        target=mine,
        args=(client, transaction["transactionId"], transaction["challenge"]),
    )
    mining_thread.start()


def handle_result(client: mqttClient, userdata, msg: MQTTMessage):
    result = json.loads(msg.payload)
    if result["result"] == 0:
        transaction = next(
            t for t in transactions if t["transactionId"] == result["transactionId"]
        )
        transaction["winner"] = result["clientId"]
        transaction["solution"] = result["solution"]
        if result["clientId"] == my_client_id:
            print("Received result from sd/result, I won!")
        else:
            print("Received result from sd/result, I lost!")
