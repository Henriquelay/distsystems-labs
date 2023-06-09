import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--server", default="localhost:1883")
args = parser.parse_args()

import paho.mqtt.client as mqtt
from handler import on_connect, on_message


def client(server_address: str) -> None:
    broker_address, broker_port = server_address.split(":")
    broker_port = int(broker_port)
    client = mqtt.Client()
    client.connect(broker_address, broker_port, 60)
    client.on_connect = on_connect
    client.on_message = on_message
    client.loop_forever()
