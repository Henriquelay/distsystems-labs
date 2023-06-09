from json import dumps, loads

from paho.mqtt.client import Client as mqttClient, MQTTMessage

from typing import Final

from transactions import *

topics: Final[dict[str, str]] = {
    "challenges": "sd/challenge",
    "solutions": "sd/solution",
    "results": "sd/result",
    "init": "sd/init",
    "voting": "sd/voting",
}


class Node:
    """A node in the network. It can be a client or a server, depending on the election result. All code regarding controlflow should be here"""

    nodes_to_wait = 3

    from enum import Enum

    class State(Enum):
        """The current1 of the node"""

        INIT = 0
        WAITING_FOR_ELECTION = 20
        ELECTION = 40
        CHALLENGE_LEADER = 60
        CHALLENGE_FOLLOWER = 80
        SOLVING = 100

    state = State.INIT

    def __init__(self, server_address: str, nodes_to_wait: int) -> None:
        """Starts a node with a given server address"""
        self.nodes_to_wait = nodes_to_wait

        import paho.mqtt.client as mqtt

        def connect(self, server_address: str) -> mqttClient:
            broker_address, broker_port = server_address.split(":")
            broker_port = int(broker_port)

            client = mqtt.Client()
            client.connect(broker_address, broker_port, 60)
            # These lambdas only pack self on front of the args, and remove the first argument which is the Client, because it is in Self already. This avoid problems
            client.on_connect = lambda _, __, ___, rc: self.on_connect(rc)
            client.on_message = lambda _, __, msg: self.on_message(msg)
            client.on_connect_fail = lambda _, __: (
                print("Falha na conexão com o broker"),
                exit(),
            )
            return client

        # generate local id
        from uuid import uuid1 as generate_uuid

        self.client_id = (generate_uuid().int >> 96) & 0xFFFF
        print(f"Meu client ID é {self.client_id}")

        # Starts the local registry, every one accessed by their transaction or client id, as fit
        self.transactions: dict[int, Transaction] = {}
        self.init: dict[int, InitTransaction] = {}
        self.voting: dict[int, VotingTransaction] = {}

        # connect to broker
        print(f"Conectando ao broker em {server_address}")
        self.client = connect(self, server_address)

    def loop_forever(self):
        """Starts a blocking infinite loop for the node"""
        self.client.loop_forever()

    def on_connect(self, rc: int):
        """Actions the node take when connecting to the broker"""
        print("Conectado com result code " + str(rc))
        subscribes = [
            (topics["challenges"], 0),
            (topics["results"], 0),
            (topics["init"], 0),
        ]

        self.client.subscribe(subscribes)
        print(f"Inscrito ao tópico {topics['challenges']}")
        print(f"Inscrito ao tópico {topics['results']}")
        print(f"Inscrito ao tópico {topics['init']}")

        self.client.publish(topics["init"], InitTransaction(self.client_id).to_json())
        self.state = self.State.WAITING_FOR_ELECTION
        print("Mensagem Init publicada")

    def on_message(self, msg: MQTTMessage):
        """Actions the node takes when receiving a message from the broker. Non-blocking"""
        from threading import Thread

        if msg.topic == topics["challenges"]:
            # spawn an unblocking thread
            Thread(target=self.handle_challenge, args=[msg]).start()
        elif msg.topic == topics["results"]:
            Thread(target=self.handle_result, args=[msg]).start()
        elif msg.topic == topics["init"]:
            Thread(target=self.handle_init, args=[msg]).start()

        else:
            print(f"Recebi mensagem de um tópico desconhecido {msg.topic}")

    def handle_result(self, msg: MQTTMessage):
        received = ResultTransaction.from_json(msg.payload)
        self.transactions["results"].append(received)
        if received.result != 0:  # Valid solution
            if received.client_id == self.client_id:
                print("Ganhei!")
            else:
                print("Perdi :(")

    def handle_challenge(self, msg: MQTTMessage):
        received = ChallengeTransaction.from_json(msg.payload)
        print(f"Desafio recebido {received.transaction_id}")
        self.transactions[received.transaction_id] = received
        # Okay to block since it's in a subthread already
        self.mine(received)

    def handle_init(self, msg: MQTTMessage):
        if self.state != self.State.WAITING_FOR_ELECTION:
            return
        received = InitTransaction.from_json(msg.payload)
        if received.client_id not in self.init.keys():
            self.init[received.client_id] = received
            if received.client_id != self.client_id:
                # Received init from someone I have never seen, it still does not have my init so I'll publish my init again
                print(
                    f"Reenviando meu Init e esperando mais {self.nodes_to_wait - len(self.init)} nós"
                )
                self.client.publish(
                    topics["init"], InitTransaction(self.client_id).to_json()
                )

        # Nodes should be synced here
        if len(self.init) == self.nodes_to_wait:
            self.state = self.State.ELECTION
            print("Starting election")
            # TODO: start election
            exit()

    def mine(self, challenge: ChallengeTransaction):
        """Solving the challenge of a transaction"""
        from random import randint
        from hashlib import sha1
        from time import sleep

        def try_generate_solution(challenge: int) -> str:
            num1 = randint(1, 1000 * challenge)
            num2 = randint(1, 1000 * challenge)

            hash = sha1()
            hash.update(str(num1 + num2).encode("utf-8"))
            return hash.hexdigest()

        print(
            f"Desafio da transação {challenge.transaction_id}: challenge {challenge.challenge}"
        )
        print("Minerando...")
        while not self.transactions[challenge.transaction_id].has_winner():
            sleep(0.01)
            solution = try_generate_solution(challenge.challenge)
            solution = SolutionTransaction(
                self.client_id, challenge.transaction_id, solution
            )
            # publish solution found
            self.client.publish(topics["solutions"], solution.to_json())


Node("localhost:1883", 3).loop_forever()
