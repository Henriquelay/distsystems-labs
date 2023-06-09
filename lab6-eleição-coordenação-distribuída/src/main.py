from threading import Thread
from random import randint

from typing import Final

from paho.mqtt.client import Client as mqttClient, MQTTMessage

from transactions import *

topics: Final[dict[str, str]] = {
    "challenges": "sd/challenge",
    "solutions": "sd/solution",
    "results": "sd/result",
    "init": "sd/init",
    "voting": "sd/voting",
}


def generate_id() -> int:
    """Random integer within a range"""
    return randint(0, 100_000_000)


def try_generate_solution(challenge: int) -> str:
    from hashlib import sha1

    num1 = randint(1, 1000 * challenge)
    num2 = randint(1, 1000 * challenge)

    hash = sha1()
    hash.update(str(num1 + num2).encode("utf-8"))
    return hash.hexdigest()


class Node:
    """A node in the network. It can be a client or a server, depending on the election result. All code regarding controlflow should be here"""

    nodes_to_wait = 3

    # Starts the local registry, every one accessed by their transaction or client id, as fit
    challenges: dict[int, ChallengeTransaction] = {}
    results: dict[int, ResultTransaction] = {}
    transactions: dict[int, Transaction] = {}
    init: dict[int, InitTransaction] = {}
    voting: dict[int, VotingTransaction] = {}
    solutions: dict[int, str] = {}

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
        self.client_id = generate_id()
        print(f"Meu clientID é {self.client_id}")

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
            (topics["voting"], 0),
            (topics["solutions"], 0),
        ]

        self.client.subscribe(subscribes)
        print(f"Inscrito ao tópico {topics['challenges']}")
        print(f"Inscrito ao tópico {topics['results']}")
        print(f"Inscrito ao tópico {topics['init']}")
        print(f"Inscrito ao tópico {topics['voting']}")
        print(f"Inscrito ao tópico {topics['solutions']}")

        self.client.publish(topics["init"], InitTransaction(self.client_id).to_json())
        self.state = self.State.WAITING_FOR_ELECTION
        print("Mensagem Init publicada")

    def on_message(self, msg: MQTTMessage):
        """Actions the node takes when receiving a message from the broker. Non-blocking"""

        if msg.topic == topics["challenges"]:
            # spawn an unblocking thread
            Thread(target=self.handle_challenge, args=[msg]).start()
        elif msg.topic == topics["results"]:
            Thread(target=self.handle_result, args=[msg]).start()
        elif msg.topic == topics["init"]:
            Thread(target=self.handle_init, args=[msg]).start()
        elif msg.topic == topics["voting"]:
            Thread(target=self.handle_voting, args=[msg]).start()
        elif msg.topic == topics["solutions"]:
            Thread(target=self.handle_solution, args=[msg]).start()
        else:
            print(f"Recebi mensagem de um tópico desconhecido {msg.topic}")

    def handle_result(self, msg: MQTTMessage):
        received = ResultTransaction.from_json(msg.payload)
        self.results[received.transaction_id] = received
        if received.result != 0:  # Valid solution
            self.transactions[received.transaction_id] = received.to_transaction(
                self.challenges[received.transaction_id]
            )
            if received.client_id == self.client_id:
                print("Ganhei!")
            else:
                print("Perdi :(")
            self.state = self.State.WAITING_FOR_ELECTION
            self.handle_election()

    def handle_solution(self, msg: MQTTMessage):
        if self.state != self.State.CHALLENGE_LEADER:
            return
        received = SolutionTransaction.from_json(msg.payload)
        solution = received.solution
        if solution == self.solutions[received.transaction_id]:
            self.client.publish(
                topics["results"],
                ResultTransaction(
                    received.client_id, received.transaction_id, solution, 1
                ).to_json(),
            )
        else:
            self.client.publish(
                topics["results"],
                ResultTransaction(
                    received.client_id, received.transaction_id, solution, 0
                ).to_json(),
            )

    def handle_challenge(self, msg: MQTTMessage):
        received = ChallengeTransaction.from_json(msg.payload)
        print(f"Desafio recebido {received.transaction_id}")
        self.challenges[received.transaction_id] = received
        self.transactions[received.transaction_id] = received.to_transaction()
        self.mine(received)
        # mining_thread_handle = Thread(target=self.mine, args=[received])
        # mining_thread_handle.start()
        # self.mining_threads[received.transaction_id] = mining_thread_handle

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
            self.handle_election()

    def handle_election(self):
        # Clear last voting
        self.voting = {}
        self.state = self.State.ELECTION
        print("Iniciando eleição")
        vote = self.elect()
        print(f"Votando {vote.vote_id}")
        self.client.publish(topics["voting"], vote.to_json())

    def handle_voting(self, msg: MQTTMessage):
        """Handles a voting message, sets `self.state` accondingly"""

        received = VotingTransaction.from_json(msg.payload)
        self.voting[received.client_id] = received  # save the vote from each node
        if len(self.voting) == self.nodes_to_wait:

            def election_results(voting: dict[int, VotingTransaction]) -> int:
                """Returns the vote_id of the winner of the election"""
                from collections import Counter

                # get only vote_id from the voting transactions
                votes = Counter(vote.vote_id for vote in voting.values())
                # get the most common vote_id
                winner = votes.most_common(1)[0][0]
                return winner

            winner = election_results(self.voting)
            if winner == self.client_id:
                print("Sou o líder!")
                self.state = self.State.CHALLENGE_LEADER

                # grab a random solution and make it the result, keep it local
                challenge = randint(1, 6)
                solution = try_generate_solution(challenge)
                transaction_id = len(self.transactions)
                self.solutions[transaction_id] = solution
                # send a challenge
                challenge = ChallengeTransaction(transaction_id, challenge)
                self.client.publish(topics["challenges"], challenge.to_json())

            else:
                print(f"{winner} é o líder")
                self.state = self.State.CHALLENGE_FOLLOWER

    def elect(self) -> VotingTransaction:
        """Elects a leader for this round"""
        # Picks a random node to vote
        from random import choice

        vote_id = choice(list(self.init.keys()))
        return VotingTransaction(self.client_id, vote_id)

    def mine(self, challenge: ChallengeTransaction):
        """Solving the challenge of a transaction"""
        from time import sleep

        print(
            f"Desafio da transação {challenge.transaction_id}: challenge {challenge.challenge}"
        )
        print("Minerando...")
        while not self.transactions[challenge.transaction_id].has_winner():
            solution = try_generate_solution(challenge.challenge)
            self.client.publish(
                topics["solutions"],
                SolutionTransaction(
                    self.client_id, challenge.transaction_id, solution
                ).to_json(),
            )
            sleep(0.01) # Prevents the broker to whine and crash


Node("localhost:1883", 3).loop_forever()
