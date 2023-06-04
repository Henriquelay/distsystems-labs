import grpc
import mine_grpc_pb2
import mine_grpc_pb2_grpc
import random
import hashlib
import uuid

uuid1 = uuid.uuid1()
client_id = (uuid1.int >> 96) & 0xFFFF

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--server', default='localhost:50051')
args = parser.parse_args()


def get_int_input(prompt: str) -> int:
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Entrada inválida, tente novamente.")

def get_transaction_id(stub) -> None:
    response = stub.getTransactionId(mine_grpc_pb2.void())
    print(f"ID da transação atual: {response.result}")

def get_challenge(stub) -> None:
    transaction_id = get_int_input("Digite o ID da transação: ")
    response = stub.getChallenge(mine_grpc_pb2.transactionId(transactionId=transaction_id))
    if response.result != -1:
        print(f"Desafio da transação {transaction_id}: {response.result}")
    else:
        print(f"Transação com ID {transaction_id} não encontrada.")

def get_transaction_status(stub) -> None:
    transaction_id = get_int_input("Digite o ID da transação: ")
    response = stub.getTransactionStatus(mine_grpc_pb2.transactionId(transactionId=transaction_id))
    if response.result == -1:
        print(f"Transação com ID {transaction_id} não encontrada.")
    elif response.result == 0:
        print(f"A transação {transaction_id} não possui vencedor.")
    elif response.result == 1:
        print(f"A transação {transaction_id} possui um vencedor.")

def get_winner(stub) -> None:
    transaction_id = get_int_input("Digite o ID da transação: ")
    response = stub.getWinner(mine_grpc_pb2.transactionId(transactionId=transaction_id))
    if response.result == -1:
        print(f"Transação com ID {transaction_id} não encontrada.")
    elif response.result == 0:
        print(f"A transação {transaction_id} não possui vencedor.")
    else:
        print(f"ID do cliente vencedor da transação {transaction_id}: {response.result}")

def get_solution(stub) -> None:
    transaction_id = get_int_input("Digite o ID da transação: ")
    response = stub.getSolution(mine_grpc_pb2.transactionId(transactionId=transaction_id))
    if not response.solution:
        print(f"A transação {transaction_id} não possui solução.")
    else:
        print(f"Solução da transação {transaction_id}: {response.solution}")

def try_generate_solution(challenge) -> str:
    num1 = random.randint(1, 100000*challenge)
    num2 = random.randint(1, 100000*challenge)

    hash = hashlib.sha1()
    hash.update(str(num1 + num2).encode('utf-8'))
    return hash.hexdigest()

def mine(stub) -> None:
    transaction_id = stub.getTransactionId(mine_grpc_pb2.void()).result
    challenge = stub.getChallenge(mine_grpc_pb2.transactionId(transactionId=transaction_id)).result
    print(f"Desafio da transação {transaction_id}: challenge {challenge}")
    print("Minerando...")
    while True:
        solution = try_generate_solution(challenge)
        response = stub.submitChallenge(mine_grpc_pb2.challengeArgs(
            transactionId=transaction_id,
            clientId=client_id,
            solution=solution))
        if response.result == -1:
            print(f"Transação com ID {transaction_id} não encontrada.")
        elif response.result == 1:
            print("Solução correta!")
            break
        elif response.result == 2:
            print("Esta transação já tem um vencedor.")
            break
        

def main() -> None:
    server_address = args.server
    with grpc.insecure_channel(server_address) as channel:
        stub = mine_grpc_pb2_grpc.apiStub(channel)
        print(f"Conectado ao servidor {server_address}.")

        while True:
            print("Menu:")
            print("1. Get current transaction ID")
            print("2. Get challenge")
            print("3. Get transaction status")
            print("4. Get transaction winner")
            print("5. Get solution")
            print("6. Mine")
            print("7. Exit")

            option = get_int_input("Digite a opção desejada: ")

            if option == 1:
                get_transaction_id(stub)
            elif option == 2:
                get_challenge(stub)
            elif option == 3:
                get_transaction_status(stub)
            elif option == 4:
                get_winner(stub)
            elif option == 5:
                get_solution(stub)
            elif option == 6:
                mine(stub)
            elif option == 7:
                break
            else:
                print("Opção inválida, tente novamente.")

main()