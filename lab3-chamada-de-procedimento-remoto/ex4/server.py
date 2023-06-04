from concurrent import futures
import grpc
import hashlib
import logging
import random
import threading

import mine_grpc_pb2
import mine_grpc_pb2_grpc

current_transaction = 0

class Transaction():
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
        num1 = random.randint(1, 100000*self.challenge)
        num2 = random.randint(1, 100000*self.challenge)

        hash = hashlib.sha1()
        hash.update(str(num1 + num2).encode('utf-8'))
        return hash.hexdigest()

class LodgeCoinServicer(mine_grpc_pb2_grpc.apiServicer):
    def __init__(self):
        self.transactions = []
        self.lock = threading.Lock()

    def getTransactionId(self, request, context):
        if len(self.transactions) > 0:
            transaction = self.transactions[-1]
            if transaction.winner == -1:
                return mine_grpc_pb2.intResult(result=transaction.transaction_id)

        transaction = Transaction()
        self.transactions.append(transaction)
        print('Created new transaction with id: %d', transaction.transaction_id)
        return mine_grpc_pb2.intResult(result=transaction.transaction_id)

    def getChallenge(self, request, context):
        if request.transactionId >= len(self.transactions):
            return mine_grpc_pb2.intResult(result=-1)

        transaction = self.transactions[request.transactionId]
        return mine_grpc_pb2.intResult(result=transaction.challenge)

    def getTransactionStatus(self, request, context):
        print('Received getTransactionStatus request.')
        if request.transactionId >= len(self.transactions):
            return mine_grpc_pb2.intResult(result=-1)

        transaction = self.transactions[request.transactionId]

        has_winner = 0

        if transaction.winner != -1:
            has_winner = 1

        return mine_grpc_pb2.intResult(result=has_winner)

    def submitChallenge(self, request, context):
        if request.transactionId >= len(self.transactions):
            return mine_grpc_pb2.intResult(result=-1)

        transaction = self.transactions[request.transactionId]

        with self.lock:
            if transaction.winner != -1:
                return mine_grpc_pb2.intResult(result=2)
            if transaction.solution == request.solution:
                transaction.winner = request.clientId
                print(f"Transaction {transaction.transaction_id} has a winner: {transaction.winner}")
                return mine_grpc_pb2.intResult(result=1)

        return mine_grpc_pb2.intResult(result=0)

    def getWinner(self, request, context):
        if request.transactionId >= len(self.transactions):
            return mine_grpc_pb2.intResult(result=-1)

        transaction = self.transactions[request.transactionId]

        if transaction.winner == -1:
            return mine_grpc_pb2.intResult(result=0)

        return mine_grpc_pb2.intResult(result=transaction.winner)

    def getSolution(self, request, context):
        if request.transactionId >= len(self.transactions):
            return mine_grpc_pb2.structResult(status=-1)

        transaction = self.transactions[request.transactionId]
        return mine_grpc_pb2.structResult(status=0, solution=transaction.solution, challenge=transaction.challenge)


def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=4))
    mine_grpc_pb2_grpc.add_apiServicer_to_server(LodgeCoinServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    server.wait_for_termination()


serve()
