from __future__ import print_function

import logging

import grpc
import grpcCalc_pb2
import grpcCalc_pb2_grpc


def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        print("Client started")
        stub = grpcCalc_pb2_grpc.CalcStub(channel)
        left = int(input('left operand: '))
        right = int(input('right operand: '))
        operands = grpcCalc_pb2.Operands(left=left, right=right)
        response = stub.add(operands)
        print("Add: " + str(response.result))
        response = stub.sub(operands)
        print("Sub: " + str(response.result))


if __name__ == '__main__':
    logging.basicConfig()
    run()
