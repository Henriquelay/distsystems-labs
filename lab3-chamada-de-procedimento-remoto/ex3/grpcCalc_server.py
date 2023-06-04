from concurrent import futures
import logging

import grpc
import grpcCalc_pb2
import grpcCalc_pb2_grpc


class Calc(grpcCalc_pb2_grpc.CalcServicer):

    def add(self, request, context):
        print(f'add: {request.left} + {request.right}')
        return grpcCalc_pb2.IntResult(result=request.left + request.right)

    def sub(self, request, context):
        print(f'sub: {request.left} - {request.right}')
        return grpcCalc_pb2.IntResult(result=request.left - request.right)


def serve():
    port = '50051'
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    grpcCalc_pb2_grpc.add_CalcServicer_to_server(Calc(), server)
    server.add_insecure_port('[::]:' + port)
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()


if __name__ == '__main__':
    logging.basicConfig()
    serve()
