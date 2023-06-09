# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import grpcCalc_pb2 as grpcCalc__pb2


class CalcStub(object):
    """Calculator service definition
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.add = channel.unary_unary(
                '/main.Calc/add',
                request_serializer=grpcCalc__pb2.Operands.SerializeToString,
                response_deserializer=grpcCalc__pb2.IntResult.FromString,
                )
        self.sub = channel.unary_unary(
                '/main.Calc/sub',
                request_serializer=grpcCalc__pb2.Operands.SerializeToString,
                response_deserializer=grpcCalc__pb2.IntResult.FromString,
                )


class CalcServicer(object):
    """Calculator service definition
    """

    def add(self, request, context):
        """Add two integers
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def sub(self, request, context):
        """Subtracts right from left
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_CalcServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'add': grpc.unary_unary_rpc_method_handler(
                    servicer.add,
                    request_deserializer=grpcCalc__pb2.Operands.FromString,
                    response_serializer=grpcCalc__pb2.IntResult.SerializeToString,
            ),
            'sub': grpc.unary_unary_rpc_method_handler(
                    servicer.sub,
                    request_deserializer=grpcCalc__pb2.Operands.FromString,
                    response_serializer=grpcCalc__pb2.IntResult.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'main.Calc', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class Calc(object):
    """Calculator service definition
    """

    @staticmethod
    def add(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/main.Calc/add',
            grpcCalc__pb2.Operands.SerializeToString,
            grpcCalc__pb2.IntResult.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def sub(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/main.Calc/sub',
            grpcCalc__pb2.Operands.SerializeToString,
            grpcCalc__pb2.IntResult.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
