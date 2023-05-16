# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import fedlearn_grcp as fedlearn__grpc__pb2


class apiStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ClientRegister = channel.unary_unary(
                '/main.api/ClientRegister',
                request_serializer=fedlearn__grpc__pb2.ClientRegisterRequest.SerializeToString,
                response_deserializer=fedlearn__grpc__pb2.ClientRegisterResponse.FromString,
                )


class apiServicer(object):
    """Missing associated documentation comment in .proto file."""

    def ClientRegister(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_apiServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'ClientRegister': grpc.unary_unary_rpc_method_handler(
                    servicer.ClientRegister,
                    request_deserializer=fedlearn__grpc__pb2.ClientRegisterRequest.FromString,
                    response_serializer=fedlearn__grpc__pb2.ClientRegisterResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'main.api', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class api(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def ClientRegister(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/main.api/ClientRegister',
            fedlearn__grpc__pb2.ClientRegisterRequest.SerializeToString,
            fedlearn__grpc__pb2.ClientRegisterResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)


class clientStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.TrainingStart = channel.unary_unary(
                '/main.client/TrainingStart',
                request_serializer=fedlearn__grpc__pb2.TrainingStartRequest.SerializeToString,
                response_deserializer=fedlearn__grpc__pb2.TrainingStartResponse.FromString,
                )
        self.ModelEvaluation = channel.unary_unary(
                '/main.client/ModelEvaluation',
                request_serializer=fedlearn__grpc__pb2.ModelEvaluationRequest.SerializeToString,
                response_deserializer=fedlearn__grpc__pb2.ModelEvaluationResponse.FromString,
                )


class clientServicer(object):
    """Missing associated documentation comment in .proto file."""

    def TrainingStart(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ModelEvaluation(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_clientServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'TrainingStart': grpc.unary_unary_rpc_method_handler(
                    servicer.TrainingStart,
                    request_deserializer=fedlearn__grpc__pb2.TrainingStartRequest.FromString,
                    response_serializer=fedlearn__grpc__pb2.TrainingStartResponse.SerializeToString,
            ),
            'ModelEvaluation': grpc.unary_unary_rpc_method_handler(
                    servicer.ModelEvaluation,
                    request_deserializer=fedlearn__grpc__pb2.ModelEvaluationRequest.FromString,
                    response_serializer=fedlearn__grpc__pb2.ModelEvaluationResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'main.client', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class client(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def TrainingStart(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/main.client/TrainingStart',
            fedlearn__grpc__pb2.TrainingStartRequest.SerializeToString,
            fedlearn__grpc__pb2.TrainingStartResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ModelEvaluation(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/main.client/ModelEvaluation',
            fedlearn__grpc__pb2.ModelEvaluationRequest.SerializeToString,
            fedlearn__grpc__pb2.ModelEvaluationResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
