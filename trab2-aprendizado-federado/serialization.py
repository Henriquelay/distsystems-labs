import pickle
import base64

def serialize_weights(weights):
    serialized_weights = pickle.dumps(weights)
    return serialized_weights


def deserialize_weights(serialized_weights):
    weights = pickle.loads(serialized_weights)
    return weights

def encode_base64(data):
    encoded_data = base64.b64encode(data).decode("utf-8")
    return encoded_data


def decode_base64(encoded_data):
    decoded_data = base64.b64decode(encoded_data)
    return decoded_data