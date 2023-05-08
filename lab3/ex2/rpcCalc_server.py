from xmlrpc.server import SimpleXMLRPCServer

def add(left: int, right: int) -> int:
    print(f'ADD Received params {left} and {right}')
    return left + right

def sub(left: int, right: int) -> int:
    print(f'SUB Received params {left} and {right}')
    return left - right


if __name__ == '__main__':
    try:
        server = SimpleXMLRPCServer(('localhost', 80))
    except Exception as e:
        print('ERROR: %s' % e)
        exit(1)

    server.register_function(add)
    server.register_function(sub)
    print("Listening")
    server.serve_forever()
