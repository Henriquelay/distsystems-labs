from xmlrpc.client import ServerProxy

try:
    sv = ServerProxy('http://localhost:80')
except Exception as e:
    print('ERROR: %s' % e)
    exit(1) 

left = int(input('left operand: '))
right = int(input('right operand: '))

print('add: %d' % sv.add(left, right))
print('sub: %d' % sv.sub(left, right))
