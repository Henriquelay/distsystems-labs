import sys
import comparison

if __name__ == '__main__':
    # Get list size and number of execution rounds from stdin
    list_size = int(sys.stdin.readline())
    execution_rounds = int(sys.stdin.readline())

    comparison.compare(list_size)
