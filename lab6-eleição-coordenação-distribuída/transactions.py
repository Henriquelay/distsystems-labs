"""Anything related to communication and overall interaction with other nodes"""

import uuid

from typing import List, Dict, Any

uuid1 = uuid.uuid1()
my_client_id = (uuid1.int >> 96) & 0xFFFF
print(f"My client id is {my_client_id}")

# Type this array for me. It is used correctly on the code below
transactions: List[Dict[str, Any]] = []

def add_transaction(id: int, challenge: int, solution: str, winner: int) -> None:
    transactions.append(
        {
            "transactionId": id,
            "challenge": challenge,
            "solution": solution,
            "winner": winner,
        }
    )

def transaction_has_winner(transaction_id: int) -> bool:
    transaction = next(t for t in transactions if t["transactionId"] == transaction_id)
    return transaction["winner"] != -1
