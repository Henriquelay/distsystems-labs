class Transaction:
    """A transaction of overall challenges and results"""

    def __init__(
        self,
        transaction_id: int,
        challenge: int,
        solution: str,
        winner: int,
    ):
        self.transaction_id = transaction_id
        self.challenge = challenge
        self.solution = solution
        self.winner = winner

    def to_json(self) -> str:
        """Converts a transaction to a json string"""
        from json import dumps

        return dumps(
            {
                "TransactionID": self.transaction_id,
                "Challenge": self.challenge,
                "Solution": self.solution,
                "Winner": self.winner,
            }
        )

    def has_winner(self) -> bool:
        return self.winner != -1


class ResultTransaction:
    """A transaction of challenge results. Result = 0 means the result is invalid. != 0 means the result is valid"""

    def __init__(
        self,
        client_id: int,
        transaction_id: int,
        solution: str,
        result: int,
    ):
        self.client_id = client_id
        self.transaction_id = transaction_id
        self.solution = solution
        self.result = result

    def to_json(self) -> str:
        """Converts a transaction to a json string"""
        from json import dumps

        return dumps(
            {
                "ClientID": self.client_id,
                "TransactionID": self.transaction_id,
                "Solution": self.solution,
                "Result": self.result,
            }
        )

    def to_transaction(self, challenge: int) -> Transaction:
        """Converts a resulttransaction to a transaction"""
        winner = -1
        if self.result != 0:
            winner = self.client_id
        return Transaction(self.transaction_id, challenge, self.solution, winner)

    @classmethod
    def from_json(cls, json: bytes | bytearray):
        """Creates a transaction from a json string"""
        from json import loads

        loaded = loads(json)
        return cls(
            loaded["ClientID"],
            loaded["TransactionID"],
            loaded["Solution"],
            loaded["Result"],
        )


class ChallengeTransaction:
    """A transaction of challenge problems"""

    def __init__(
        self,
        transaction_id: int,
        challenge: int,
    ):
        self.transaction_id = transaction_id
        self.challenge = challenge

    def to_json(self) -> str:
        """Converts a transaction to a json string"""
        from json import dumps

        return dumps(
            {
                "TransactionID": self.transaction_id,
                "Challenge": self.challenge,
            }
        )

    def to_transaction(self) -> Transaction:
        """Converts a challengetransaction to a transaction"""
        return Transaction(self.transaction_id, self.challenge, solution="", winner=-1)

    @classmethod
    def from_json(cls, json: bytes | bytearray):
        """Creates a transaction from a json string"""
        from json import loads

        loaded = loads(json)
        return cls(loaded["TransactionID"], loaded["Challenge"])


class SolutionTransaction:
    """A transaction of solutions"""

    def __init__(
        self,
        client_id: int,
        transaction_id: int,
        solution: str,
    ):
        self.client_id = client_id
        self.transaction_id = transaction_id
        self.solution = solution

    def to_json(self) -> str:
        """Converts a transaction to a json string"""
        from json import dumps

        return dumps(
            {
                "ClientID": self.client_id,
                "TransactionID": self.transaction_id,
                "Solution": self.solution,
            }
        )

    @classmethod
    def from_json(cls, json: bytes | bytearray):
        """Creates a transaction from a json string"""
        from json import loads

        loaded = loads(json)
        return cls(loaded["ClientID"], loaded["TransactionID"], loaded["Solution"])


class InitTransaction:
    """A transaction of initialization"""

    def __init__(
        self,
        client_id: int,
    ):
        self.client_id = client_id

    def to_json(self) -> str:
        """Converts a transaction to a json string"""
        from json import dumps

        return dumps(
            {
                "ClientID": self.client_id,
            }
        )

    @classmethod
    def from_json(cls, json: bytes | bytearray):
        """Creates a transaction from a json string"""
        from json import loads

        loaded = loads(json)
        return cls(loaded["ClientID"])


class VotingTransaction:
    """A transaction of voting"""

    def __init__(
        self,
        client_id: int,
        vote_id: int,
    ):
        self.client_id = client_id
        self.vote_id = vote_id

    def to_json(self) -> str:
        """Converts a transaction to a json string"""
        from json import dumps

        return dumps(
            {
                "ClientID": self.client_id,
                "VoteID": self.vote_id,
            }
        )

    @classmethod
    def from_json(cls, json: bytes | bytearray):
        """Creates a transaction from a json string"""
        from json import loads

        loaded = loads(json)
        return cls(loaded["ClientID"], loaded["VoteID"])
