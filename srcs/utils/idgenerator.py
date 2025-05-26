import uuid


class IDGenerator:
    def __init__(self, digit: int = 8) -> None:
        self.digit = digit
        self.uuid = str(uuid.uuid4())

    def generate(self) -> str:
        return self.uuid[: self.digit]
