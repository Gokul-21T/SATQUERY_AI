from typing import List


class ExecutionTrace:
    def __init__(self, query: str):
        self.query = query
        self.steps: List[str] = [f"Query received: '{query}'"]

    def add_step(self, step: str):
        self.steps.append(step)

    def to_list(self) -> List[str]:
        return self.steps
