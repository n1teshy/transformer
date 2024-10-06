from typing import Callable


class CircularBatchGenerator:
    def __init__(self, get_generator: Callable):
        self.get_generator = get_generator

    def __iter__(self):
        while True:
            yield from self.get_generator()
