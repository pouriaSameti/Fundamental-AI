import keras
import tensorflow as tf
from collections import deque


class DeepQLearning:
    def __init__(self, batch_size: int, memory_length: int):
        self.batch_size = batch_size
        self.memory_len = memory_length

        self.memory = deque(maxlen=self.memory_len)
        self.model = None

