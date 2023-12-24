import keras
import numpy as np
import tensorflow as tf
from collections import deque


class DeepQLearning:
    def __init__(self, discount_factor: float, n_states: int, n_actions: int, batch_size: int, memory_length: int,
                 neuron_per_layer: int):

        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_length)
        self.discount_factor = discount_factor
        self.model = DeepQLearning.__model_initialization(neuron_per_layer=neuron_per_layer, n_states=n_states,
                                                          n_actions=n_actions)

    def sampling_from_memory(self):
        random_indices = np.random.randint(len(self.memory), size=self.batch_size)
        batch = [self.memory[index] for index in random_indices]

        states, actions, rewards, next_states, dones = [np.array([observation[field_index] for observation in batch])
                                          for field_index in range(4)]
        return states, actions, rewards, next_states, dones

    @classmethod
    def __model_initialization(cls, neuron_per_layer: int, n_states: int, n_actions: int):
        model = keras.Sequential([
            keras.layers.Dense(neuron_per_layer, activation='elu', input_shape=n_states),
            keras.layers.Dense(neuron_per_layer, activation='elu'),
            keras.layers.Dense(neuron_per_layer, activation='elu'),
            keras.layers.Dense(n_actions)
        ])
        return model
