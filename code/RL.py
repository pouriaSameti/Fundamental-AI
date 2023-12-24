import keras
import tensorflow as tf
from collections import deque


class DeepQLearning:
    def __init__(self, n_states:int, n_actions: int, batch_size: int, memory_length: int,
                 neuron_per_layer: int):

        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_length)
        self.model = DeepQLearning.__model_initialization(neuron_per_layer=neuron_per_layer, n_states=n_states,
                                                          n_actions=n_actions)

    @classmethod
    def __model_initialization(cls, neuron_per_layer: int, n_states: int, n_actions: int):
        model = keras.Sequential([
            keras.layers.Dense(neuron_per_layer, activation='elu', input_shape=n_states),
            keras.layers.Dense(neuron_per_layer, activation='elu'),
            keras.layers.Dense(neuron_per_layer, activation='elu'),
            keras.layers.Dense(n_actions)
        ])
        return model
