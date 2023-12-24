import keras
import numpy as np
import tensorflow as tf
from collections import deque


class DeepQLearning:
    def __init__(self, discount_factor: float, n_states: int, n_actions: int, batch_size: int, memory_length: int,
                 neuron_per_layer: int, learning_rate: float):
        self.batch_size = batch_size
        self.memory = deque(maxlen=memory_length)
        self.discount_factor = discount_factor
        self.model = DeepQLearning.__model_initialization(neuron_per_layer=neuron_per_layer, n_states=n_states,
                                                          n_actions=n_actions)

        self.nS = n_states
        self.nA = n_actions

    def play(self, env, state, epsilon):
        action = self.epsilon_greedy_policy(state, epsilon)
        next_state, reward, done, truncated = env.step(action)

        self.memory.append((state, action, reward, next_state, done))
        return next_state, reward, done, truncated

    def epsilon_greedy_policy(self, state, epsilon=0):
        print(state)
        # print(np.newaxis)
        if np.random.rand() < epsilon:
            return np.random.randint(self.nA)
        q_values = self.model.predict(state[np.newaxis])
        return np.argmax(q_values[0])

    def train_network(self):
        observations = self.sampling_from_memory()
        loss_function = keras.losses.mean_squared_error
        optimizer = keras.optimizers.Adam(lr=1e-3)

        states, actions, rewards, next_states, dones = observations

        next_q_values = self.model.predict(next_states)
        max_next_q = np.max(next_q_values)
        target_q_values = (rewards + (1 - dones) * self.discount_factor * max_next_q)

        mask = tf.one_hot(actions, self.nA)
        with tf.GradientTape() as tape:
            all_q_values = self.model(states)
            q_values = tf.reduce_sum(all_q_values * mask, axis=1, keepdims=True)
            error = tf.reduce_mean(loss_function(target_q_values, q_values))

        gradients = tape.gradient(error, self.model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

    def sampling_from_memory(self):
        random_indices = np.random.randint(len(self.memory), size=self.batch_size)
        batch = [self.memory[index] for index in random_indices]
        states, actions, rewards, next_states, dones = [np.array([observation[field_index] for observation in batch])
                                                        for field_index in range(4)]
        return states, actions, rewards, next_states, dones

    @classmethod
    def __model_initialization(cls, n_states: int, n_actions: int, learning_rate: float):
        model = keras.models.Sequential([
            keras.layers.Dense(units=30, input_dim=n_states, activation='elu'),
            keras.layers.Dense(units=30, activation='elu'),
            keras.layers.Dense(units=n_actions, activation='linear')
        ])

        model.compile(loss="mse", optimizer = keras.optimizers.Adam(lr=learning_rate))
        return model


class QLearning:

    @staticmethod
    def qlearning_equation(Q: np.ndarray, current_state: int, action: int, next_state: int, reward: float,
                           alpha: float, gamma: float):
        return alpha * (reward + gamma * np.max(Q[next_state]) - Q[current_state][action])

    @staticmethod
    def calculate_v(Q: np.ndarray):
        return [np.max(Q[state]) for state in range(len(Q))]

    @staticmethod
    def epsilon_greedy_policy(Q: np.ndarray, state: int, n_actions: int, epsilon: float):
        if np.random.rand() < epsilon:
            return np.random.randint(n_actions)
        return np.argmax(Q[state])

    @staticmethod
    def approximation_utility_policy(Q: np.ndarray, state: int):
        return np.argmax(Q[state])
