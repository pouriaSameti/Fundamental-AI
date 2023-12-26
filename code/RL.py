import keras
import numpy as np
import tensorflow as tf
from collections import deque


class DeepQLearning:
    def __init__(self, discount_factor: float, n_states: int, n_actions: int, batch_size: int, memory_length: int,
                 learning_rate: float):
        self.batch_size = batch_size
        self.memory_length = memory_length
        self.memory = list()
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.nS = n_states
        self.nA = n_actions
        self.model = DeepQLearning.__model_initialization(self, learning_rate=learning_rate)

    def play(self, env, state, epsilon):
        action = self.epsilon_greedy_policy(state, epsilon)
        next_state, reward, done, truncated = env.step(action)

        map_state = self.__mapping(state)
        map_next_state = self.__mapping(next_state)

        self.sampling(current_state=map_state, action=action, reward=reward, next_state=map_next_state, done=done)
        return next_state, reward, done, truncated

    def epsilon_greedy_policy(self, state, epsilon=0):
        if np.random.uniform(0, 1) < epsilon:
            return np.random.choice(range(self.nA))

        current_state = self.__mapping(state)
        q_values = self.model.predict([current_state])[0]
        return np.argmax(q_values)

    def train_network(self):
        np.random.shuffle(self.memory)
        batch_sample = self.memory[:self.batch_size]

        for experience in batch_sample:
            q_current_predicted = self.model.predict([experience["current_state"]])
            q_target = experience["reward"]

            if not experience["done"]:
                q_target = q_target + self.learning_rate * np.max(self.model.predict([experience["next_state"]])[0])

            q_current_predicted[0][experience["action"]] = q_target
            print(np.array([experience["current_state"]], shape=(1, self.nS)))

            self.model.fit(np.narray([experience["current_state"]], shape=(1, self.nS)),
                           np.narray([q_current_predicted]), verbose=0, epochs=1)

    def sampling(self, current_state, action, reward, next_state, done):
        self.memory.append({"current_state": current_state, "action": action, "reward": reward,
                            "next_state": next_state, "done": done})

        if len(self.memory) > self.memory_length:
            self.memory.pop(0)

    def __model_initialization(self, learning_rate: float):
        model = keras.models.Sequential([
            keras.layers.Dense(units=30, input_dim=1, activation='elu'),
            keras.layers.Dense(units=self.nA, activation='linear')
        ])

        model.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=learning_rate))
        return model

    @classmethod
    def __mapping(cls, observation: tuple):
        return int(observation[0] * 10 + observation[1])

    @classmethod
    def epsilon_exp_decay(cls, epsilon, decay_rate):
        epsilon = epsilon * np.exp(-decay_rate)
        return epsilon


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

    @staticmethod
    def decay_exponential(epsilon0: float, iteration: int, s: int):
        return epsilon0 * 0.1 ** (iteration/s)
