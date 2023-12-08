import numpy as np


class MDP:

    def __init__(self, n_states, n_actions):
        self.n_states = n_states
        self.n_actions = n_actions

        self.q_star = self.q_star_initialization()
        self.v_star = self.v_star_initialization()

    def q_star_initialization(self):
        return [[0 for a in range(self.n_actions)] for _ in range(self.n_states)]

    def v_star_initialization(self):
        return np.zeros(self.n_states)
