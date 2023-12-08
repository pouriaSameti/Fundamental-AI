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

    def value_iteration(self, state_action: list, discount_factor: float, max_iteration: int):
        for _ in range(max_iteration):
            for state in state_action:
                for action in state_action[state]:
                    for probability, nextState, reward, isTerminalState in state_action[state][action]:
                        if action == 0:
                            probability1, nextState1, reward1, isTerminalState1 = state_action[state][action + 1][0]
                            probability3, nextState3, reward3, isTerminalState3 = state_action[state][action + 3][0]
                            self.q_star[state][action] = (probability * (reward + discount_factor * self.v_star[nextState])) + \
                                                    (probability1 * (reward1 + discount_factor * self.v_star[nextState1])) + \
                                                    (probability3 * (reward3 + discount_factor * self.v_star[nextState3]))
                        elif action == 1:
                            probability2, nextState2, reward2, isTerminalState2 = state_action[state][action + 1][0]
                            probability0, nextState0, reward0, isTerminalState0 = state_action[state][action - 1][0]
                            self.q_star[state][action] = (probability * (reward + discount_factor * self.v_star[nextState])) + \
                                                    (probability2 * (reward2 + discount_factor * self.v_star[nextState2])) + \
                                                    (probability0 * (reward0 + discount_factor * self.v_star[nextState0]))
                        elif action == 2:
                            probability1, nextState1, reward1, isTerminalState1 = state_action[state][action - 1][0]
                            probability3, nextState3, reward3, isTerminalState3 = state_action[state][action + 1][0]
                            self.q_star[state][action] = (probability * (reward + discount_factor * self.v_star[nextState])) + \
                                                    (probability1 * (reward1 + discount_factor * self.v_star[nextState1])) + \
                                                    (probability3 * (reward3 + discount_factor * self.v_star[nextState3]))
                        else:
                            probability2, nextState2, reward2, isTerminalState2 = state_action[state][action - 1][0]
                            probability0, nextState0, reward0, isTerminalState0 = state_action[state][action - 3][0]
                            self.q_star[state][action] = (probability * (reward + discount_factor * self.v_star[nextState])) + \
                                                    (probability2 * (reward2 + discount_factor * self.v_star[nextState2])) + \
                                                    (probability0 * (reward0 + discount_factor * self.v_star[nextState0]))

                self.v_star[state] = max(self.q_star[state])

        return self.v_star, self.q_star

    @classmethod
    def policy_extraction(cls, q_star: list):
        policy = {}  # key:state  value:direction

        for index in range(len(q_star)):
            state = q_star[index]
            max_index = np.argmax(state)
            policy[index] = max_index
        return policy
