import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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

    def value_iteration(self, state_action_matrix, discount_factor: float, max_iteration: int):
        score_list = []
        for iteration in range(max_iteration):
            previous_v_star = np.copy(self.v_star)
            for state in state_action_matrix:
                for action in state_action_matrix[state]:
                    for probability, nextState, reward, isTerminalState in state_action_matrix[state][action]:
                        if action == 0:
                            probability1, nextState1, reward1, isTerminalState1 = state_action_matrix[state][action + 1][0]
                            probability3, nextState3, reward3, isTerminalState3 = state_action_matrix[state][action + 3][0]
                            self.q_star[state][action] = (probability * (reward + discount_factor * self.v_star[nextState])) + \
                                                    (probability1 * (reward1 + discount_factor * self.v_star[nextState1])) + \
                                                    (probability3 * (reward3 + discount_factor * self.v_star[nextState3]))
                        elif action == 1:
                            probability2, nextState2, reward2, isTerminalState2 = state_action_matrix[state][action + 1][0]
                            probability0, nextState0, reward0, isTerminalState0 = state_action_matrix[state][action - 1][0]
                            self.q_star[state][action] = (probability * (reward + discount_factor * self.v_star[nextState])) + \
                                                    (probability2 * (reward2 + discount_factor * self.v_star[nextState2])) + \
                                                    (probability0 * (reward0 + discount_factor * self.v_star[nextState0]))
                        elif action == 2:
                            probability1, nextState1, reward1, isTerminalState1 = state_action_matrix[state][action - 1][0]
                            probability3, nextState3, reward3, isTerminalState3 = state_action_matrix[state][action + 1][0]
                            self.q_star[state][action] = (probability * (reward + discount_factor * self.v_star[nextState])) + \
                                                    (probability1 * (reward1 + discount_factor * self.v_star[nextState1])) + \
                                                    (probability3 * (reward3 + discount_factor * self.v_star[nextState3]))
                        else:
                            probability2, nextState2, reward2, isTerminalState2 = state_action_matrix[state][action - 1][0]
                            probability0, nextState0, reward0, isTerminalState0 = state_action_matrix[state][action - 3][0]
                            self.q_star[state][action] = (probability * (reward + discount_factor * self.v_star[nextState])) + \
                                                    (probability2 * (reward2 + discount_factor * self.v_star[nextState2])) + \
                                                    (probability0 * (reward0 + discount_factor * self.v_star[nextState0]))

                self.v_star[state] = max(self.q_star[state])
                if np.sum(np.abs(self.v_star)) - np.sum(np.abs(previous_v_star)) == 0:
                    return self.v_star, self.q_star, score_list, iteration

            score_list.append(np.abs(np.sum(self.v_star)))

        return self.v_star, self.q_star, score_list, max_iteration

    @classmethod
    def policy_extraction(cls, q_star: list):
        policy = {}  # key:state  value:direction

        for index in range(len(q_star)):
            state = q_star[index]
            max_index = np.argmax(state)
            policy[index] = max_index
        return policy

    @classmethod
    def show_score_per_iteration(cls, n_iterations, scores):
        iteration_list = list(range(n_iterations))

        x_axis = np.array(iteration_list)
        y_axis = np.array(scores)
        sns.lineplot(x=x_axis, y=y_axis)
        plt.suptitle(f'Plot of sum of the v_star values per iteration\n Iterations: {n_iterations}')
        plt.show()

    @classmethod
    def show_heatmap(cls, v_star, n_rows: int, n_columns: int):
        game_state_score = []
        for i in range(n_rows):
            row_score = []
            for j in range(n_columns):
                position = (i * n_columns) + j
                row_score.append(v_star[position])
            game_state_score.append(row_score)

        sns.heatmap(game_state_score, annot=True)
        plt.suptitle('Heatmap Based on v_star')
        plt.show()

