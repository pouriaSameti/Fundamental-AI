import gym
import gym_maze
import time
import itertools
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from RL import DeepQLearning
from RL import QLearning


def mapping(observation: tuple):
    return int(observation[0] * 10 + observation[1])


def show_convergence_plot(converge_list: list):
    x = list(range(len(converge_list)))
    y = converge_list
    sns.lineplot(x=x, y=y)
    plt.show()


if __name__ == '__main__':

    # Create an environment
    env = gym.make("maze-random-10x10-plus-v0")
    observation = env.reset()

    nS = 100
    nA = 4
    NUM_EPISODES = 1000
    NUM_ITERATION = 800
    epsilon = 0.4
    alpha = 0.1
    gamma = 0.95

    v = list(range(nS))
    Q = np.zeros([nS, nA])

    state = observation
    convergence = []

    win_num = 0
    # for episode in range(NUM_EPISODES):
    #     observation = env.reset()
    #
    #     for t in range(NUM_ITERATION):
    #         action = QLearning.approximation_utility_policy(Q, mapping(state))
    #         # action = QLearning.epsilon_greedy_policy(Q, state=mapping(state), n_actions=nA, epsilon=epsilon)
    #         epsilon -= 0.01
    #
    #         next_state, reward, done, truncated = env.step(action)
    #         if done or truncated:
    #             win_num += 1
    #             v = QLearning.calculate_v(Q=Q)
    #             convergence.append(np.abs(np.sum(v)))
    #             state = env.reset()
    #             print(win_num)
    #             break
    #
    #         Q[mapping(state)][action] += QLearning.qlearning_equation(Q=Q, current_state=mapping(state), action=action,
    #                                                                   next_state=mapping(next_state), reward=reward,
    #                                                                   alpha=alpha, gamma=gamma)
    #         state = next_state
    #         env.render()
    #
    # # Close the environment
    # env.close()
    # show_convergence_plot(convergence)

    agent = DeepQLearning(discount_factor=gamma, n_states=nS, n_actions=nA, batch_size=10000, memory_length=2000,
                          learning_rate=0.001)
    total_steps = 0
    for e in range(NUM_EPISODES):
        current_state = env.reset()

        for step in itertools.count():
            total_steps += 1
            next_state, reward, done, truncated = agent.play(env=env, state=current_state, epsilon=epsilon)
            env.render()
            if done:
                agent.train_network()
                total_steps = 0
                current_state = env.reset()
                break

    print('End of the learning')
    current_state = env.reset()
    for _ in range(5000):
        next_state, reward, done, truncated = agent.play(env=env, state=current_state, epsilon=epsilon)
        env.render()
    env.cloce()
