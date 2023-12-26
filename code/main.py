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


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def show_convergence_plot(converge_list: list, information: str):
    converge_list = moving_average(converge_list, n=3)
    x = list(range(len(converge_list)))
    y = converge_list
    sns.lineplot(x=x, y=y)
    plt.suptitle(information)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.show()


def show_mse_plot(error_list: list, information: str):
    x = list(range(len(error_list)))
    y = error_list
    sns.lineplot(x=x, y=y)
    plt.suptitle(information)
    plt.xlabel('Episode')
    plt.ylabel('MSE')
    plt.show()


if __name__ == '__main__':

    # Create an environment
    env = gym.make("maze-random-10x10-plus-v0")
    observation = env.reset()

    nS = 100
    nA = 4
    NUM_EPISODES = 4000
    NUM_ITERATION = 800
    epsilon = 0.4
    alpha = 0.5
    gamma = 0.995

    v = list(range(nS))
    Q = np.zeros([nS, nA])

    state = observation
    convergence = []

    # win_num = 0
    # for episode in range(NUM_EPISODES):
    #     observation = env.reset()
    #
    #     v = QLearning.calculate_v(Q=Q)
    #     convergence.append(np.abs(np.sum(v)))
    #
    #     for t in range(NUM_ITERATION):
    #         action = QLearning.approximation_utility_policy(Q, mapping(state))
    #         # action = QLearning.epsilon_greedy_policy(Q, state=mapping(state), n_actions=nA, epsilon=epsilon)
    #         # epsilon = QLearning.decay_exponential(epsilon0=epsilon, iteration=episode, s=40)
    #
    #         next_state, reward, done, truncated = env.step(action)
    #         if done or truncated:
    #             win_num += 1
    #             state = env.reset()
    #             print(win_num, 'Episode:', episode)
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
    # explanation = f'Episodes: {NUM_EPISODES}, Iteration per Episode: {NUM_ITERATION}, Number of Winning; {win_num}'
    # show_convergence_plot(convergence, information=explanation)

    TRAINING_EPISODES = 500
    BATCH_SIZE = 40
    MEMORY_LENGTH = 100
    LEARNING_RATE = 0.01
    DECAY_RATE = 0.0000001
    EPOCH = 20
    agent = DeepQLearning(discount_factor=gamma, n_states=nS, n_actions=nA, batch_size=BATCH_SIZE,
                          memory_length=MEMORY_LENGTH, learning_rate=LEARNING_RATE, each_epoch=EPOCH)

    # Learning Loop
    total_steps = 0
    mse_errors = []
    for e in range(TRAINING_EPISODES):
        current_state = env.reset()

        for step in itertools.count():
            env.render()

            total_steps += 1

            action = agent.random_policy()
            next_state, reward, done, truncated = env.step(action)
            map_state = DeepQLearning.mapping(state)
            map_next_state = DeepQLearning.mapping(next_state)

            agent.sampling(current_state=map_state, action=action, reward=reward, next_state=map_next_state, done=done)

            if done:
                mean_mse = agent.train_network()
                mse_errors.append(mean_mse)
                total_steps = 0
                current_state = env.reset()
                break

    print('End of the learning')

    current_state = env.reset()
    for _ in range(NUM_EPISODES):
        action = agent.best_utility_policy(current_state)
        next_state, reward, done, truncated = env.step(action)
        current_state = next_state
        env.render()

    env.cloce()
