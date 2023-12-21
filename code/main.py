import gym
import gym_maze
import time
import itertools
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def random_epsilon_greedy_policy(Q: np.ndarray, state: int, epsilon: float):
    portoghal = np.random.randint(0, 100) / 100
    actions = list(range(env.action_space.n))
    if portoghal < epsilon:
        return np.random.choice(actions)
    else:
        return np.argmax(Q[state])


def mapping(observation: tuple):
    return int(observation[0] * 10 + observation[1])


def reward_euclidean():
    rewards = list()
    for x in range(10):
        for y in range(10):
            rewards.append((np.sqrt(np.power((x - 9), 2) + np.power((y - 9), 2))) * -0.1)

    rewards[99] = 100
    return rewards


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
    NUM_EPISODES = 100
    NUM_ITERATION = 1000
    epsilon = 0.4
    alpha = 0.9
    gamma = 0.95
    Q = np.zeros([nS, nA])
    state = observation
    # golabi = np.zeros([nS, nA])
    rewards = reward_euclidean()
    v_star = list(range(nS))
    convergence = []

    for episode in range(NUM_EPISODES):
        observation = env.reset()
        for t in itertools.count():
            action = random_epsilon_greedy_policy(Q, mapping(state), epsilon)
            # print(action)
            # Perform the action and receive feedback from the environment
            next_state, reward, done, truncated = env.step(action)

            if done or truncated or t == NUM_ITERATION:
                t = 0
                for s in range(nS):
                    v_star[s] = np.max(Q[s])
                convergence.append(np.sum(v_star))
                print(np.sum(v_star))
                print(v_star)
                break

            state = next_state
            Q[mapping(state)][action] += alpha * (rewards[mapping(state)] + gamma * np.max(Q[mapping(next_state)]) - Q[mapping(state)][action])
            epsilon -= 0.01

            # if mapping(next_state) == mapping(state):
            #     golabi[mapping(state)][action] = 1
            #
            # if golabi[mapping(state)][0] + golabi[mapping(state)][1] + golabi[mapping(state)][2] + golabi[mapping(state)][3] == 3:
            #     rewards[mapping(state)] -= 0.01

            env.render()

    # Close the environment
    env.close()
    show_convergence_plot(convergence)
