import gym
import gym_maze
import time
import itertools
import numpy as np


def random_epsilon_greedy_policy(Q: np.ndarray, state: int, epsilon: float):
    portoghal = np.random.randint(0, 100) / 100
    actions = list(range(env.action_space.n))
    if portoghal < epsilon:
        return np.random.choice(actions)
    else:
        return np.argmax(Q[state])


def mapping(observation: tuple):
    return int(observation[0] * 10 + observation[1])


if __name__ == '__main__':

    # Create an environment
    env = gym.make("maze-random-10x10-plus-v0")
    observation = env.reset()

    nS = 100
    nA = 4
    NUM_EPISODES = 1000
    epsilon = 0.4
    Q = np.zeros([nS, nA])
    state = observation

    for episode in range(NUM_EPISODES):
        for t in itertools.count():
            action = random_epsilon_greedy_policy(Q, mapping(state), epsilon)
            # Note: .sample() is used to sample random action from the environment's action space

            # Perform the action and receive feedback from the environment
            next_state, reward, done, truncated = env.step(action)
            # print(next_state, reward, done, truncated)

            epsilon -= 0.01
            if done or truncated:
                observation = env.reset()
                state = observation

        env.render()
        time.sleep(0.1)

    # Close the environment
    env.close()
