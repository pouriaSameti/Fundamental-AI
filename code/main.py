import gym
import gym_maze
import time

# Create an environment

env = gym.make("maze-random-10x10-plus-v0")
observation = env.reset()

# Define the maximum number of iterations
NUM_EPISODES = 1000

for episode in range(NUM_EPISODES):

    # Note: .sample() is used to sample random action from the environment's action space

    # Choose an action (Replace this random action with your agent's policy)
    action = env.action_space.sample()

    # Perform the action and receive feedback from the environment
    next_state, reward, done, truncated = env.step(action)
    print(next_state, reward, done, truncated)

    if done or truncated:
        observation = env.reset()

    env.render()
    time.sleep(0.1)


# Close the environment
env.close()

