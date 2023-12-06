# Import nessary libraries
import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.cliffwalking import CliffWalkingEnv
from gymnasium.error import DependencyNotInstalled
from os import path

# Do not change this class
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
image_path = path.join(path.dirname(gym.__file__), "envs", "toy_text")


class CliffWalking(CliffWalkingEnv):
    def __init__(self, is_hardmode=True, num_cliffs=10, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_hardmode = is_hardmode

        # Generate random cliff positions
        if self.is_hardmode:
            self.num_cliffs = num_cliffs
            self._cliff = np.zeros(self.shape, dtype=bool)
            self.start_state = (3, 0)
            self.terminal_state = (self.shape[0] - 1, self.shape[1] - 1)
            self.cliff_positions = []
            while len(self.cliff_positions) < self.num_cliffs:
                new_row = np.random.randint(0, 4)
                new_col = np.random.randint(0, 11)
                state = (new_row, new_col)
                if (
                        (state not in self.cliff_positions)
                        and (state != self.start_state)
                        and (state != self.terminal_state)
                ):
                    self._cliff[new_row, new_col] = True
                    if not self.is_valid():
                        self._cliff[new_row, new_col] = False
                        continue
                    self.cliff_positions.append(state)

        # Calculate transition probabilities and rewards
        self.P = {}
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            self.P[s] = {a: [] for a in range(self.nA)}
            self.P[s][UP] = self._calculate_transition_prob(position, [-1, 0])
            self.P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1])
            self.P[s][DOWN] = self._calculate_transition_prob(position, [1, 0])
            self.P[s][LEFT] = self._calculate_transition_prob(position, [0, -1])

    def _calculate_transition_prob(self, current, delta):
        new_position = np.array(current) + np.array(delta)
        new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        # if self._cliff[tuple(new_position)]:
        #     return [(1.0, self.start_state_index, -100, False)]

        # terminal_state = (self.shape[0] - 1, self.shape[1] - 1)
        # is_terminated = tuple(new_position) == terminal_state
        # if is_terminated:
        #     return [(1 / 3, new_state, 100, is_terminated)]
        # return [(1 / 3, new_state, -1, is_terminated)]

        # if self._cliff[tuple(new_position)]:
        #     return [(1.0, self.start_state_index, (-7.5) + euclidean_distance(new_position[0], new_position[1]), False)]
        #
        # terminal_state = (self.shape[0] - 1, self.shape[1] - 1)
        # is_terminated = tuple(new_position) == terminal_state
        # if is_terminated:
        #     return [(1 / 3, new_state, 0, is_terminated)]
        # return [(1 / 3, new_state, euclidean_distance(new_position[0], new_position[1]), is_terminated)]

        if self._cliff[tuple(new_position)]:
            return [(1.0, self.start_state_index, (-7.5) + manhattan_distance(new_position[0], new_position[1]), False)]

        terminal_state = (self.shape[0] - 1, self.shape[1] - 1)
        is_terminated = tuple(new_position) == terminal_state
        if is_terminated:
            return [(1 / 3, new_state, 0, is_terminated)]
        return [(1 / 3, new_state, manhattan_distance(new_position[0], new_position[1]), is_terminated)]

    # DFS to check that it's a valid path.
    def is_valid(self):
        frontier, discovered = [], set()
        frontier.append((3, 0))
        while frontier:
            r, c = frontier.pop()
            if not (r, c) in discovered:
                discovered.add((r, c))
                directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
                for x, y in directions:
                    r_new = r + x
                    c_new = c + y
                    if r_new < 0 or r_new >= self.shape[0] or c_new < 0 or c_new >= self.shape[1]:
                        continue
                    if (r_new, c_new) == self.terminal_state:
                        return True
                    if not self._cliff[r_new][c_new]:
                        frontier.append((r_new, c_new))
        return False

    def step(self, action):
        if action not in [0, 1, 2, 3]:
            raise ValueError(f"Invalid action {action}   must be in [0, 1, 2, 3]")

        if self.is_hardmode:
            match action:
                case 0:
                    action = np.random.choice([0, 1, 3], p=[1 / 3, 1 / 3, 1 / 3])
                case 1:
                    action = np.random.choice([0, 1, 2], p=[1 / 3, 1 / 3, 1 / 3])
                case 2:
                    action = np.random.choice([1, 2, 3], p=[1 / 3, 1 / 3, 1 / 3])
                case 3:
                    action = np.random.choice([0, 2, 3], p=[1 / 3, 1 / 3, 1 / 3])

        return super().step(action)

    def _render_gui(self, mode):
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gymnasium[toy-text]`"
            ) from e
        if self.window_surface is None:
            pygame.init()

            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("CliffWalking - Edited by Audrina & Kian")
                self.window_surface = pygame.display.set_mode(self.window_size)
            else:  # rgb_array
                self.window_surface = pygame.Surface(self.window_size)
        if self.clock is None:
            self.clock = pygame.time.Clock()
        if self.elf_images is None:
            hikers = [
                path.join(image_path, "img/elf_up.png"),
                path.join(image_path, "img/elf_right.png"),
                path.join(image_path, "img/elf_down.png"),
                path.join(image_path, "img/elf_left.png"),
            ]
            self.elf_images = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in hikers
            ]
        if self.start_img is None:
            file_name = path.join(image_path, "img/stool.png")
            self.start_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.goal_img is None:
            file_name = path.join(image_path, "img/cookie.png")
            self.goal_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )
        if self.mountain_bg_img is None:
            bg_imgs = [
                path.join(image_path, "img/mountain_bg1.png"),
                path.join(image_path, "img/mountain_bg2.png"),
            ]
            self.mountain_bg_img = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in bg_imgs
            ]
        if self.near_cliff_img is None:
            near_cliff_imgs = [
                path.join(image_path, "img/mountain_near-cliff1.png"),
                path.join(image_path, "img/mountain_near-cliff2.png"),
            ]
            self.near_cliff_img = [
                pygame.transform.scale(pygame.image.load(f_name), self.cell_size)
                for f_name in near_cliff_imgs
            ]
        if self.cliff_img is None:
            file_name = path.join(image_path, "img/mountain_cliff.png")
            self.cliff_img = pygame.transform.scale(
                pygame.image.load(file_name), self.cell_size
            )

        for s in range(self.nS):
            row, col = np.unravel_index(s, self.shape)
            pos = (col * self.cell_size[0], row * self.cell_size[1])
            check_board_mask = row % 2 ^ col % 2
            self.window_surface.blit(self.mountain_bg_img[check_board_mask], pos)

            if self._cliff[row, col]:
                self.window_surface.blit(self.cliff_img, pos)
            if s == self.start_state_index:
                self.window_surface.blit(self.start_img, pos)
            if s == self.nS - 1:
                self.window_surface.blit(self.goal_img, pos)
            if s == self.s:
                elf_pos = (pos[0], pos[1] - 0.1 * self.cell_size[1])
                last_action = self.lastaction if self.lastaction is not None else 2
                self.window_surface.blit(self.elf_images[last_action], elf_pos)

        if mode == "human":
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window_surface)), axes=(1, 0, 2)
            )


def policy_initialization(policy: dict, states: list, actions: list):
    for s in states:
        policy[s] = np.random.choice(actions)


def v_star_initialization(n_states: int):
    return np.zeros(n_states)


def q_star_initialization(n_states: int, n_actions: int):
    return [[0 for a in range(n_actions)] for _ in range(n_states)]


def value_iteration(q, v_star, q_star, discount_factor: float, max_iteration: int):
    for _ in range(max_iteration):
        for state in q:
            for action in q[state]:
                for probability, nextState, reward, isTerminalState in q[state][action]:
                    if action == 0:
                        probability1, nextState1, reward1, isTerminalState1 = q[state][action + 1][0]
                        probability3, nextState3, reward3, isTerminalState3 = q[state][action + 3][0]
                        q_star[state][action] = (probability * (reward + discount_factor * v_star[nextState])) + \
                                                (probability1 * (reward1 + discount_factor * v_star[nextState1])) + \
                                                (probability3 * (reward3 + discount_factor * v_star[nextState3]))
                    elif action == 1:
                        probability2, nextState2, reward2, isTerminalState2 = q[state][action + 1][0]
                        probability0, nextState0, reward0, isTerminalState0 = q[state][action - 1][0]
                        q_star[state][action] = (probability * (reward + discount_factor * v_star[nextState])) + \
                                                (probability2 * (reward2 + discount_factor * v_star[nextState2])) + \
                                                (probability0 * (reward0 + discount_factor * v_star[nextState0]))
                    elif action == 2:
                        probability1, nextState1, reward1, isTerminalState1 = q[state][action - 1][0]
                        probability3, nextState3, reward3, isTerminalState3 = q[state][action + 1][0]
                        q_star[state][action] = (probability * (reward + discount_factor * v_star[nextState])) + \
                                                (probability1 * (reward1 + discount_factor * v_star[nextState1])) + \
                                                (probability3 * (reward3 + discount_factor * v_star[nextState3]))
                    else:
                        probability2, nextState2, reward2, isTerminalState2 = q[state][action - 1][0]
                        probability0, nextState0, reward0, isTerminalState0 = q[state][action - 3][0]
                        q_star[state][action] = (probability * (reward + discount_factor * v_star[nextState])) + \
                                                (probability2 * (reward2 + discount_factor * v_star[nextState2])) + \
                                                (probability0 * (reward0 + discount_factor * v_star[nextState0]))

            v_star[state] = max(q_star[state])

    return v_star, q_star


def euclidean_distance(x, y):
    termination_pos = (3, 11)
    return (-1) * np.sqrt(np.power(x - termination_pos[0], 2) + np.power(y - termination_pos[1], 2))


def manhattan_distance(x, y):
    termination_pos = (3, 11)
    return (-1) * np.abs(np.abs(x - termination_pos[0]) + np.abs(y - termination_pos[1]))


def policy_extraction(q_star):
    policy = {}  # key:state  value:direction

    for index in range(len(q_star)):
        state = q_star[index]
        max_index = np.argmax(state)
        policy[index] = max_index
    return policy


if __name__ == '__main__':

    # Create an environment
    env = CliffWalking(render_mode="human")
    observation, info = env.reset(seed=30)

    # Define the maximum number of iterations
    max_iter_number = 1000
    q = env.P
    v_star = v_star_initialization(env.nS)
    q_star = q_star_initialization(n_states=env.nS, n_actions=env.nA)

    actions = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}
    states = list(range(env.nS))
    policy = {}  # key:state  value:action

    discount_factor = 0.9

    policy_initialization(policy, states, [*actions.keys()])

    v_star, q_star = value_iteration(q=q, v_star=v_star, q_star=q_star, discount_factor=discount_factor, max_iteration=max_iter_number)
    policy = policy_extraction(q_star)

    # print(v_star)
    # print(q_star)
    # print(policy)

    n_victory = 0
    for __ in range(max_iter_number):
        # TODO: Implement the agent policy here
        # Note: .sample() is used to sample random action from the environment's action space
        # Choose an action (Replace this random action with your agent's policy)

        # action = env.action_space.sample()
        action = policy[env.s]

        # Perform the action and receive feedback from the environment
        next_state, reward, done, truncated, info = env.step(action)

        if done or truncated:
            n_victory += 1
            observation, info = env.reset()

    # Close the environment
    env.close()
    print('number of victory', n_victory)
