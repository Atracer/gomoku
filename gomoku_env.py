import gym
from gym import spaces
import numpy as np

class GomokuEnv(gym.Env):
    def __init__(self, board_size=15):
        super(GomokuEnv, self).__init__()
        self.board_size = board_size
        self.action_space = spaces.Discrete(board_size * board_size)
        self.observation_space = spaces.Box(low=0, high=2, shape=(board_size, board_size), dtype=np.int)

        self.reset()

    def reset(self):
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int)
        self.done = False
        self.current_player = 1
        return self.board

    def step(self, action):
        if self.done:
            return self.board, 0, self.done, {}

        row, col = divmod(action, self.board_size)
        if self.board[row, col] != 0:
            return self.board, -1, self.done, {}

        self.board[row, col] = self.current_player
        reward = self.check_winner(row, col)

        if reward == 0 and np.all(self.board != 0):
            self.done = True
            reward = 0.5  # Draw

        self.current_player = 3 - self.current_player
        return self.board, reward, self.done, {}

    def check_winner(self, row, col):
        directions = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for i in range(1, 5):
                r, c = row + dr * i, col + dc * i
                if 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == self.current_player:
                    count += 1
                else:
                    break

            for i in range(1, 5):
                r, c = row - dr * i, col - dc * i
                if 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == self.current_player:
                    count += 1
                else:
                    break

            if count >= 5:
                self.done = True
                return 1 if self.current_player == 1 else -1

        return 0

    def render(self, mode='human'):
        for row in self.board:
            print(' '.join(['.' if x == 0 else 'X' if x == 1 else 'O' for x in row]))
        print()
