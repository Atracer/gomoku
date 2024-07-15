from gomoku_env import GomokuEnv
from dqn_agent import DQNAgent
from model import create_model
import numpy as np

def test_agent():
    env = GomokuEnv()
    board_size = env.board_size
    model = create_model(board_size)
    agent = DQNAgent(model, board_size)
    model_path = 'gomoku_model.h5'

    # Load the trained model weights if saved
    model.load_weights(model_path)

    state = env.reset()
    state = np.reshape(state, [1, board_size, board_size, 1])
    done = False
    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        env.render()
        next_state = np.reshape(next_state, [1, board_size, board_size, 1])
        state = next_state

if __name__ == "__main__":
    test_agent()
