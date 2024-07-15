from gomoku_env import GomokuEnv
from dqn_agent import DQNAgent
from model import create_model
import numpy as np

def train_agent(episodes=1000):
    env = GomokuEnv()
    board_size = env.board_size
    model = create_model(board_size)
    agent = DQNAgent(model, board_size)
    save_path = 'gomoku_model.h5'

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, board_size, board_size, 1])
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, board_size, board_size, 1])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                print(f"episode: {e}/{episodes}, score: {time}, e: {agent.epsilon:.2}")
                break
        agent.replay()

    model.save(save_path)

if __name__ == "__main__":
    train_agent()
