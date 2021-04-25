# tetris game and AI
from tetris_trainer import TetrisApp
from tetris_ai import TetrisAI
from multiprocessing import Process
from dqn_agent import DQNAgent
from test_dql import TestAgent
import time, threading


def train_dql(discount, epsilon_stop_episode, learning_rate, step_size, model, logs_file):
  episodes = 2000
  max_steps = None
  mem_size = 20000
  discount = 0.95
  batch_size = 512
  epochs = 1
  render_every = 50
  log_every = 50
  replay_start_size = 2000
  train_every = 1
  n_neurons = [32, 32]
  render_delay = None
  activations = ['relu', 'relu', 'linear']
  app = TetrisApp()
  agent = DQNAgent(4, n_neurons=n_neurons, activations=activations,
                  epsilon_stop_episode=epsilon_stop_episode, mem_size=mem_size,
                  discount=discount, replay_start_size=replay_start_size, learning_rate)
  ai = TetrisAI(app, agent, step_size, model, logs_file)

  threading.Thread(target=app.run).start()
  ai.start_dql()


def test_dql(model):
  '''Test a dql model
    model: STR of a model saved as a .h5'''
  app = TetrisApp()
  agent = TestAgent(model)
  ai = TetrisAI(app, agent)
  threading.Thread(target=app.run).start()
  ai.test_dql()


if __name__ == '__main__':

  train_dql( 'test_init_model.h5' , 'scores/test_init.csv')
  # test_dql('dql_model_regular.h5')

