
# tetris game and AI
#from tetris import TetrisApp
from tetris_trainer import TetrisApp
from tetris_ai import TetrisAI
from multiprocessing import Process
from dqn_agent import DQNAgent
from test_dql import TestAgent

import time, threading

def tetris_p():
    app = TetrisApp()
    ai = TetrisAI(app)

    threading.Thread(target=app.run).start()


if __name__ == '__main__':

  episodes = 2000
  max_steps = None
  epsilon_stop_episode = 1500
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

  '''

  uncomment to run multiple trainings at one time

  processes = []
  num_tetris_ai = 5

  for m in range(num_tetris_ai):
    p = Process(target=tetris_p)
    p.start()
    processes.append(p)

  for p in processes:
    p.join()'''
  
  # DQL ALGORITHM
  app = TetrisApp()
  agent = DQNAgent(4,
                  n_neurons=n_neurons, activations=activations,
                  epsilon_stop_episode=epsilon_stop_episode, mem_size=mem_size,
                  discount=discount, replay_start_size=replay_start_size)
  ai = TetrisAI(app, agent)

  threading.Thread(target=app.run).start()
  ai.start_dql()



  # TEST DQL ALGORITHM
  # app = TetrisApp()
  # agent = TestAgent()

  # ai = TetrisAI(app, agent)

  # threading.Thread(target=app.run).start()
  # ai.test_dql()

  # RUN GENETIC 
  # ai.start(50, seed=(-0.178, -0.525, -0.198, -0.284, -0.685, 0.873))
  # ai.start(100)


