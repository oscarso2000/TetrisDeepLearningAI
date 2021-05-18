# tetris game and AI
#from tetris import TetrisApp
from tetris_trainer import TetrisApp
from tetris_ai import TetrisAI
from multiprocessing import Process
from dqn_agent import DQNAgent
from datetime import datetime
import time, threading
from logs import CustomTensorBoard
from tqdm import tqdm
from collections import defaultdict, OrderedDict
import time


def dqn():
     ### Code for DQL
    app = TetrisApp()
    ai = TetrisAI(app)
    threading.Thread(target=app.run).start()
    # num_units = 100
    # ai.num_units = num_units
    # ai.gen_weights = OrderedDict()
    # ai.cur_gen = 1
    # ai.cur_unit = -1
    # ai.mutation_val = mutation_val
    episodes = 200
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
    agent = DQNAgent(4,
                     n_neurons=n_neurons, activations=activations,
                     epsilon_stop_episode=epsilon_stop_episode, mem_size=mem_size,
                     discount=discount, replay_start_size=replay_start_size)

    scores = []

    for episode in tqdm(range(episodes)):
        print(1)
        cur_state = ai.tetris_app.get_state()
        ai.set_board(cur_state["board"])
        ai.set_tetromino(cur_state["tetromino"], cur_state["tetromino_x"], cur_state["tetromino_y"])
        print(2)
        current_state = ai._get_board_props()
        # done = False
        done = cur_state["gameover"]
        steps = 0
        render = False 
        print(3)
        while not done and (not max_steps or steps < max_steps):
            next_states = ai.get_next_states()
            # best_state = agent.best_state(next_states.values())
            best_action, best_state, reward = agent.best_state(next_states)
            # best_action = None
            # print(4)
            # for action, state in next_states.items():
            
            #     if state == best_state:
            #         best_action = action
            #         break
            print(5)
            # reward = ai.get_reward(cur_state["board"])
            actions = ai.get_actions(best_action)
            ai.tetris_app.add_actions(actions)
            print(6)
            # Update

            cur_state = ai.tetris_app.get_state()
            done = cur_state["gameover"]
            # reward = 1 + (ai._get_board_props()[0] ** 2) * len(ai.board[0])
    
            agent.add_to_memory(current_state, next_states[best_action], reward, done)
            current_state = ai._get_board_props()
            steps += 1
            print("HERE",ai.tetris_app.score, len(agent.memory))

        scores.append(ai.tetris_app.score)

        ai.tetris_app.add_actions(["space"])
        print(7)

        # Train
        if episode % train_every == 0:
            agent.train(batch_size=batch_size, epochs=epochs)
            # log._plot()
        
        '''
        TODO:

        fix render lags 
        train
        '''

   




if __name__ == '__main__':
    dqn()

   