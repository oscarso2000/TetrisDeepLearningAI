from tetris import Tetris
from datetime import datetime
from statistics import mean, median
import random
# from logs import CustomTensorBoard
from tqdm import tqdm
from keras.models import load_model
import numpy as np

RENDER_DELAY = None

class TestAgent:

    def __init__(self):
        self.model = load_model('test_model.h5')
        self.state_size = 4

    def predict_value(self, state):
        '''Predicts the score for a certain state'''
        return self.model.predict(state)[0]

    def best_state(self, states):
        '''Returns the best state for a given collection of states'''
        max_value = None
        best_state = None

        for state in states:
            value = self.predict_value(np.reshape(state, [1, self.state_size]))
            if not max_value or value > max_value:
                max_value = value
                best_state = state

        return best_state

if __name__ == "__main__":
    test_model = TestAgent()
    env = Tetris()

    current_state = env.reset()
    done = False
    steps = 0
    render = True

    # Game
    while not done :
        next_states = env.get_next_states()
        best_state = test_model.best_state(next_states.values())
        
        best_action = None
        for action, state in next_states.items():
            if state == best_state:
                best_action = action
                break

        reward, done = env.play(best_action[0], best_action[1], render=render,
                                render_delay=RENDER_DELAY)
        
        # agent.add_to_memory(current_state, next_states[best_action], reward, done)
        current_state = next_states[best_action]
        steps += 1

