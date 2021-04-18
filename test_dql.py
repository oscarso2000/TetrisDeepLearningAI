from tqdm import tqdm
# from keras.models import load_model
import numpy as np
import tensorflow as tf


class TestAgent:

    def __init__(self):
        self.model = tf.keras.models.load_model('dql_model.h5')
        self.state_size = 4

    def predict_value(self, state):
        '''Predicts the score for a certain state'''
        return self.model.predict(state)[0]

    def best_state(self, states):
        '''Returns the best state for a given collection of states'''
        max_value = None
        best_state = None
        best_action = None

        for state in states.values():
                
            value = self.predict_value(np.reshape(state, [1, self.state_size]))
            # value = self.predict_value( tf.convert_to_tensor(np.reshape(np.array(state), [1, self.state_size]) ,  dtype=tf.float64   )  )
            if not max_value or value > max_value:
                max_value = value[0]
                best_state = state

        for action, state in states.items():
            if state == best_state:
                best_action = action
                break

        return best_action, best_state, max_value


if __name__ == "__main__":
    print("hi")
    test_model = TestAgent()
    
    # env = Tetris()

    # current_state = env.reset()
    # done = False
    # steps = 0

    # # Game
    # while not done :
    #     next_states = env.get_next_states()
    #     best_state = test_model.best_state(next_states.values())
        
    #     best_action = None
    #     for action, state in next_states.items():
    #         if state == best_state:
    #             best_action = action
    #             break
    #     print(best_action)
    #     reward, done = env.play(best_action[0], best_action[1], render=RENDER,
    #                             render_delay=RENDER_DELAY)
        
    #     current_state = next_states[best_action]
    #     steps += 1

