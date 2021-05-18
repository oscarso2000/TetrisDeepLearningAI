from keras.models import Sequential , save_model, load_model
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from collections import deque
import numpy as np
import random
import tensorflow as tf
from datetime import datetime
import math

class DQNAgent:

    '''Deep Q Learning Agent + Maximin
    Args:
        state_size (int): Size of the input domain
        mem_size (int): Size of the replay buffer
        discount (float): How important is the future rewards compared to the immediate ones [0,1]
        epsilon (float): Exploration (probability of random values given) value at the start
        epsilon_min (float): At what epsilon value the agent stops decrementing it
        epsilon_stop_episode (int): At what episode the agent stops decreasing the exploration variable
        n_neurons (list(int)): List with the number of neurons in each inner layer
        activations (list): List with the activations used in each inner layer, as well as the output
        loss (obj): Loss function
        optimizer (obj): Otimizer used
        replay_start_size: Minimum size needed to train
    '''
 #   mem_size=10000,
    def __init__(self, 
        state_size, 
        n_neurons, 
        activations,
        epsilon_stop_episode,
        mem_size,
        discount,
        replay_start_size,
        learning_rate, 
        epsilon=1,
        epsilon_min=0, 
        loss='mse',
        optimizer='adam'):

        assert len(activations) == len(n_neurons) + 1
        self.file_path = '/tmp/checkpoint'
        self.state_size = state_size
        self.mem_size = 10000
        self.epochs = 1
        self.memory = deque(maxlen=mem_size)
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_stop_episode = epsilon_stop_episode
        self.batch_size = 512
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / (self.epsilon_stop_episode)
        self.n_neurons = n_neurons
        self.activations = activations
        self.loss = loss
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        if not replay_start_size:
            replay_start_size = mem_size / 2
        self.replay_start_size = replay_start_size
        # self.model = self._build_model()
        self.model = self.build_lr_model()
        self.log_dir = f'logs/tetris-nn={str(self.n_neurons)}-mem={self.mem_size}-bs={self.batch_size}-e={self.epochs}-{datetime.now().strftime("%Y_%m_%d")}'
        self.cp_callbacks = [
                            # tf.keras.callbacks.ModelCheckpoint(filepath= self.file_path),
                            # tf.keras.callbacks.TensorBoard(log_dir=self.log_dir),
                            tf.keras.callbacks.CSVLogger('scores/test_dql.csv'),]


    def _build_model(self):
        '''Builds a Keras deep neural network model'''
        model = Sequential()
        model.add(Dense(self.n_neurons[0], input_dim=self.state_size, activation=self.activations[0]))

        for i in range(1, len(self.n_neurons)):
            model.add(Dense(self.n_neurons[i], activation=self.activations[i]))

        model.add(Dense(1, activation=self.activations[-1]))

        model.compile(loss=self.loss, optimizer=self.optimizer)
       
        return model

    def build_lr_model(self):
        '''Builds a model with a learning rate. Source:  https://towardsdatascience.com/deep-q-learning-tutorial-mindqn-2a4c855abffc '''
        init = tf.keras.initializers.HeUniform()
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu', kernel_initializer=init))
        model.add(Dense(12, activation='relu', kernel_initializer=init))
        model.add(Dense(1, activation='linear', kernel_initializer=init))
        model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate), metrics=['accuracy'])
        return model


    def add_to_memory(self, current_state, next_state, reward, done):
        '''Adds a play to the replay memory buffer'''
        self.memory.append((current_state, next_state, reward, done))


    def random_value(self):
        '''Random score for a certain action'''
        return random.random()


    def predict_value(self, state):
        '''Predicts the score for a certain state'''
        return self.model.predict(state)[0]


    def act(self, state):
        '''Returns the expected score of a certain state'''
        state = np.reshape(state, [1, self.state_size])
        if random.random() <= self.epsilon:
            return self.random_value()
        else:
            return self.predict_value(state)

    def best_state(self, states):
        '''Returns the best state for a given collection of states'''
        max_value = -math.inf
        best_state = None
        if random.random() <= self.epsilon:
            best_state = random.choice(list(states.values()))
            #Reward Heuristics are changed here
            #[rows_cleared, hole_count, rough, cum_height] + [floor_blocks, get_contig_sections]
            reward = (1 + (best_state[0] ** 2) * 15) + (-7.51 * best_state[3] ) + (-10.2* best_state[1]) + (-2.18* best_state[2]) + (best_state[4] * 15) + (best_state[5]* -5.55)

            return best_state, reward

        else:
            for state in states.values():
                value = self.predict_value(np.reshape(state, [1, self.state_size]))
                if not max_value or value > max_value:
                    max_value = value[0]
                    best_state = state
            return best_state, max_value 



    def train(self, batch_size=32, epochs=3):
        '''Trains the agent'''
        n = len(self.memory)

        if n >= self.replay_start_size and n >= batch_size:
            batch = random.sample(self.memory, batch_size)
            next_states = np.array([x[1] for x in batch])
            next_qs = [x[0] for x in self.model.predict(next_states)]
            x = []
            y = []

            # Build xy structure to fit the model in batch (better performance)
            for i, (state, _, reward, done) in enumerate(batch):
                if not done:
                    # Partial Q formula
                    new_q = reward + self.discount * next_qs[i]
                else:
                    new_q = reward

                x.append(state)
                y.append(new_q)

               
            self.model.fit(np.array(x), np.array(y), batch_size=batch_size, epochs=epochs, verbose=0)
            # Update the exploration variable
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.epsilon_decay




