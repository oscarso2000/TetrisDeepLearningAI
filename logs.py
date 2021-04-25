from keras.callbacks import TensorBoard
import tensorflow as tf
import pandas as pd 
from datetime import datetime
from matplotlib import pyplot as plt

class CustomTensorBoard(TensorBoard):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self.d =  {'episodes_':[], 'mean_': [], 'max_': [], 'min_':[]}
        self.dataframe = pd.DataFrame(data=self.d)

    def set_model(self, model):
        pass

    def _add(self,episodes_, mean_, max_, min_, file_name):
        self.dataframe = self.dataframe.append({'episodes_':episodes_, 'mean_': mean_, 'max_': max_, 'min_': min_}, ignore_index=True)
        self.dataframe.to_csv(file_name, index = False)

    def log(self, episode,avg_score, max_score, min_score):
        with self.writer.as_default():
            tf.summary.scalar('Tetris max Score', data=max_score, step=episode)



