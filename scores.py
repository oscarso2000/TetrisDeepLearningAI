import pandas as pd 
from datetime import datetime
from matplotlib import pyplot as plt



def plot_():
    scores = pd.read_csv("scores/test_scores_rule_based_dql.csv")
    plt.plot(scores.episodes_, scores.max_, color = 'red')
    plt.xlabel("Episodes")
    plt.ylabel("Max Score")
    plt.title("Rule Based Deep Q Learning")
    plt.show()

if __name__ == "__main__":
    plot_()