# TetrisDeepLearningAI

By Oscar So and Mike Russo 

## Table of Contents

Project Goal: To research different training approaches for a Deep Q-Learning algorithm in our Tetris game. 


## Getting Started 

1. [Accessing the source code](#accessing-the-source-code)
1. [Prerequisites](#prerequisites)
1. [Installation](#installation)
1. [How to Run](#how-to-run)


## Accessing the Source Code

```
bash
git clone
```

## Prerequisites

- Python ([link](https://www.python.org/downloads/ ))

```
Verify with pip 

# Unix/macOS
$ python -m pip --version

# Windows
C:\> py -m pip --version

```


## Installation

```
# we suggest to work in a new environment before installing

# packages
pip install -r /path/to/requirements.txt

# Tensorflow

# Requires the latest pip
pip install --upgrade pip

# Current stable release for CPU and GPU
pip install tensorflow

```
- More info on [TensorFlow](https://www.tensorflow.org/install)


## How to Run

- At the bottom of the [main.py](main.py) file, comment out the function you want to run.

    - train_dql_agent(discount, epsilon_stop_episode, learning_rate, step_size, model, logs_file)
        - discount: Gamma value. close to 0 favors immidiate rewards. close to 1 favors long term rewards
        - epsilon_stop_episode: When we stop "exploring"
        - Learning rate: How rewards update. If learning rate is 1, use first model
        - Step size: How often model trains after enough memory is collected 
        - Model: models file name. Saved under [train_models/](trained_models)
        - logs_file: records episode #, min, max, average. Saved under [scores/](scores)
        - creates a Tetris environment from [tetris_trainer.py](tetris_trainer.py) 
        - create an AI from [dqn_agent](dqn_agent.py)
        - calls start_dql() which trains the model, located in [train_ai.py](train_ai.py)
            - change model name here 
            - change log directory and name in [logs.py](logs.py) 

    - test_dql_agent()
        - make sure the correct model is called in [test_dql.py](test_dql.py) 

- Update heuristics, replay size, epsilon decay rates, etc.
    - All located in [dqn_agent](dqn_agent.py)
    - change reward declaration in best_state()

```
# navigate to repo location
python main.py 

```

