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
# We suggest to work in a new environment before installing

# packages
pip install -r /path/to/requirements.txt

# Tensorflow

# Requires the latest pip
pip install --upgrade pip

# Current stable release for CPU and GPU
pip install tensorflow

```
- More info on [TensorFlow](https://www.tensorflow.org/install)

## Setting up your environment

**System Configuration**

Perform the following steps in order:

#### 1. Check your Version of Python3 (should be 3.7.6)

You can check via:

````bash
> python3 --version
Python 3.7.6
````

If your version differs, then download `3.7` [`here`](https://www.python.org/downloads/).

####  2. Check that Pip is Installed and Up-to-date.  
You should already have pip installed if you have Python downloaded from python.org. Make sure that yours is up-to-date.

Upgrade pip :
````bash
> python3 -m pip install -U pip
````
If not, install it following instructions. (also found [`here`](https://pip.pypa.io/en/stable/installing/))
````bash
> curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

> python3 get-pip.py
````
#### 3. Download Virtualenv

[`Virtualenv`](https://virtualenv.pypa.io/en/stable/installation/) helps establish an isolated `Python` environment.  The environment allows you to separate project-specific dependencies and their versions from the `Python` modules installed locally on your computer.  Once you have `virtualenv`, `cd` into the directory where the extracted assignment is stored (e.g. assignment1), and run:
````bash
> virtualenv -p python3 venv
````

This creates a virtual environment called `venv`.  In order to enter than virtual environment, run the following:

Linux or MacOS:
````bash
> source venv/bin/activate
````
Windows:
````bat
> venv\Scripts\activate

The following command line prompt will indicate that you’re in the virtual environment:

````bash
(venv) >
````

To deactivate the virtual environment, run the following:

````bash
(venv) > deactivate
>
````

Whenever you work with this project, you should **always** be in your virtual environment.  Without this isolation, we might run into module versioning issues and other problems when trying to run your project, which creates administrative overhead.  

#### 4. Install Dependencies

At the root of directory of the project skeleton code, run the following:

````bash
(venv) > pip3 install -r requirements.txt
````

This installs within your virtual environment all the necessary modules that are required at the beginning of the project.


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

