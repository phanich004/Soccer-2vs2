# Soccer Agents with PPO Implementation


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

ML-Agents is an open-source plugin for creating intelligent agents in Unity games and simulations using reinforcement learning, imitation learning, neuroevolution, or other machine learning methods through a simple-to-use Python API.

## Installation

### Prerequisites

- Python 3.6+
- Unity 2018.4 or later

### Steps

1. **Clone the repository:**

    ```sh
    git clone https://github.com/Unity-Technologies/ml-agents
    cd ml-agents
    ```

2. **Create and activate a new conda environment:**

    ```sh
    conda create --name rl python=3.10.12
    conda activate rl
    ```

3. **Install the ML-Agents package:**

    ```sh
    pip install -e ./ml-agents-envs
    pip install -e ./ml-agents
    ```


This repository contains reinforcement learning agents trained for a Unity ML-Agents soccer simulation using the Proximal Policy Optimization algorithm. The provided Unity environment, `SoccerTwos.app`, simulates a 2v2 soccer scenario.

## Files Overview
- **1striker.py**: Trains a single striker agent.
- **2v2_stationary.py**: Trains a 2v2 soccer game with stationary opponents.
- **2v2_random.py**: Trains a 2v2 soccer game with randomly moving opponents.

## Setup and Requirements
Install requirements.txt
Install the given SoccerTwos excetuable.

git clone https://github.com/Unity-Technologies/ml-agents.git

Create a file named training-envs-executables in the ml-agents file and place the SoccerTwos excetuable in it.


### Clone the Repository

git clone https://github.com/phanich004/Soccer-2vs2.git

run python 1striker.py to run the python files

