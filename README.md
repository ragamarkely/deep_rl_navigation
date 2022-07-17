[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Trained Agent"

# Project 1: Navigation

### Introduction

This project is part of [Udacity Deep Reinforcement Learning Nanodegree](https://www.udacity.com/course/deep-reinforcement-learning-nanodegree--nd893). This project aims to develop and train a Deep Reinforcement Learning (RL) agent to navigate and collect bananas in one of [Unity environments](https://github.com/Unity-Technologies/ml-agents/blob/main/docs/Learning-Environment-Examples.md). 

The goal of the agent is to navigate in the environment to collect as many yellow bananas as possible while avoiding blue bananas. A reward of +1 is given for collecting a yellow banana, and -1 for a blue banana. 

The state space has 37 dimensions and contains the agent's velocity, along with ray-based perception of objects around agent's forward direction.  Given this information, the agent has to learn how to best select actions.  Four discrete actions are available, corresponding to:
- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The task is episodic, and the agent is considered successful in solving the environment when it gets an average score of $\geq$ 13 over 100 consecutive episodes. An example of a trained agent is shown in the following figure.

![Trained Agent][image1]

### Instructions

1. **Dependencies.** Set up the python environment by following [these steps](https://github.com/udacity/Value-based-methods#dependencies).   
2. **Unity Environment.** Set up the Unity environment by following [these steps](https://github.com/udacity/deep-reinforcement-learning/blob/master/p1_navigation/README.md).  
3. **Deep Reinforcement Learning Agent.** A Double Deep Q-Network (Double DQN) agent is used to solve the environment in this project. The agent is defined in agent.py file, and each of the two Deep Q-Network is defined in model.py. The Navigation.ipynb notebook contains a script for training the agent and demonstrates that the agent can successfully solve the environment within a few hundred episodes.

