For the Third Assignment of Reinforcement learning
There are 3 main fucntions the function DQN(Deep Q learning),
##########################DQN####################################
to make more experiments you need to change the variables manually,
line 32: learning_rate of optimizer
line 33: number of episodes 
line 34: gamma (discount factor) 
line 36: dense layer
line 38: the size of the buffer
line 39: the size of the batch(number of training data)
line 43: egreedy  (starting value of e-greedy (high number big exploration))
line 44: egreedy final (final value of e-greedy)
line 45: egreedy decay (rate of decreasing value)
line 102: optimizer

Import packages:
torch
gym
random(no installation, default on anaconda)
math(no installation, default on anaconda)
time(no installation, default on anaconda)
matplotlib(no installation, default on anaconda)

All figures are generated and save in form of png
The best implementation results:

DQN: Avg reward 16

All parameters are written on the report!