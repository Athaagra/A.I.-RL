#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2022
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import softmax, argmax

class SarsaAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))
        
    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):
        
        if policy == 'egreedy':
            if epsilon is None:
                raise KeyError("Provide an epsilon")
                
            if epsilon < sum(self.Q_sa[s][:2]):
            # TO DO: Add own code
                a = np.random.randint(0,self.n_actions) # Replace this with correct action selection
            else:
                a=np.argmax(self.Q_sa[s])
            
                
        elif policy == 'softmax':
            if temp is None:
                raise KeyError("Provide a temperature")
            if temp is not None:    
            # TO DO: Add own code
                #a = np.random.randint(0,self.n_actions) # Replace this with correct action selection
                actions=[np.exp(self.Q_sa[s][i]/temp)/sum(np.exp(self.Q_sa[s]/temp)) for i in range(0,self.n_actions)]
                actions=np.array(actions)
                a=np.argmax(actions)
        return a
        
    def update(self,s,a,r,s_next,a_next,done):
        # TO DO: Add own code
        self.Q_sa[s][a]=(1-self.learning_rate)*self.Q_sa[s][a]+self.learning_rate*( r+ self.gamma * (self.Q_sa[s_next][a_next])-self.Q_sa[s][a])
        pass
        
def sarsa(n_timesteps, learning_rate, gamma, policy='egreedy', epsilon=None, temp=None, plot=True):
    ''' runs a single repetition of SARSA
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    pi = SarsaAgent(env.n_states, env.n_actions, learning_rate, gamma)
    rewards = []

    # TO DO: Write your SARSA algorithm here!
    state=env.reset()
    done=False
    #while done!=True:
    for i in range(0,n_timesteps):
        print(i)
        action=pi.select_action(state,policy,epsilon)
        next_state,reward,done=env.step(action)
        next_action=pi.select_action(next_state,policy,epsilon)
        print(reward)
        pi.update(state,action,reward,next_state,next_action,done)
        state=next_state
        rewards.append(reward)
        if done==True:
            break
    if plot:
        env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during SARSA execution

    return rewards 


def test():
    import matplotlib.pyplot as plt
    n_timesteps = 1000
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    rewards = sarsa(n_timesteps, learning_rate, gamma, policy, epsilon, temp, plot)
    print("Obtained rewards: {}".format(rewards))        
    plt.figure(figsize=(13, 13))
    plt.plot(rewards)
    plt.xlabel(f'Number of episode')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.title('This is the average reward {}'.format(np.average(rewards)))
    plt.savefig('rewardsQ-learningAgent'+str(n_timesteps)+'steps'+str(gamma)+'gamma'+str(learning_rate)+'learningrate'+str(epsilon)+'epsilon'+str(temp)+'temperature.png')
    plt.show()
if __name__ == '__main__':
    test()
