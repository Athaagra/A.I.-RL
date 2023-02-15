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
from Helper import argmax

class QValueIterationAgent:
    ''' Class to store the Q-value iteration solution, perform updates, and select the greedy action '''

    def __init__(self, n_states, n_actions, gamma, delta, threshold=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.delta = delta
        self.Q_sa = np.zeros((n_states,n_actions))
        
    def select_action(self,s):
        ''' Returns the greedy best action in state s ''' 
        #e=0.001*np.random.randint(0,2,1)# TO DO: Add own code
        #if e < 0.5:
        Q_sa = self.Q_sa
        #a = np.random.randint(0,self.n_actions) # Replace this with correct action selection
        a=np.random.randint(4, size=(1,self.n_actions))/100
        #print('This is the actions {}'.format(a))
        Q_sa[s]=a
        #else:
        a=argmax(Q_sa[s])
        return a,self.Q_sa[s][a]
        
    def update(self,s,a,p_sas,r_sas):
        ''' Function updates Q(s,a) using p_sas and r_sas '''
        # TO DO: Add own code
        
        paa=np.sum(r_sas + self.gamma * self.Q_sa[p_sas]) *self.Q_sa[s][a] 
        self.delta = max(float(self.delta), abs(float(self.Q_sa[s][a])-float(paa)))
        self.Q_sa[s][a]=abs(float(self.Q_sa[s][a])-float(paa))
        #if self.delta <=self.gamma *(1-self.gamma)/self.gamma:
         #   print('This is the delta {} and stop'.format(self.delta))
        pass
    
    
    
def Q_value_iteration(env, gamma=0.99, threshold=0.01,delta=0):
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''
    
    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma,delta)    
        
    # TO DO: IMPLEMENT Q-VALUE ITERATION HERE
        
    # Plot current Q-value estimates & print max error
    # env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.2)
    # print("Q-value iteration, iteration {}, max error {}".format(i,max_error))
     
    return QIagent

def experiment():
    gamma = 0.99
    threshold = 0.001
    delta=0
    env = StochasticWindyGridworld(initialize_model=True)
    env.render()
    QIagent = Q_value_iteration(env,gamma,threshold,delta)
    #print('This is the Q_sa {}'.format())
    # View optimal policy
    done = False
    s = env.reset()
    print('This is the state {}'.format(s))
    while not s==52:
        a,pa = QIagent.select_action(s)
        print('This is the action {} and the probabilit action {}'.format(a,pa))
        s_next, r, done = env.step(a)
        print('this is the reward {}'.format(r))
        QIagent.update(s, a, s_next, r)
        print('This is the state {}'.format(s))
        #Delta=abs(pa-QIagent.Q_sa[s][a])
        #print(Delta)
        env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.5)
        s = s_next
    #Delta < eta
    # TO DO: Compute mean reward per timestep under the optimal policy
    # print("Mean reward per timestep under optimal policy: {}".format(mean_reward_per_timestep))

if __name__ == '__main__':
    experiment()
