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
        self.rewards=np.zeros(n_states)
        #self.pi=np.zeros((n_states,n_states))
        
    def select_action(self,s):
        ''' Returns the greedy best action in state s ''' 
        e=0.2*np.random.randint(0,5,1)# TO DO: Add own code
        #a = np.random.randint(0,self.n_actions) # Replace this with correct action selection
        #a=np.random.randint(4, size=(1,self.n_actions))/100
        
        if e < 0.75:
            #print(self.Q_sa[s])
            a=np.argmax(self.Q_sa[s])
        else:
            a = np.random.randint(0,self.n_actions)
        return a
        
    def update(self,s,a,p_sas,r_sas,s_next):
        ''' Function updates Q(s,a) using p_sas and r_sas '''
        # TO DO: Add own code
        self.Q_sa[s][a]+=p_sas*(r_sas+self.gamma*self.Q_sa[s_next][a])
        pass
    
    
    
def Q_value_iteration(env, gamma=0.99, threshold=0.1,delta=0):
    ''' Runs Q-value iteration. Returns a converged QValueIterationAgent object '''
    
    QIagent = QValueIterationAgent(env.n_states, env.n_actions, gamma,delta)    
        
    # TO DO: IMPLEMENT Q-VALUE ITERATION HERE
    done=False
    s=env.reset()
    while done!=True:
        a=np.random.randint(0,4,1)[0]#np.argmax(QIagent.Q_sa[s])##QIagent.select_action(s)
        s_next,r,done=env.step(a)
        tmp=np.random.random()#np.random.randint(4, size=(1,env.n_actions))/100#
        QIagent.Q_sa[s][a]=tmp
        s=s_next
    # Plot current Q-value estimates & print max error
    #env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.2)
    #print("Q-value iteration, iteration {}, max error {}".format(i,max_error))
     
    return QIagent

def experiment():
    import matplotlib.pyplot as plt
    gamma = 1.00
    threshold = 0.1
    delta=0
    env = StochasticWindyGridworld(initialize_model=True)
    QIagent = Q_value_iteration(env,gamma,threshold,delta)
    
    # View optimal policy
    while delta < 1.4:#threshold * 1- gamma /gamma:
        biggest_change = 0
        s=env.reset()
        done=False
        rew=0
        while done!=True:
                a=QIagent.select_action(s)
                s_n,r,done=env.step(a)
                psr=a/env.n_actions
                tmp=QIagent.Q_sa[s][a]
                rew+=r
                QIagent.rewards[s]=r
                QIagent.update(s, a, psr, r,s_n)
                delta=max(delta,tmp-QIagent.Q_sa[s][a])
                s=s_n    
    #env.render(Q_sa=QIagent.Q_sa,plot_optimal_policy=True,step_pause=0.5)
    r=QIagent.rewards
    print(r)
    valiter=[max(QIagent.Q_sa[s]) for s in range(0,len(r))]
    # TO DO: Compute mean reward per timestep under the optimal policy
    # print("Mean reward per timestep under optimal policy: {}".format(mean_reward_per_timestep))
    print('This is the average reward {}'.format(np.average(r)))
    print('This is the average value Function {}'.format(np.average(valiter)))
    plt.figure(figsize=(13, 13))
    plt.plot(valiter)
    plt.plot(r)
    plt.xlabel(f'Number of states')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.title('This is the average reward {}'.format(np.average(r)))
    plt.savefig('rewardsValueIteration.png')
    plt.show()

if __name__ == '__main__':
    experiment()
