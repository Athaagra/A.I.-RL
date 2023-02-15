#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Practical for course 'Reinforcement Learning',
Leiden University, The Netherlands
2021
By Thomas Moerland
"""

import numpy as np
from Environment import StochasticWindyGridworld
from Helper import softmax, argmax

class NstepQLearningAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma, n):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.n = n
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
        
    def update(self, states, actions, rewards, done):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        # TO DO: Add own code
        pass

def n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True, n=5):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    pi = NstepQLearningAgent(env.n_states, env.n_actions, learning_rate, gamma, n)
    t = 0 # in which step the agent is
    t_update = 0 # where we are updating the policy, because always behind t (tau in formula)
    terminal_state_index = np.inf # where the Terminal state in the episode is, if we found it (T in formula)    
    # for calculating average return
    steps = 0
    returns = 0    
    rewards = []
    actions=[]
    
    done=False
    state=env.reset()
    # TO DO: Write your n-step Q-learning algorithm here!
    for i in range(0,n_timesteps):
        print(i)
        action=pi.select_action(state,policy,epsilon)
        next_state,reward,done=env.step(action)
        steps+=1
        returns+=reward
        print(reward)
        actions.append(action)
        # remember state and reward for later policy updates
        state= np.append(state) # state[t+1]
        reward = np.append(reward) # reward[t]
        pi.update(state,action,reward,done)
        state=next_state
        rewards.append(reward)
        #if timesteps 
        if done:
            terminal_state_index=t+1
        visited_states=[]
        while done or t_update + n <= t:
            #if np.mean(np.equal(state[t_update],np.array(self.gridworld.getTerminal()))) == 1:
            #        break
            # implement first visited check for Monte Carlo (Is this the first time we are here in this episode)
            if(n == np.inf):
                for visited_state in visited_states:
                    if np.mean(np.equal(state[t_update],visited_state)) == 1:
                            break
            visited_states.append(state[t_update])

            # calcualte value for n steps or until the terminal if found
            mc_estimate = np.sum([pi.gamma**(i-t_update) * reward[i] for i in range(t_update,min(t_update+n,terminal_state_index))])
            print('This is the mc state {}'.format(mc_estimate))
            future_estimate = 0
            if t_update+n < terminal_state_index: # if we are not yet at the terminals state
            # calculate the estimate after n
                future_estimate =  pi.gamma**n * pi.Q_sa[state[t_update+n],action[t_update+n]]
            estimate = mc_estimate + future_estimate
            print('This is the estimate {}'.format(estimate))                    
            # improve policy
            pi.Q_sa[state[t_update],action[t_update]] += pi.alpha * (estimate - pi.Q_sa[state[t_update],action[t_update]])               
            t_update += 1                                 
                
        t += 1
    if plot:
        env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during n-step Q-learning execution

    return rewards 

def test():
    n_timesteps = 10000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1
    n = 5
    
    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    rewards = n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot, n=n)
    print("Obtained rewards: {}".format(rewards))    
    
if __name__ == '__main__':
    test()
