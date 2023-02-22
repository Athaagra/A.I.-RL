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

class MonteCarloAgent:

    def __init__(self, n_states, n_actions, learning_rate, gamma,virtual_env):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.Q_sa = np.zeros((n_states,n_actions))
        self.state = np.array([virtual_env.reset()],dtype=int) # [[y,x],[y,x],...]
        self.action = np.empty(shape=(0), dtype=int)
        self.reward = np.empty(shape=(0), dtype=int)
        self.done=np.empty(shape=(0), dtype=int)
        
    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):
        
        if policy == 'egreedy':
            if epsilon is None:
                raise KeyError("Provide an epsilon")
                
            if epsilon < np.random.randint(0,2,1):
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
        
    def update(self, states, actions, rewards,dones,t,terminal_state_index):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        # TO DO: Add own code
        t_update = 0 # where we are updating the policy, because always behind t (tau in formula)
        self.action = np.append(self.action,[actions],axis=0)
        self.state = np.append(self.state,[states],axis=0) # state[t+1]
        self.reward = np.append(self.reward,[rewards],axis=0) # reward[t]
        self.done = np.append(self.done, [dones],axis=0)
        visited_states=[]
        while dones or t_update <= t:
            if np.mean(np.equal(self.state[t_update],np.array(52))) == 1:
                    break
            # implement first visited check for Monte Carlo (Is this the first time we are here in this episode)
            #print('this is the state {}'.format(states))
            if(states == np.inf):
                for visited_state in visited_states:
                    if np.mean(np.equal(self.state[t_update],visited_state)) == 1:
                            break
            visited_states.append(self.state[t_update])
            # calcualte value for n steps or until the terminal if found
            estimate = np.sum([self.gamma**(i-t_update) * self.reward[i] for i in range(t_update,min(t_update,terminal_state_index))])
            #print('This is the mc estim22ate {}'.format(estimate))                 
            # improve policy
            self.Q_sa[self.state[t_update],self.action[t_update]] += self.learning_rate * (estimate - self.Q_sa[self.state[t_update],self.action[t_update]])               
            t_update += 1                               
        pass

def monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    env = StochasticWindyGridworld(initialize_model=False)
    pi = MonteCarloAgent(env.n_states, env.n_actions, learning_rate, gamma,env)
    rewards = []
    

    # TO DO: Write your Monte Carlo RL algorithm here!
    t = 0 # in which step the agent is
    terminal_state_index = np.inf# where the Terminal state in the episode is, if we found it (T in formula)    
    # for calculating average return
    steps = 0
    returns = 0    
    rewards = []
    #actions=[]
    
    done=False
    state=env.reset()
    # TO DO: Write your n-step Q-learning algorithm here!
    for i in range(0,n_timesteps):
        action=pi.select_action(state,policy,epsilon)
        next_state,reward,done=env.step(action)
        print(action,reward)
        steps+=1
        returns+=reward
        rewards.append(reward)
        # remember state and reward for later policy updates
        pi.update(state,action,reward,done,t,terminal_state_index)
        state=next_state
        print(state)
        #if timesteps 
        if done:
            terminal_state_index=t+1
        t += 1
        #if i == max_episode_length:
        #    break
    if plot:
        env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during Monte Carlo RL execution

    return rewards 
    
def test():
    import matplotlib.pyplot as plt
    n_timesteps = 10000
    max_episode_length = 100
    gamma = 1.0
    learning_rate = 0.1

    # Exploration
    policy = 'egreedy' # 'egreedy' or 'softmax' 
    epsilon = 0.1
    temp = 1.0
    
    # Plotting parameters
    plot = True

    rewards = monte_carlo(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy, epsilon, temp, plot)
    print("Obtained rewards: {}".format(rewards))  
    plt.figure(figsize=(13, 13))
    plt.plot(rewards)
    plt.xlabel(f'Number of episode')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.title('This is the average reward {}'.format(np.average(rewards)))
    plt.savefig('rewardsMonteCarlo'+str(n_timesteps)+'steps'+str(gamma)+'gamma'+str(learning_rate)+'learningrate'+str(epsilon)+'epsilon'+str(temp)+'temperature.png')
    plt.show()
            
if __name__ == '__main__':
    test()
