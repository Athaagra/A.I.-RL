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

    def __init__(self, n_states, n_actions, learning_rate, gamma, n,virtual_env):
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.n = n
        self.Q_sa = np.zeros((n_states,n_actions))
        self.state = np.array([virtual_env.reset()],dtype=int) # [[y,x],[y,x],...]
        self.action = np.empty(shape=(0), dtype=int)
        self.reward = np.empty(shape=(0), dtype=int)
        self.done=np.empty(shape=(0), dtype=int)
    def select_action(self, s, policy='egreedy', epsilon=None, temp=None):
        
        if policy == 'egreedy':
            if epsilon is None:
                raise KeyError("Provide an epsilon")
            
            if epsilon < np.random.randint(0,101,1)/100:
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
        
    def update(self, states, actions, rewards, dones):
        ''' states is a list of states observed in the episode, of length T_ep + 1 (last state is appended)
        actions is a list of actions observed in the episode, of length T_ep
        rewards is a list of rewards observed in the episode, of length T_ep
        done indicates whether the final s in states is was a terminal state '''
        # TO DO: Add own code
        self.action = np.append(self.action,[actions],axis=0)
        self.state = np.append(self.state,[states],axis=0) # state[t+1]
        self.reward = np.append(self.reward,[rewards],axis=0) # reward[t]
        self.done = np.append(self.done, [dones],axis=0)
        pass

def n_step_Q(n_timesteps, max_episode_length, learning_rate, gamma, 
                   policy='egreedy', epsilon=None, temp=None, plot=True, n=5):
    ''' runs a single repetition of an MC rl agent
    Return: rewards, a vector with the observed rewards at each timestep ''' 
    
    
    env = StochasticWindyGridworld(initialize_model=False)
    pi = NstepQLearningAgent(env.n_states, env.n_actions, learning_rate, gamma, n,env)
    t = 0 # in which step the agent is
    # for calculating average return
    steps = 0
    returns = 0    
    rewards = []
    #actions=[]
    
    done=False
    state=env.reset()
    # TO DO: Write your n-step Q-learning algorithm here!
    for step in range(0,n_timesteps):
        epsl=epsilon
        for t in range(0,max_episode_length):
            epsl+=2 *t/10000
            action=pi.select_action(state,policy,epsl)
            next_state,reward,done=env.step(action)
            steps+=1
            rewards.append(reward)
            # remember state and reward for later policy updates
            pi.update(state,action,reward,done)
            state=next_state
            #if timesteps 
            if done:
                print('You have won the game!')
                break
        Tep=t + 1
        G={}
        for i in range(-Tep+1,0):
            it=abs(i)-n
            m=min(n,Tep-i)
            if it+m<max_episode_length and pi.done[it+m]==True:
                #mc_estimate 
                G[it]= np.sum([pi.gamma**(imc) * pi.reward[it+imc] for imc in range(0,m-1)])
            # implement first visited check for Monte Carlo (Is this the first time we are here in this episode)
            else:
                #future_estimate 
                G[it]=np.sum([pi.gamma**(ins) * pi.reward[it+ins] for ins in range(0,m-1)])# + pi.gamma**(m) * max(pi.Q_sa[pi.state[it+m],pi.action[it+m]]))
                G[it]=+ pi.gamma**(m) * max(pi.Q_sa[pi.state[it+m]])
            pi.Q_sa[pi.state[it],pi.action[it]] += pi.learning_rate * (G[it] - pi.Q_sa[pi.state[it],pi.action[it]])
    if plot:
        env.render(Q_sa=pi.Q_sa,plot_optimal_policy=True,step_pause=0.1) # Plot the Q-value estimates during n-step Q-learning execution

    return rewards 

def test():
    import matplotlib.pyplot as plt
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
    plt.figure(figsize=(13, 13))
    plt.plot(rewards)
    plt.xlabel(f'Number of episode')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.title('This is the average reward {}'.format(np.average(rewards)))
    plt.savefig('rewardsNsteps'+str(n_timesteps)+'steps'+str(gamma)+'gamma'+str(learning_rate)+'learningrate'+str(epsilon)+'epsilon'+str(temp)+'temperature.png')
    plt.show()
if __name__ == '__main__':
    test()
