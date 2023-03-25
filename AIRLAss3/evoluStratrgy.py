#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 13:54:49 2023

@author: Optimus
"""
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import sys
HISTORY_LENGTH = 0  
# #hyperparameters
D = 4#len(env.reset())*HISTORY_LENGTH
M = 32
K = 12


actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
#print('This is the D {} M {} and K {}'.format(D,M,K))
#print('This is the D {} M {} and K {}'.format(D,M,K))
#def softmax(a):
#   c = np.max(a, axis=1, keepdims=True)
#   e = np.exp(a-c)
#   return e/e.sum(axis=-1, keepdims=True)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

   
def relu(x):
   return x *(x >0)
    
class ANN:
    #def __init__(self, D, K, f=relu):
    def __init__(self, D, M, K, f=relu):
        self.D=D
        self.M=M
        self.K=K
        self.f=f
    def init(self):
        #D, K = self.D, self.K
        #self.W1 = np.random.randn(D, K) /np.sqrt(D)
        #self.b1 = np.zeros(K)        
        D, M, K = self.D, self.M, self.K
        self.W1 = np.random.randn(D, M) /np.sqrt(D)
        self.b1 = np.zeros(M)
        self.W2 = np.random.randn(M, K)/ np.sqrt(M)
        self.b2 = np.zeros(K)
            
    def forward(self, X):
        #self.f
        #print('This is the input data X{}'.format(X))
        #print('This is the w1 {}'.format(self.W2))
        #print('This is the b1 {}'.format(self.b2.shape))
        Z = np.tanh(np.dot(X,self.W1)+ self.b1)
        #return softmax(Z) 
        return softmax(Z.dot(self.W2)+ self.b2)
        
    def sample_action(self, x):
        X=np.atleast_2d(x)
        #X=x
        Y=self.forward(X)
        #print('Forward process {}'.format(Y))
        y=Y[0]
        return np.argmax(y)
        
    def get_params(self):
        return np.concatenate([self.W1.flatten(), self.b1, self.W2.flatten(), self.b2])
        
    def get_params_dict(self):
        #return {
         #   'W1': self.W1,
         #   'b1': self.b1
          #  'W2': self.W2,
          #  'b2': self.b2
        #    }
        return {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2
            }
    def set_params(self, params):
        #D,K = self.D, self.K
        #self.W1 = params[:D * K].reshape(D, K)
        #self.b1 = params[D*K:D*K +K]
        D,M,K = self.D, self.M, self.K
        self.W1 = params[:D * M].reshape(D, M)
        self.b1 = params[D*M:D*M +M]
        self.W2 = params[D * M + M:D*M+M+M*K].reshape(M,K)
        self.b2 = params[-K:]
        
def evolution_strategy(f,population_size, sigma, lr, initial_params, num_iters):
    #assume initial params is a 1-D array
    num_params = len(initial_params)
    reward_per_iteration = np.zeros(num_iters)
    learning_rate = np.zeros(num_iters)
    sigma_v = np.zeros(num_iters)
    parms = np.zeros(num_iters)
    params = initial_params
    for t in range(num_iters):
        t0 = datetime.now()
        N = np.random.randn(population_size, num_params)
        ### slow way
        R = np.zeros(population_size)
        acts=np.zeros(population_size)
        print('This is the number of acts {}'.format(len(acts)))
        #loop through each "offspring"
        for j in range(population_size):
            params_try = params + sigma * N[j]
            R[j],acts[j],_ = f(params_try)
            #print('This is the action {}'.format(acts[j]))
            #print('This is the reward {}'.format(R[j]))
        m = R.mean()
        s = R.std()+0.001
        #print('This is s {}'.format(s))
        if s == 0:
            # we cannot apply the following equation
            print("Skipping")
            continue
        
        A = (R-m)/s
        reward_per_iteration[t]= m
        params = params + lr/(population_size*sigma)+np.dot(N.T, A)
        parms[t]=params.mean() 
        learning_rate[t]=lr
        sigma_v[t]=sigma
        #update the learning rate
        #lr *= 0.001
        lr *=0.992354
        sigma += 0.7891
        print("Iter:",t, "Avg Reward: %.3f" % m, "Max:", R.max(), "Duration:",(datetime.now()-t0))
        if m > 0.01:# or R.max() >= 1:#m > R.max()/1.5 or R.max() >= 1:
            actis = acts
            print('True')
            #break
        else:
            actis=np.zeros(population_size)
    return params, reward_per_iteration,actis,learning_rate,sigma_v,population_size,parms
    
def reward_function(params):
    #model=ANN(D,K)
    model = ANN(D, M, K)
    env=Qprotocol(4)
    inpu=[0]
    #models = ANN(D, M, K)
    model.set_params(params)
    #models.set_params(params)
#     # play one episode and return the total reward
    episode_reward = 0
    episode_length = 0
    done = False
    counterr=0
    state_n=env.reset(4,inpu)
    obs = state_n[0]#np.concatenate((, state_n[1]), axis=None)#state_n#obs[0]
    obs_dim= len(obs)
    if HISTORY_LENGTH >1:
        state =np.zeros(HISTORY_LENGTH*obs_dim)
        state[obs_dim:] = obs
    else:
        state = obs
    while not done:
#         #get the action
#         #state=np.array(state)
        #print('This is the state {}'.format(state))
        actiona = model.sample_action(state)
        action=np.array(actions_list[actiona])
#        actionb = model.sample_action(state)
   #     action=np.array(actions_list[action])
        #action=(actiona,actionb)
        #print('This is the action {}'.format(action))
        #print('This is the action {}'.format(action))
#         #perform the action
        new_state, reward, done,info=env.step(action)
#         #update total reward
        #done=do
        obs=new_state[0]
        #if do:
        #    counterr+=1
        #if counterr%2==0:
        done=done
# #        print(obs)
        episode_reward += reward
        episode_length +=1
#         #update state
        if HISTORY_LENGTH > 1:
            state = np.roll(state, -obs_dim)
            state[-obs_dim:]=obs
        else:
            state = obs
    return episode_reward,actiona,episode_length
#     
if __name__=='__main__':
    model = ANN(D,M,K)
    if len(sys.argv) > 1 and sys.argv[1] =='play':
        #play with a saved model
        j = np.load('es_qkprotocol_results0000.npz')
        best_params = np.concatenate([j['W1'].flatten(), j['b1'], j['W2'].flatten(), j['b2']])
        
        # in case intial shapes are not correct
        D, M =j['W1'].shape
        K = len(j['b2'])
        model.D, model.M, model.K = D, M, K
        #model.D,model.K=D,K
        x=np.arange(0,len(j['train']))
        plt.figure(figsize=(13, 13))
        plt.plot(x, j['train'], label='Rewards')
        plt.grid(True,which="both",ls="--",c='gray') 
        plt.title(f"Best reward={j['train'].max()}")
        plt.legend()
        plt.xlabel(f'Number of Steps of episode')
        plt.ylabel('Rewards')
        plt.savefig("Rewards-evolutionstrategy.png")
        plt.show()
        x=np.arange(0,len(j['learning_rate_v']))
        plt.figure(figsize=(13, 13))
        plt.plot(x, j['learning_rate_v'], label='learning_rate')
        plt.plot(x, j['sigmav'], label='sigma')
        plt.grid(True,which="both",ls="--",c='gray') 
        plt.title(f"Population size={j['populat_s']}")
        plt.legend()
        plt.xlabel(f'Number of Steps of episode')
        plt.ylabel('Learing rate')
        plt.savefig("learning_rate-evolutionstrategy.png")
        plt.show()
        x=np.arange(0,len(j['learning_rate_v']))
        plt.figure(figsize=(13, 13))
        plt.plot(x, j['pmeters'], label='weights')
        plt.grid(True,which="both",ls="--",c='gray') 
        plt.title(f"Weight Average={j['pmeters'].mean()}")
        plt.legend()
        plt.xlabel(f'Number of Steps of episode')
        plt.ylabel('Weights')
        plt.savefig("NN-evolutionstrategy.png")
        plt.show()
    else:
        # train and save
        model.init()
        params = model.get_params()
        best_params, rewards, actions, learn_r,sigmv,pop_s,parms = evolution_strategy(
            f=reward_function,
            population_size=500,
            sigma=0.002,
            lr=0.006,
            initial_params=params,
            num_iters=400,
        )
            
        model.set_params(best_params)
        np.savez('es_qkprotocol_results111100.npz',
                 learning_rate_v=learn_r,
                 sigmav=sigmv,
                 populat_s=pop_s,
                 pmeters=parms,
                 actions_e=actions,
                 train=rewards,
                 **model.get_params_dict(),
        )
        #play 5 test episodes
        #env.set_display(True)
        #state_n,actions,data,al_coun,al_data,bob_count,bob_mail,bob_mailbox,bob_k,done,act_hist,cum_re,state_space,action_space,max_moves,al_obs,bob_data,=reset()
        #env.reset()
        #for t in range(0,len(actions)):
        #    render()
        #    stat,re,do,action_h,bob_key=step(actions[t],act_hist,max_moves,al_coun,data,al_data,bob_count,al_obs,bob_k,bob_data,cum_re,bob_mailbox,bob_mail,done, verbose=0,)
        total_episodes0=[]
        solved=0
        episodes=100
        Rewa=0
        cum_re=[]
        cum_re1=[]
        total_ep=[]
        steps_ep=[]
        cumre=0
        for _ in range(episodes):
            Rew0, ac0,steps0=reward_function(best_params)
            total_episodes0.append(Rew0)
            Rewa += total_episodes0[-1]
            steps_ep.append(steps0)
            if Rew0>0 :
                solved+=1
                cumre+=1
                total_ep.append(1)
                cum_re1.append(cumre)
            else:
                total_ep.append(0)
            print("Episode {} Reward per episode {}".format(_,solved))
    plt.figure(figsize=(13, 13))
    plt.plot(cum_re1)
    plt.xlabel(f'Number of Steps of episode')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    x=np.arange(0,len(steps_ep))
    steps=np.repeat(min(steps_ep),len(x))
    #steps=np.arange(0,len(steps_ep),min(steps_ep))
    plt.plot(x,steps)
    plt.title('Number of steps per episode {}'.format(np.mean(steps)))
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Number of steps')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    #x=np.arange(0,len(j['learning_rate_v']))
    plt.figure(figsize=(13, 13))
    plt.plot(total_ep)
    plt.title('The simulation has been solved the environment Evolutionary Strategy:{}'.format(solved/episodes))
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    print('Average steps per episode {}'.format(np.mean(steps_ep)))
