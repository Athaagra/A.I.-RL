#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 18:37:37 2023

@author: Optimus
"""

import os 
import time 
import random
import numpy as np
import matplotlib.pyplot as plt
import pybullet_envs
import gym
import torch 
import torch.nn as nn
import torch.nn.functional as F 
from gym import wrappers 
from torch.autograd import Variable 
from collections import deque

class ReplayBuffer(object):
    def __init__(self, max_size=1e6):
        self.storage=[]
        self.max_size=max_size
        self.ptr = 0
        
    def add(self, transition):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)]=transition
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(transition)
    def sample(self,batch_size):
        ind=np.random.randint(0, len(self.storage),size=batch_size)
        batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = [], [], [], [], []
        for i in ind:
            state,next_state,action,reward,done=self.storage[i]
            batch_states.append(np.array(state, copy=False))
            batch_next_states.append(np.array(next_state, copy=False))
            batch_actions.append(np.array(action, copy=False))
            batch_rewards.append(np.array(reward, copy=False))
            batch_dones.append(np.array(done, copy=False))
        return np.array(batch_states),np.array(batch_next_states), np.array(batch_actions), np.array(batch_rewards).reshape(-1,1), np.array(batch_dones).reshape(-1,1)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1=nn.Linear(state_dim,400)
        self.layer_2=nn.Linear(400,300)
        self.layer_3=nn.Linear(300, action_dim)
        self.max_action=max_action
        
    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.max_action * torch.tanh(self.layer_3(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Defining the first Critic neural network
        self.layer_1 = nn.Linear(state_dim + action_dim, 400)
        self.layer_2 = nn.Linear(400,300)
        self.layer_3 = nn.Linear(300,1)
        # Defining the second Critic neural network
        self.layer_4 = nn.Linear(state_dim + action_dim, 400)
        self.layer_5 = nn.Linear(400,300)
        self.layer_6 = nn.Linear(300,1)
        
    def Ql(self, x, u):
        xu = torch([x, u], 1)
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x))
        x1 = self.layer_3(x1)
        return x1

# Selecting the device(CPU or GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Building the whole Training Process into a class

class TDJ(object):
    def __init__(self,state_dim,action_dim,max_action):
        self.actor=Actor(state_dim,action_dim,max_action).to(device)
        self.actor_target=Actor(state_dim,action_dim,max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer=torch.optim.Adam(self.actor.parameters())
        self.critic = Critic(state_dim,action_dim).to(device)
        self.critic_target = Critic(state_dim,action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer=torch.optim.Adam(self.critic.parameters())
        self.max_action=max_action
    
    def select_action(self,state):
        state=torch.Tensor(state.reshape(1,-1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def train(self,replay_buffer,iterations,batch_size=100,discount=0.99, tau=0.005, policy_noise=0.2,noise_clip=0.5, policy_freq=2):
        
        for b in range(iterations):
            #sample a batch of transition (s,s',a,r) from the memory
            batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
            state = torch.Tensor(batch_states).to(device)
            next_state= torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done=torch.Tensor(batch_dones).to(device)
            
            next_action=self.actor_target(next_state)
            noise = torch.Tensor(batch_actions).data.normal(0,policy_noise).to(device)
            moise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
            target_Q1,target_Q2=self.critic_target(next_state,next_action)
            target_Q=torch.min(target_Q1,target_Q2)
            target_Q=reward + ((1-done) * discount * target_Q).detach()
            current_Q1,current_Q2 = self.critic(state,action)
            critic_loss = F.mse_loss(current_Q1,target_Q) + F.mse_loss(current_Q2,target_Q)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            if b % policy_freq == 0:
                actor_loss = - self.critic.Ql(state, self.actor(state)).mean()
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory,filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory,filename))
    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))
        self.critic.load_state_dict(torch.load('%s/%s_actor.pth' % (directory, filename)))

def evaluate_policy(policy,eval_episodes=10):
    avg_reward=0
    for _ in range(eval_episodes):
        obs=env.reset()
        done=False
        while not done:
            action = policy.select_action(np.array(obs))
            obs,reward,done, _ =env.step(action)
            avg_reward += reward
    avg_reward /= eval_episodes
    print("Average Reward over the Evaluation Step: %f" % (avg_reward))
    return avg_reward
#Parameters Initialization
env_name="HalfCheetahBulletEnv-v0"
seed = 0
start_timesteps=1e4
eval_freq=5e3
max_timesteps=1e6
save_models=True
expl_noise=0.1
batch_size=100
discount=0.99
tau=0.005
policy_noise=0.2
noise_clip=0.5
policy_freq=2

file_name="%s_%s_%s" % ('TD3', env_name, str(seed))

if not os.path.exist("./results"):
    os.makedirs("./results")
if save_models and  not os.path.exists("./pytorch_models"):
    os.makedirs("./pytorach_models")

env = gym.make(env_name)    
env.seeed(seed)
torch.manual_seed(seed)
state_dim = env.observation_space.shape[0]
action_dim=env.action_space.shape[0]
max_action=float(env.action_space.high[0])
policy = TD3(state_dim,action_dim,max_action)
replay_bufer=ReplayBuffer()

evaluations=[evaluate_policy(policy)]

while total_timesteps < max_timesteps:
    if done:
        if total_timesteps !=0:
            print("Total Timesteps:{} Episode Num {} Reward {}".format(total_timesteps,episode_num,episode_reward))
            policy.train(replay_bufer, episode_timesteps, batch_size, discount, tau, policy_noise, noise_clip, policy_freq)
        
        if timesteps_since_eval >=eval_freq:
            timesteps_since_eval %= eval_freq
            evaluations.append(evaluate_policy(policy))
            policy.save(file_name, directory="./pytorach_models")
            np.save("./results/%s" % (file_name), evaluations)
            
        obs=env.reset()
        done=False
        episode_reward=0
        episode_timesteps=0
        episode_num+=
    if total_timesteps < start_timesteps:
        action = env.action_space.sample()
    else:
        action = policy.select_action(np.array(obs))
        if expl_noise !=0:
            action = (action + np.random.normal(0, expl_noise, size=env.action_space.shape[0])).clip(env.action_space.low, env.action_space.high)
    
    new_obs,reward,done,_=env.step(action)
    done_bool=0 if episode_timesteps + 1 == env._max_episode_steps else float(done)
    episode_reward += reward
    replay_buffer.add((obs,new_obs,action, reward, done_bool))
    obs=new_obs
    episode_timesteps +=1
    total_timesteps+=1
    timesteps_since_eval+=1
evaluations.append(evaluate_policy(policy))
if save_models:policy.save('%s' % (file_name),directory="./pytorch_models")
np.save("./results/%s" % (file_name),evaluations)