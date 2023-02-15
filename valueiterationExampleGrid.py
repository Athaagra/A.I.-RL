#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 14:10:23 2023

@author: Optimus
"""
import numpy as np
from Environment import StochasticWindyGridworld
import matplotlib.pyplot as plt
#gamma = 0.001
threshold = 0.1
delta=0
env = StochasticWindyGridworld(initialize_model=True)
env.render()
grid = env
states = grid.n_states 
actions = grid.n_actions
Qsa=np.zeros((states,actions))
gamma = 1.0
def policyIteration():
    V={}
    pi={}
    for s in range(0,states):
        V[s] = [0,0,0,0]
        pi[s]=[0]
    done=False
    s=env.reset()
    while done!=True:#for s in range (0,states):
        a=np.random.randint(0,4,1)
        s_next,r,done=env.step(a[0])
        pi[s][0]=a[0]
        tmp=np.random.random()
        V[s][3]=tmp
        s=s_next
    return pi,V

pi,V=policyIteration()
#Value iteration
while delta < 1.4:#threshold * 1- gamma /gamma:
    biggest_change = 0
    delta=0
    s=env.reset()
    done=False
    while done!=True:#for s in range (0,states):
        a=pi[s]
        s_next,r,done=env.step(a[0])
        V[s][0]=pi[s]
        psr=len(a)/actions
        tmp=V[s][3]
        V[s][1]=r
        V[s][2]=s_next
        V[s][3]=psr*(r+gamma*V[s_next][3])
        delta=max(delta,tmp-V[s][3])
        s=s_next
r=[V[s][2] for s in range(0,states)]
valiter=[V[s][3] for s in range(0,states)]
print('This is the average reward {}'.format(np.average(r)))
print('This is the average value Function {}'.format(np.average(valiter)))
plt.figure(figsize=(13, 13))
plt.plot(valiter,r)
plt.xlabel(f'Number of episode')
plt.ylabel('Rewards')
plt.grid(True,which="both",ls="--",c='gray')
plt.title('This is the average reward {}'.format(np.average(r)))
plt.savefig('rewardsql.png')
plt.show()



plt.plot(valiter)
plt.show
#while True:
#while delta < threshold * 1- gamma /gamma:
#    biggest_change = 0
#    delta=0
#    done=False
#    s=env.reset()
#    while done!=True:#for s in range(0,states):
#     tmp=V[s][3]
#     a=pi[s]#np.random.randint(0,4,1)
#     psr=len(a)/actions
     #print(psr)
     #pi=1
#     s_next, r, done=env.step(a[0])
#     print(r)
#     V[s][0]=a[0]    
#     s=s_next
#for s in range(0,states):
#    pi[s]=np.argmax(np.sum(V[s][3]*pi[s],0))
