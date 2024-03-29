# -*- coding: utf-8 -*-
"""
Created on Sat Jan 23 22:37:29 2021

@author: athaa
"""


import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid, negative_grid
from iterative_policy_evaluation import print_values, print_policy

SMALL_ENOUGH = 10e-4
GAMMA = 0.9
ALL_POSSIBLE_ACTIONS = ('U', 'D', 'L', 'R')

#this is deterministic
#all p(s', r|s,a)= 1 or 0

if __name__ == '__main__':
  # this grid gives you a reward of -0.1 for every non-terminal state
  # we want to see if this will encourage finding a shorter path to the goal
  grid = negative_grid()
  
  #print rewards
  print("rewards:")
  print_values(grid.rewards, grid)
  
  # state -> action
  # we'll randomly choose an action and update as we learn
  policy={}
  for s in states: #grid.actions.key():
      policy[s] = np.random.choice(ALL_POSSIBLE_ACTIONS)
  
  #initial policy
  print("initial policy:")
  print_policy(policy, grid)
  
  # initialize V(s)
  V={}
  states = grid.all_states()
  for s in states:
      if s in range(0, 8): #grid.actions:
          V[s] = np.random.random()
      else:
          V[s] = 0
  #repeat until convergence - will break out when policy does not change
  while True:

	#policy evaluation step - we already know how to do this!
    while True:  
        biggest_change = 0
        for s in states:
            print(s)
            old_v=V[s]
        #V(s) only has value if it is not a terminal state
            if s in policy:
                a=policy[s]
                print(a)
                grid.set_state(s)
                r = grid.move(a)
                print(r)
                V[s] = r + GAMMA * V[grid.current_state()]
                biggest_change = max(biggest_change, np.abs(old_v - V[s]))
            if biggest_change < SMALL_ENOUGH:
                break
  #policy improvement step
  is_policy_converged = True
  for s in states:
      if s in policy:
        old_a = policy[s]
        new_a = None
        best_value = float('-inf')
		#loop through all possible actions to find the best current action
        for a in ALL_POSSIBLE_ACTIONS:
            grid.set_state(s)
            r = grid.move(a)
            v = r + GAMMA * V[grid.current_state()]
            if v > best_value:
                best_value = v
                new_a = a
        policy[s] = new_a
        if new_a != old_a:
            is_policy_converged = False
  if is_policy_converged:
      break
print("values:")
print_values(V, grid)
print("policy:")
print_policy(policy, grid)