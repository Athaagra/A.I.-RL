import numpy as np
import matplotlib.pyplot as plt
from grid_world import standard_grid

SMALL_ENOUGH = 10e-4

def print_values(V, g):
	for i in range(g.rows):
		print("-------")
		for j in range(g.cols):
			v = V.get((i,j),0)
			if v >=0 :
				print(v)
			else:
				print(v)
def print_policy(P, g):
	for i in range(g.rows):
		print("---------------")
		for j in range(g.cols):
			a = P.get((i,j), '')
			print(a)
	
#if __name__ == '__main__':
grid = standard_grid()
states = grid.all_states()
V={}
for s in states:
 V[s] = 0
 print(s)
gamma = 1.0 #discount factor
#while True:
biggest_change = 0
for s in states:
 old_v=V[s]
 print("Old_value {}".format(old_v))			
			#V(s) only has value if it is not a terminal state
 if s in grid.actions:			
   new_v = 0
   p_a = 1.0/len(grid.actions[s])
   print(grid.actions[s])
   print('probability {}'.format(p_a))
   for a in grid.actions[s]:
     print(a)
     grid.set_state(s)
     r = grid.move(a)
     print(r)
     new_v += p_a * (r + gamma * V[grid.current_state()])
     print(new_v)
     V[s] = new_v
     biggest_change = max(biggest_change, np.abs(old_v - V[s]))
			
     if biggest_change < SMALL_ENOUGH:
         break
     print("values for uniformly random actions:")
	#	print_values(V, grid)	
     policy = {
        (2, 0): 'U',
		(1, 0): 'U',
		(0, 0): 'R',
		(0, 1): 'R',
		(0, 2): 'R',
		(1, 2): 'R',
		(2, 1): 'R',
		(2, 2): 'R',
		(2, 3): 'U',
        }
	#	print_policy(policy, grid)
	#initialize V(s) = 0
V={}
for s in states:
    V[s] = 0
#let's see how V(s) changes as we get further away from the reward
gamma = 0.9 #discount factor	
#repeat until convergence
while True:
 biggest_change = 0
 for s in states:
  old_v = V[s]
  print(old_v)		
  if s in policy:
   print('print state:{}'.format(s))
   a = policy[s]
   print('print policy:{}'.format(a))
   print(grid.set_state(s))
   r = grid.move(a)
   print('print reward:{}'.format(r))
   print('current_state:{}'.format(grid.current_state()))
   V[s] = r + gamma * V[grid.current_state()]
   print('print state policy:{}'.format(V[s]))
   biggest_change = max(biggest_change, np.abs(old_v - V[s]))
   print('print biggest change:'.format(biggest_change))		
 if biggest_change > SMALL_ENOUGH:
				break
print("values for fixed policy:")
print_values(V,grid)