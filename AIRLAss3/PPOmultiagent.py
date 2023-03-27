#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 17:15:59 2023

@author: Optimus
"""

"""
Created on Fri Nov 18 00:55:15 2022
@author: Optimus
"""

import numpy as np
from itertools import count
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import scipy.signal 
import sys

#temp = sys.stdout                 # store original stdout object for later
#sys.stdout = open('log.txt', 'w')
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 20:46:53 2023

@author: Optimus
"""
"""
The environment for Level1
# Actions for Alice:
# 0 - Idle
# 1 - Read next bit from data1, store in datalog
# 2 - Place datalog in Bob's mailbox
# Actions for Bob:
# 0 - Idle
# 1 - Read next bit from mailbox
# 2 - Write 0 to key
# 3 - Write 1 to key
# Actions are input to the environment as tuples
# e.g. (1,0) means Alice takes action 1 and Bob takes action 0
# Rewards accumulate: negative points for wrong guess, positive points for correct guess
# Game terminates with correct key or N moves
# """
import numpy as np



def discounted_cumulative_sums(x, discount):
    # Discounted cumulative sums of vectors for computing rewards-to-go and advantage estimates
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class Buffer:
    # Buffer for storing trajectories
    def __init__(self, observation_dimensions, size, gamma=0.99, lam=0.95):
        # Buffer initialization
        self.observation_buffer = np.zeros(
            (size, observation_dimensions), dtype=np.float32
        )
        self.action_buffer = np.zeros(size, dtype=np.int32)
        self.advantage_buffer = np.zeros(size, dtype=np.float32)
        self.reward_buffer = np.zeros(size, dtype=np.float32)
        self.return_buffer = np.zeros(size, dtype=np.float32)
        self.value_buffer = np.zeros(size, dtype=np.float32)
        self.logprobability_buffer = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.pointer, self.trajectory_start_index = 0, 0
    
    def store(self, observation, action, reward, value, logprobability):
        # Append one step of agent-environment interaction
        self.observation_buffer[self.pointer] = observation
        self.action_buffer[self.pointer] = action
        self.reward_buffer[self.pointer] = reward
        self.value_buffer[self.pointer] = value
        self.logprobability_buffer[self.pointer] = logprobability
        self.pointer += 1
    
    def finish_trajectory(self, last_value=0):
        # Finish the trajectory by computing advantage estimates and rewards-to-go
        path_slice = slice(self.trajectory_start_index, self.pointer)
        rewards = np.append(self.reward_buffer[path_slice], last_value)
        values = np.append(self.value_buffer[path_slice], last_value)
        
        deltas = rewards[:-1] + self.gamma * values[1:] - values[:-1]
        
        self.advantage_buffer[path_slice] = discounted_cumulative_sums(
            deltas, self.gamma * self.lam
        )
        self.return_buffer[path_slice] = discounted_cumulative_sums(
            rewards, self.gamma
        )[:-1]
        
        self.trajectory_start_index = self.pointer
    
    def get(self):
        # Get all data of the buffer and normalize the advantages
        self.pointer, self.trajectory_start_index = 0, 0
        advantage_mean, advantage_std = (
            np.mean(self.advantage_buffer),
            np.std(self.advantage_buffer),
        )
        self.advantage_buffer = (self.advantage_buffer - advantage_mean) / advantage_std
        return (
            self.observation_buffer,
            self.action_buffer,
            self.advantage_buffer,
            self.return_buffer,
            self.logprobability_buffer,
        )


def mlp(x, sizes, activation=tf.tanh, output_activation=None):
    # Build a feedforward neural network
    for size in sizes[:-1]:
        print(size)
        x = layers.Dense(units=size, activation=activation)(x)
    return layers.Dense(units=sizes[-1], activation=output_activation)(x)


def logprobabilities(logits, a):
    # Compute the log-probabilities of taking actions a by using the logits (i.e. the output of the actor)
    logprobabilities_all = tf.nn.log_softmax(logits)
    #print('This is the logprobabilities all {}'.format(logprobabilities_all))
    logprobability = tf.reduce_sum(
        tf.one_hot(a, num_actions) * logprobabilities_all, axis=1
    )
    return logprobability

@tf.function
def sample_action(observation,md):
    logits = md(observation)
    action = tf.squeeze(tf.random.categorical(logits, 1), axis=1)
    return logits, action

# Train the policy by maxizing the PPO-Clip objective
@tf.function
def train_policy(
    observation_buffer, action_buffer, logprobability_buffer, advantage_buffer,md
):
    
    #print('This is the train policy')
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        ratio = tf.exp(
            #(observation_buffer)
            logprobabilities(md(observation_buffer), action_buffer)
            - logprobability_buffer
        )
        #print('This is the ratio {} advantage buffer'.format(ratio))
        min_advantage = tf.where(
            advantage_buffer > 0,
            (1 + clip_ratio) * advantage_buffer,
            (1 - clip_ratio) * advantage_buffer,
        )
        #print('This is the min_advantage {} 1+ clip_ratio*advantage_buffer, 1-clip_ratio*advantage_buffer '.format(min_advantage))
        policy_loss = -tf.reduce_mean(tf.minimum(ratio * advantage_buffer, min_advantage))
        #ld=np.array(policy_loss)
        #fp = open('NN_for_Dymola.txt', 'w' ,encoding='UTF-8')
        #tf.print(policy_loss, output_stream=sys.stdout)
        #tf.io.write_file('Training.txt',policy_loss)
    policy_grads = tape.gradient(policy_loss, md.trainable_variables)
    #print('Policy grads {}'.format(policy_grads))
    policy_optimizer.apply_gradients(zip(policy_grads, md.trainable_variables))
    #print('policy optimizer {}'.format(policy_optimizer))
    kl = tf.reduce_mean(
        logprobability_buffer
        - logprobabilities(md(observation_buffer), action_buffer)
    )
    #print('This is the kl {} logprobability_buffer - logprobabilities(actor(observation_buffer), action_buffer)'.format(kl))
    kl = tf.reduce_sum(kl)
    return kl#,policy_loss


# Train the value function by regression on mean-squared error
@tf.function
def train_value_function(observation_buffer, return_buffer,cc):
    #print('This is the observation_buffer {} and the return buffer {}'.format(observation_buffer, return_buffer))
    with tf.GradientTape() as tape:  # Record operations for automatic differentiation.
        #print('This is the tape {}'.format(tape))
        value_loss = tf.reduce_mean((return_buffer - cc(observation_buffer)) ** 2)
        #ldd=value_loss
        #tf.print(value_loss, output_stream=sys.stdout)#print('This is the value loss {}'.format(ldd))
    value_grads = tape.gradient(value_loss, cc.trainable_variables)
    #print('This is the value_grads {} value_loss, critic.trainbable_variables'.format(value_grads))
    value_optimizer.apply_gradients(zip(value_grads, cc.trainable_variables))
    #return value_loss

# Hyperparameters of the PPO algorithm
gamma = 0.001
clip_ratio = 0.01
policy_learning_rate = 1e-11
value_function_learning_rate = 1e-11
train_policy_iterations = 200
train_value_iterations = 200
lam = 0.97
target_kl = 0.01
hidden_sizes = (16, 16)

# True if you want to render the environment
render = False

# Initialize the environment and get the dimensionality of the
# Initialize the observation, episode return and episode length
#observation, episode_return, episode_length = env.reset(), 0, 0
# Iterate over the number of epochs
LogicalStates=np.array([[1,0],[0,1]])
LogicalStates2bit=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
LogicalStates3bit=np.array([[1,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0],[0,0,1,0,0,0,0,0],[0,0,0,1,0,0,0,0],[0,0,0,0,1,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,1]])
LogicalStates4bit=np.array([[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]])
import pandas as pd
columns2bit=['00','01','10','11']
columns3bit=['000','001','010','011','100','101','110','111']
columns4bit=['0000','0001','0010','0011','0100','0101','0110','0111','1000','1001','1010','1011','1100','1101','1110','1111']
LogicalStates2bit=pd.DataFrame(LogicalStates2bit, columns=columns2bit)
LogicalStates3bit=pd.DataFrame(LogicalStates3bit, columns=columns3bit)
LogicalStates4bit=pd.DataFrame(LogicalStates4bit, columns=columns4bit)
LogicalStates2bit=LogicalStates2bit.rename(index={0:'00',1:'01',2:'10',3:'11'})
LogicalStates3bit=LogicalStates3bit.rename(index={0:'000',1:'001',2:'010',3:'011',4:'100',5:'101',6:'110',7:'111'})
LogicalStates4bit=LogicalStates4bit.rename(index={0:'0000',1:'0001',2:'0010',3:'0011',4:'0100',5:'0101',6:'0110',7:'0111',8:'1000',9:'1001',10:'1010',11:'1011',12:'1100',13:'1101',14:'1110',15:'1111'})
def mannwhitney(total_episodes,error):
    from scipy.stats import mannwhitneyu
    # seed the random number generator
    resultss=[]
    if sum(total_episodes)!=sum(error):
        stat, pvalue = mannwhitneyu(total_episodes, error)
        print('Statistics=%.3f, p=%.3f' % (stat, pvalue))
        # interpret
        if pvalue > 0.05:
            print('We accept the null hypothesis')
            resultss.append(['Qlearning p-value We accept the null hypothesis:',pvalue])
        else:
            print("The p-value is less than we reject the null hypothesis")
            resultss.append(['Qlearning p-value the p-value is less than we reject the null hypothesis:',pvalue])
    else:
        print('identical')
        pvalue=0
    import matplotlib.pyplot as plt
    plt.figure(figsize=(13, 13))
    plt.bar(1,pvalue)
    plt.xlabel(f'Mannwhitney Test')
    plt.ylabel('Probability')
    plt.title(str(resultss))#.format(solved/EPISODES))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    return resultss
def proximalpo(inpu,ac,cr,ma,qp):
    epochs = 100
    steps_ep=[]
    q_value_critic=[]
    action_actor=[]
    cum=[]
    total_episodes=[]
    cumre=0
    total_fidelity=[]
    for epoch in range(epochs):
        # Initialize the sum of the returns, lengths and number of episodes for each epoch
        sum_return = 0
        sum_length = 0
        num_episodes = 0
        done=False
        env=Qprotocol(4,inpu,MultiAgent=ma,Qb=qp)
        state,inp=env.reset(4)
        actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
        observation = state[0]
        episode_return=0
        episode_length=0
        while done != True:
        # Iterate over the steps of each epoch
            if render:
                print(render)
            observation = observation.reshape(1, -1)
            logits, actiona = sample_action(observation,ac)
            actiona=np.array(actiona)
            log=np.array(logits[0])
            actiona=actiona[0]
            action_actor.append(log[actiona])
            action=np.array(actions_list[actiona])
            new_state,reward,done,info,bob_key=env.step(action)
            episode_return += reward
            episode_length += 1        
            # Get the value and log-probability of the action
            value_t = cr(observation)
            q_value_critic.append(value_t)
            logprobability_t = logprobabilities(logits, actiona)
            # Store obs, act, rew, v_t, logp_pi_t
            buffer.store(observation, actiona, reward, value_t, logprobability_t)
            # Update the observation
            observation = np.array(new_state[0])
            # Finish trajectory if reached to a terminal state
            terminal = done
            if terminal: #or (t == steps_per_epoch - 1):
                last_value = 0 if done else cr(observation.reshape(1, -1))
                buffer.finish_trajectory(last_value)
                sum_return += episode_return
                sum_length += episode_length
                steps_ep.append(episode_length)
                num_episodes += 1
                cumre+=reward
                cum.append(cumre)
                state,inp=env.reset(4)
                observation=np.array(state[0]) 
                episode_return=0
                episode_length = 0
                if reward==1:
                    total_episodes.append(1)
                else:
                    total_episodes.append(0)
                if len(inp)==len(bob_key):
                    if len(inp)==1 and len(bob_key)==len(inp):
                        tp=LogicalStates[:,inp].T*LogicalStates[bob_key,:]
                        tp=tp[0]
                        Fidelity=abs(sum(tp))**2
                        total_fidelity.append(Fidelity)
                    if len(inp)==2 and len(bob_key)==len(inp):
                        inpus=''.join(str(x) for x in inp)
                        bob_keys=''.join(str(x) for x in bob_key[:len(inp)])
                        tp=np.array(LogicalStates2bit.loc[:,inpus]).T*np.array(LogicalStates2bit.loc[bob_keys,:])
                        Fidelity=abs(sum(tp))**2
                        total_fidelity.append(Fidelity)
                    if len(inp)==3 and len(bob_key)==len(inp):
                        inpus=''.join(str(x) for x in inp)
                        bob_keys=''.join(str(x) for x in bob_key[:len(inp)])
                        tp=np.array(LogicalStates3bit.loc[:,inpus]).T*np.array(LogicalStates3bit.loc[bob_keys,:])
                        Fidelity=abs(sum(tp))**2
                        total_fidelity.append(Fidelity)
                    if len(inp)==4 and len(bob_key)==len(inp):
                        inpus=''.join(str(x) for x in inp)
                        bob_keys=''.join(str(x) for x in bob_key[:len(inp)])
                        tp=np.array(LogicalStates4bit.loc[:,inpus]).T*np.array(LogicalStates4bit.loc[bob_keys,:])
                        Fidelity=abs(sum(tp))**2
                        total_fidelity.append(Fidelity)
                else:
                    total_fidelity.append(0)
        # Get values from the buffer
        (
            observation_buffer,
            action_buffer,
            advantage_buffer,
            return_buffer,
            logprobability_buffer,
        ) = buffer.get()    
        # Update the policy and implement early stopping using KL divergence
        locy=[]
        for _ in range(train_policy_iterations):
            kl = train_policy(
                observation_buffer, action_buffer, logprobability_buffer, advantage_buffer, ac
            )
            #locy.append(lc)
            if kl > 1.5 * target_kl:
                # Early Stopping
                break
    #    print('This is the kl {}'.format(kl))
        # Update the value function
        cry=[]
        for _ in range(train_value_iterations):
            train_value_function(observation_buffer, return_buffer,cr)

        rewards_during_training.append(sum_return / num_episodes)
        # Print mean return and length for each epoch
        print(
            f" Epoch: {epoch + 1}. Mean Return: {sum_return / num_episodes}. Mean Length: {sum_length / num_episodes}"
        )
    def save_weights(ac,cr,inpt,maxm):
        path= '/home/Optimus/Desktop/QuantumComputingThesis/'
        ac.save(path+ '_actor'+str(maxm)+'One'+str(inpt)+'.h5')
        cr.save(path+ '_critic'+str(maxm)+'One'+str(inpt)+'.h5')
    plt.figure(figsize=(13, 13))
    plt.plot(total_episodes)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.title('The simulation has been solved the environment '+str(inpu)+' Proximal Policy:{}'.format(sum(total_episodes)))#.format(solved/episodes))
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(cum)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.title('The simulation has been solved the environment '+str(inpu)+' Proximal Policy Cumulative:{}'.format(max(cum)))#.format(solved/episodes))
    plt.savefig('cumulativeppo.png')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(total_fidelity)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Fidelity')
    plt.title('The simulation has been solved the environment '+str(inpu)+' Proximal Policy Evaluation Fidelity:{}'.format(sum(total_fidelity)))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(q_value_critic)
    plt.plot(action_actor)
    plt.xlabel(f'loss of episode')
    plt.ylabel('Q-value')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.title('The simulation '+str(inpu)+' and the Q-value of Proximal Policy Evaluation')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(steps_ep)
    plt.xlabel(f'Number of steps of each episode')
    plt.ylabel('Steps')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.title('The simulation and the number of steps '+str(inpu)+' of Proximal Policy Evaluation {}'.format(np.mean(steps_ep)))
    plt.show()
    save_weights(ac,cr,inpu,4)
    error=env.error_counter
    results=mannwhitney(rewards_during_training,error)
    results.append(['Reward:'+str(count),'Cumulative:'+str(max(cum)),'Steps:'+str(np.mean(steps_ep))])
    return actor,critic,results

def pposimulation(inpu,ac,ma,qp):
    total_episodes=[]
    solved=0
    episodes=100
    steps_epi=[]
    cum_rev=0
    cumulative_reward=[]
    r=0
    total_fidelity=[]
    count=0
    actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
    # run infinitely many episodes
    for i_episode in range(episodes):
        env=Qprotocol(4,inpu,MultiAgent=ma,Qb=qp)
        state,inp=env.reset(4)
        # reset environment and episode reward
        print('This is input {}'.format(inp))
        ep_reward = 0
        done=False
        observation = state[0]
        steps_ep=0
        while done!=True:
            print('This is the episode {}'.format(i_episode))
            observation = observation.reshape(1, -1)
            logits, actiona = sample_action(observation,ac)
            actiona=actiona[0]
            actiona=np.array(actions_list[actiona])
            stat,reward,done,action_h,bob_keya=env.step(actiona)
            observation=stat[0]
            steps_ep+=1
            if done==True:
                bob_key=bob_keya
                steps_epi.append(steps_ep)
                if len(inp)==len(bob_key):
                    if len(inp)==1 and len(bob_key)==len(inp):
                        tp=LogicalStates[:,inp].T*LogicalStates[bob_key,:]
                        tp=tp[0]
                        Fidelity=abs(sum(tp))**2
                        total_fidelity.append(Fidelity)
                    if len(inp)==2 and len(bob_key)==len(inp):
                        inpus=''.join(str(x) for x in inp)
                        bob_keys=''.join(str(x) for x in bob_key[:len(inp)])
                        tp=np.array(LogicalStates2bit.loc[:,inpus]).T*np.array(LogicalStates2bit.loc[bob_keys,:])
                        Fidelity=abs(sum(tp))**2
                        total_fidelity.append(Fidelity)
                    if len(inp)==3 and len(bob_key)==len(inp):
                        inpus=''.join(str(x) for x in inp)
                        bob_keys=''.join(str(x) for x in bob_key[:len(inp)])
                        tp=np.array(LogicalStates3bit.loc[:,inpus]).T*np.array(LogicalStates3bit.loc[bob_keys,:])
                        Fidelity=abs(sum(tp))**2
                        total_fidelity.append(Fidelity)
                    if len(inp)==4 and len(bob_key)==len(inp):
                        inpus=''.join(str(x) for x in inp)
                        bob_keys=''.join(str(x) for x in bob_key[:len(inp)])
                        tp=np.array(LogicalStates4bit.loc[:,inpus]).T*np.array(LogicalStates4bit.loc[bob_keys,:])
                        Fidelity=abs(sum(tp))**2
                        total_fidelity.append(Fidelity)
                else:
                    total_fidelity.append(0)
                if reward==1:
                    r=1
                    cum_rev+=r
                    cumulative_reward.append(cum_rev)
                    solved+=1
                    total_episodes.append(r)
                    break
                else:
                    r=-1
                    cum_rev+=r
                    solved+=0
                    cumulative_reward.append(cum_rev)
                    total_episodes.append(r)
                    break
    plt.figure(figsize=(13, 13))
    plt.plot(total_episodes)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.title('The simulation has been solved the environment '+str(inpu)+' Proximal Policy Evaluation Rewards:{}'.format(solved/episodes))
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(cumulative_reward)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Rewards')
    plt.title('The simulation has been solved the environment '+str(inpu)+' Proximal Policy Evaluation Cumulative:{}'.format((cumulative_reward[-1])))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(total_fidelity)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Fidelity')
    plt.title('The simulation has been solved the environment '+str(inpu)+' Proximal Policy Evaluation Fidelity:{}'.format(sum(total_fidelity)))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(steps_epi)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Steps')
    plt.title('The number of steps:{}'.format(np.average(steps_epi)))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    error=env.error_counter
    results=mannwhitney(rewards_during_training,error)
    results.append(['Reward:'+str(solved/episodes),'Cumulative:'+str(cumulative_reward[-1]),'Steps:'+str(np.mean(steps_epi)),'Fidelity:'+str(sum(total_fidelity))])
    return results

def onebitsimulation(inpa,ac,ac1,ma,qp):
    total_episodes=[]
    solved=0
    episodes=100
    steps_epi=[]
    cum_rev=0
    cumulative_reward=[]
    r=0
    total_fidelity=[]
    count=0
    actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
    # run infinitely many episodes
    for i_episode in range(episodes):
        # reset environment and episode reward
        inpu=np.random.randint(0,2,inpa)
        env=Qprotocol(4,inpu,MultiAgent=ma,Qb=qp)
        env1=Qprotocol(4,inpu,MultiAgent=ma,Qb=qp)
        state,inp=env.reset(4)
        state1,inp=env1.reset(4)
        ep_reward = 0
        done1=False
        done2=False
        observation1 = state[0]
        observation2 = state1[0]
        steps_ep=0
        while done1!=True or done2!=True:
            print('This is the episode {}'.format(i_episode))
            observation1 = observation1.reshape(1, -1)
            observation2 = observation2.reshape(1, -1)
            logits, actiona = sample_action(observation1,ac)
            logits, actionb = sample_action(observation2,ac1)
            actiona=actiona[0]
            actionb=actionb[0]
            actionAA=np.array(actions_list[actiona])
            actionBB=np.array(actions_list[actionb])
            stat1,reward1,done1,action_h1,bob_key1=env.step(actionAA)
            stat2,reward2,done2,action_h2,bob_key2=env1.step(actionBB)
            observation1=stat1[0]
            observation2=stat2[0]
            steps_ep+=1
            if done1:
                bob_key=bob_key1
            if done2:
                bob_key=bob_key2
            if done1==True or done2==True:
                steps_epi.append(steps_ep)
                if len(inp)==1 and len(bob_key)==len(inp):
                    tp=LogicalStates[:,inp].T*LogicalStates[bob_key,:]
                    tp=tp[0]
                    Fidelity=abs(sum(tp))**2
                    total_fidelity.append(Fidelity)
                else:
                    total_fidelity.append(0)
            if reward1>0 or reward2>0 :
                r=1
                cum_rev+=r
                cumulative_reward.append(cum_rev)
                solved+=1
                total_episodes.append(r)
            else:
                r=-1
                solved+=0
                cum_rev+=r
                cumulative_reward.append(cum_rev)
                total_episodes.append(r)
                break
    plt.figure(figsize=(13, 13))
    plt.plot(total_episodes)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.title('The simulation has been solved the environment '+str(len(inpu))+' Proximal Policy Evaluation Rewards:{}'.format(solved/episodes))
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(cumulative_reward)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Rewards')
    plt.title('The simulation has been solved the environment '+str(len(inpu))+' Proximal Policy Evaluation Cumulative:{}'.format(np.max(cumulative_reward)))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(total_fidelity)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Fidelity')
    plt.title('The simulation has been solved the environment '+str(len(inpu))+'Proximal Policy Evaluation Fidelity:{}'.format(sum(total_fidelity)))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(steps_epi)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Rewards')
    plt.title('The number of steps:{}'.format(np.average(steps_epi)))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    error=env.error_counter
    error1=env1.error_counter
    results=mannwhitney(total_episodes,error)
    results1=mannwhitney(total_episodes,error1)
    results.append([results1,'Reward:'+str(solved/episodes),'Cumulative:'+str(cumulative_reward[-1]),'Steps:'+str(np.mean(steps_epi)),'Fidelity:'+str(sum(total_fidelity))])
    return results

def twobitsimulation(inpa,ac,ac1,ac2,ac3,ma,qp):
    total_episodes=[]
    solved=0
    episodes=100
    steps_epi=[]
    cum_rev=0
    cumulative_reward=[]
    r=0
    total_fidelity=[]
    count=0
    actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
    # run infinitely many episodes
    for i_episode in range(episodes):
        # reset environment and episode reward
        inpu=np.random.randint(0,2,inpa)
        env=Qprotocol(4,inpu,MultiAgent=ma,Qb=qp)
        env1=Qprotocol(4,inpu,MultiAgent=ma,Qb=qp)
        env2=Qprotocol(4,inpu,MultiAgent=ma,Qb=qp)
        env3=Qprotocol(4,inpu,MultiAgent=ma,Qb=qp)
        state,inp=env.reset(4)
        state1,inp=env1.reset(4)
        state2,inp=env2.reset(4)
        state3,inp=env3.reset(4)
        ep_reward = 0
        done1=False
        done2=False
        done3=False
        done4=False
        observation1 = state[0]
        observation2 = state1[0]
        observation3 = state2[0]
        observation4 = state3[0]
        steps_ep=0
        while done1!=True or done2!=True or done3!=True or done4!=True:
            print('This is the episode {}'.format(i_episode))
            observation1 = observation1.reshape(1, -1)
            observation2 = observation2.reshape(1, -1)
            observation3 = observation3.reshape(1, -1)
            observation4 = observation4.reshape(1, -1)
            logits, actiona = sample_action(observation1,ac)
            logits, actionb = sample_action(observation2,ac1)
            logits, actionc = sample_action(observation3,ac2)
            logits, actiond = sample_action(observation4,ac3)
            actiona=actiona[0]
            actionb=actionb[0]
            actionc=actionc[0]
            actiond=actiond[0]
            actionAA=np.array(actions_list[actiona])
            actionBB=np.array(actions_list[actionb])
            actionCC=np.array(actions_list[actionc])
            actionDD=np.array(actions_list[actiond])
            stat1,reward1,done1,action_h1,bob_key1=env.step(actionAA)
            stat2,reward2,done2,action_h2,bob_key2=env1.step(actionBB)
            stat3,reward3,done3,action_h3,bob_key3=env2.step(actionCC)
            stat4,reward4,done4,action_h4,bob_key4=env3.step(actionDD)
            observation1=stat1[0]
            observation2=stat2[0]
            observation3=stat3[0]
            observation4=stat4[0]
            steps_ep+=1
            if done1:
                bob_key=bob_key1
            if done2:
                bob_key=bob_key2
            if done3:
                bob_key=bob_key3
            if done4:
                bob_key=bob_key4
            if done1==True or done2==True or done3==True or done4==True:
                steps_epi.append(steps_ep)
                if len(inp)==2 and len(bob_key)==len(inp):
                    inpus=''.join(str(x) for x in inp)
                    bob_keys=''.join(str(x) for x in bob_key[:len(inp)])
                    tp=np.array(LogicalStates2bit.loc[:,inpus]).T*np.array(LogicalStates2bit.loc[bob_keys,:])
                    Fidelity=abs(sum(tp))**2
                    total_fidelity.append(Fidelity)
                if reward1==1 or reward2==1 or reward3==1 or reward4==1:
                    count+=1
                    r=1
                    solved+=1
                    cum_rev+=r
                    cumulative_reward.append(cum_rev)
                    total_episodes.append(r)
                    break
                else:
                    r=-1
                    cum_rev+=r
                    cumulative_reward.append(cum_rev)
                    total_episodes.append(r)
                    break
    plt.figure(figsize=(13, 13))
    plt.plot(total_episodes)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.title('The simulation has been solved the environment '+str(len(inpu))+' Proximal Policy Evaluation Rewards:{}'.format(solved/episodes))
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(cumulative_reward)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Cumulative')
    plt.title('The simulation has been solved the environment '+str(len(inpu))+' Proximal Policy Evaluation Cumulative:{}'.format(np.max(cumulative_reward)))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(total_fidelity)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Fidelity')
    plt.title('The simulation has been solved the environment '+str(len(inpu))+' Proximal Policy Evaluation Fidelity:{}'.format(sum(total_fidelity)))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(steps_epi)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Rewards')
    plt.title('The number of steps:{}'.format(np.average(steps_epi)))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    error=env.error_counter
    error1=env1.error_counter
    error2=env2.error_counter
    error3=env3.error_counter
    results=mannwhitney(total_episodes,error)
    results1=mannwhitney(total_episodes,error1)
    results2=mannwhitney(total_episodes,error2)
    results3=mannwhitney(total_episodes,error3)
    results.append([results1,results2,results3,'Reward:'+str(solved/episodes),'Cumulative:'+str(cumulative_reward[-1]),'Steps:'+str(np.mean(steps_epi)),'Fidelity:'+str(sum(total_fidelity))])
    return results


def threebitsimulation(inpa,ac,ac1,ac2,ac3,ac4,ac5,ac6,ac7,ma,qp):
    total_episodes=[]
    solved=0
    episodes=100
    steps_epi=[]
    cum_rev=0
    cumulative_reward=[]
    r=0
    total_fidelity=[]
    count=0
    actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
    # run infinitely many episodes
    for i_episode in range(episodes):
        # reset environment and episode reward
        inpu=np.random.randint(0,2,inpa)
        env=Qprotocol(4,inpu,MultiAgent=ma,Qb=qp)
        env1=Qprotocol(4,inpu,MultiAgent=ma,Qb=qp)
        env2=Qprotocol(4,inpu,MultiAgent=ma,Qb=qp)
        env3=Qprotocol(4,inpu,MultiAgent=ma,Qb=qp)
        env4=Qprotocol(4,inpu,MultiAgent=ma,Qb=qp)
        env5=Qprotocol(4,inpu,MultiAgent=ma,Qb=qp)
        env6=Qprotocol(4,inpu,MultiAgent=ma,Qb=qp)
        env7=Qprotocol(4,inpu,MultiAgent=ma,Qb=qp)
        state,inp=env.reset(4)
        state1,inp=env1.reset(4)
        state2,inp=env2.reset(4)
        state3,inp=env3.reset(4)
        state4,inp=env4.reset(4)
        state5,inp=env5.reset(4)
        state6,inp=env6.reset(4)
        state7,inp=env7.reset(4)
        ep_reward = 0
        done1=False
        done2=False
        done3=False
        done4=False
        done5=False
        done6=False
        done7=False
        done8=False
        observation1 = state[0]
        observation2 = state1[0]
        observation3 = state2[0]
        observation4 = state3[0]
        observation5 = state4[0]
        observation6 = state5[0]
        observation7 = state6[0]
        observation8 = state7[0]
        steps_ep=0
        while done1!=True or done2!=True or done3!=True or done4!=True or done5!=True or done6!=True or done7!=True or done8!=True:
            print('This is the episode {}'.format(i_episode))
            observation1 = observation1.reshape(1, -1)
            observation2 = observation2.reshape(1, -1)
            observation3 = observation3.reshape(1, -1)
            observation4 = observation4.reshape(1, -1)
            observation5 = observation5.reshape(1, -1)
            observation6 = observation6.reshape(1, -1)
            observation7 = observation7.reshape(1, -1)
            observation8 = observation8.reshape(1, -1)
            logits, actiona = sample_action(observation1,ac)
            logits, actionb = sample_action(observation2,ac1)
            logits, actionc = sample_action(observation3,ac2)
            logits, actiond = sample_action(observation4,ac3)
            logits, actione = sample_action(observation5,ac4)
            logits, actionf = sample_action(observation6,ac5)
            logits, actiong = sample_action(observation7,ac6)
            logits, actionh = sample_action(observation8,ac7)
            actiona=actiona[0]
            actionb=actionb[0]
            actionc=actionc[0]
            actiond=actiond[0]
            actione=actione[0]
            actionf=actionf[0]
            actiong=actiong[0]
            actionh=actionh[0]
            actionAA=np.array(actions_list[actiona])
            actionBB=np.array(actions_list[actionb])
            actionCC=np.array(actions_list[actionc])
            actionDD=np.array(actions_list[actiond])
            actionEE=np.array(actions_list[actione])
            actionFF=np.array(actions_list[actionf])
            actionGG=np.array(actions_list[actiong])
            actionHH=np.array(actions_list[actionh])
            stat1,reward1,done1,action_h1,bob_key1=env.step(actionAA)
            stat2,reward2,done2,action_h2,bob_key2=env1.step(actionBB)
            stat3,reward3,done3,action_h3,bob_key3=env2.step(actionCC)
            stat4,reward4,done4,action_h4,bob_key4=env3.step(actionDD)
            stat5,reward5,done5,action_h5,bob_key5=env4.step(actionEE)
            stat6,reward6,done6,action_h6,bob_key6=env5.step(actionFF)
            stat7,reward7,done7,action_h7,bob_key7=env6.step(actionGG)
            stat8,reward8,done8,action_h8,bob_key8=env7.step(actionHH)
            observation1=stat1[0]
            observation2=stat2[0]
            observation3=stat3[0]
            observation4=stat4[0]
            observation5=stat5[0]
            observation6=stat6[0]
            observation7=stat7[0]
            observation8=stat8[0]
            steps_ep+=1
            if done1:
                bob_key=bob_key1
            if done2:
                bob_key=bob_key2
            if done3:
                bob_key=bob_key3
            if done4:
                bob_key=bob_key4
            if done5:
                bob_key=bob_key5
            if done6:
                bob_key=bob_key6
            if done7:
                bob_key=bob_key7
            if done8:
                bob_key=bob_key8
            if done1==True or done2==True or done3==True or done4==True or done5==True or done6==True or done7==True or done8==True:
                steps_epi.append(steps_ep)
                if len(inp)==3 and len(bob_key)==len(inp):
                    inpus=''.join(str(x) for x in inp)
                    bob_keys=''.join(str(x) for x in bob_key[:len(inp)])
                    tp=np.array(LogicalStates3bit.loc[:,inpus]).T*np.array(LogicalStates3bit.loc[bob_keys,:])
                    Fidelity=abs(sum(tp))**2
                    total_fidelity.append(Fidelity)
                else:
                    total_fidelity.append(0)
                if reward1==1 or reward2==1 or reward3==1 or reward4==1 or reward5==1 or reward6==1 or reward7==1 or reward8==1:
                    count+=1
                    r=1
                    solved+=1
                    cumulative_reward.append(cum_rev)
                    total_episodes.append(r)
                    break
                else:
                    r=-1
                    cum_rev+=r
                    cumulative_reward.append(cum_rev)
                    total_episodes.append(r)
                    break
    plt.figure(figsize=(13, 13))
    plt.plot(total_episodes)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.title('The simulation has been solved the environment '+str(len(inpu))+' Proximal Policy Evaluation Rewards:{}'.format(solved/episodes))
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(cumulative_reward)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Rewards')
    plt.title('The simulation has been solved the environment '+str(len(inpu))+' Proximal Policy Evaluation Cumulative:{}'.format(np.max(cumulative_reward)))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(total_fidelity)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Rewards')
    plt.title('The simulation has been solved the environment '+str(len(inpu))+' Proximal Policy Evaluation fidelity:{}'.format(sum(total_fidelity)))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(steps_epi)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Rewards')
    plt.title('The number of steps:{}'.format(np.average(steps_epi)))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    error=env.error_counter
    error1=env1.error_counter
    error2=env2.error_counter
    error3=env3.error_counter
    error4=env4.error_counter
    error5=env5.error_counter
    error6=env6.error_counter
    error7=env7.error_counter
    results=mannwhitney(total_episodes,error)
    results1=mannwhitney(total_episodes,error1)
    results2=mannwhitney(total_episodes,error2)
    results3=mannwhitney(total_episodes,error3)
    results4=mannwhitney(total_episodes,error4)
    results5=mannwhitney(total_episodes,error5)
    results6=mannwhitney(total_episodes,error6)
    results7=mannwhitney(total_episodes,error7)
    results.append([results1,results2,results3,results4,results5,results6,results7,'Reward:'+str(solved/episodes),'Cumulative:'+str(cumulative_reward[-1]),'Steps:'+str(np.mean(steps_epi)),'Fidelity:'+str(sum(total_fidelity))])
    return results
def fourbitsimulation(inpa,ac,ac1,ac2,ac3,ac4,ac5,ac6,ac7,ac8,ac9,ac10,ac11,ac12,ac13,ac14,ac15,ma,qp):
    total_episodes=[]
    solved=0
    episodes=100
    steps_epi=[]
    cum_rev=0
    cumulative_reward=[]
    r=0
    total_fidelity=[]
    count=0
    actions_list=[(0,0),(0,1),(0,2),(0,3),(1,0),(2,0),(1,1),(1,2),(1,3),(2,1),(2,2),(2,3)]
    # run infinitely many episodes
    for i_episode in range(episodes):
        inpu=np.random.randint(0,2,inpa)
        # reset environment and episode reward
        env=Qprotocol(4,inpu,MultiAgent=ma,Qb=qp)
        env1=Qprotocol(4,inpu,MultiAgent=ma,Qb=qp)
        env2=Qprotocol(4,inpu,MultiAgent=ma,Qb=qp)
        env3=Qprotocol(4,inpu,MultiAgent=ma,Qb=qp)
        env4=Qprotocol(4,inpu,MultiAgent=ma,Qb=qp)
        env5=Qprotocol(4,inpu,MultiAgent=ma,Qb=qp)
        env6=Qprotocol(4,inpu,MultiAgent=ma,Qb=qp)
        env7=Qprotocol(4,inpu,MultiAgent=ma,Qb=qp)
        env8=Qprotocol(4,inpu,MultiAgent=ma,Qb=qp)
        env9=Qprotocol(4,inpu,MultiAgent=ma,Qb=qp)
        env10=Qprotocol(4,inpu,MultiAgent=ma,Qb=qp)
        env11=Qprotocol(4,inpu,MultiAgent=ma,Qb=qp)
        env12=Qprotocol(4,inpu,MultiAgent=ma,Qb=qp)
        env13=Qprotocol(4,inpu,MultiAgent=ma,Qb=qp)
        env14=Qprotocol(4,inpu,MultiAgent=ma,Qb=qp)
        env15=Qprotocol(4,inpu,MultiAgent=ma,Qb=qp)
        state,inp=env.reset(4)
        state1,inp=env1.reset(4)
        state2,inp=env2.reset(4)
        state3,inp=env3.reset(4)
        state4,inp=env4.reset(4)
        state5,inp=env5.reset(4)
        state6,inp=env6.reset(4)
        state7,inp=env7.reset(4)
        state8,inp=env8.reset(4)
        state9,inp=env9.reset(4)
        state10,inp=env10.reset(4)
        state11,inp=env11.reset(4)
        state12,inp=env12.reset(4)
        state13,inp=env13.reset(4)
        state14,inp=env14.reset(4)
        state15,inp=env15.reset(4)
        ep_reward = 0
        done1=False
        done2=False
        done3=False
        done4=False
        done5=False
        done6=False
        done7=False
        done8=False
        done9=False
        done10=False
        done11=False
        done12=False
        done13=False
        done14=False
        done15=False
        done16=False
        observation1 = state[0]
        observation2 = state1[0]
        observation3 = state2[0]
        observation4 = state3[0]
        observation5 = state4[0]
        observation6 = state5[0]
        observation7 = state6[0]
        observation8 = state7[0]
        observation9 = state8[0]
        observation10 = state9[0]
        observation11 = state10[0]
        observation12 = state11[0]
        observation13 = state12[0]
        observation14 = state13[0]
        observation15 = state14[0]
        observation16 = state15[0]
        steps_ep=0
        while done1!=True or done2!=True or done3!=True or done4!=True or done5!=True or done6!=True or done7!=True or done8!=True or done9!=True or done10!=True or done11!=True or done12!=True or done13!=True or done14!=True or done15!=True or done16!=True:
            print('This is the episode {}'.format(i_episode))
            observation1 = observation1.reshape(1, -1)
            observation2 = observation2.reshape(1, -1)
            observation7 = observation7.reshape(1, -1)
            observation8 = observation8.reshape(1, -1)
            observation3 = observation3.reshape(1, -1)
            observation4 = observation4.reshape(1, -1)
            observation5 = observation5.reshape(1, -1)
            observation6 = observation6.reshape(1, -1)
            observation9 = observation9.reshape(1, -1)
            observation10 = observation10.reshape(1, -1)
            observation11 = observation11.reshape(1, -1)
            observation12 = observation12.reshape(1, -1)
            observation13 = observation13.reshape(1, -1)
            observation14 = observation14.reshape(1, -1)
            observation15 = observation15.reshape(1, -1)
            observation16 = observation16.reshape(1, -1)
            logits, actiona = sample_action(observation1,ac)
            logits, actionb = sample_action(observation2,ac1)
            logits, actionc = sample_action(observation3,ac2)
            logits, actiond = sample_action(observation4,ac3)
            logits, actione = sample_action(observation5,ac4)
            logits, actionf = sample_action(observation6,ac5)
            logits, actiong = sample_action(observation7,ac6)
            logits, actionh = sample_action(observation8,ac7)
            logits, actioni = sample_action(observation9,ac8)
            logits, actionk = sample_action(observation10,ac9)
            logits, actionl = sample_action(observation11,ac10)
            logits, actionm = sample_action(observation12,ac11)
            logits, actionn = sample_action(observation13,ac12)
            logits, actiono = sample_action(observation14,ac13)
            logits, actionp = sample_action(observation15,ac14)
            logits, actionq = sample_action(observation16,ac15)
            actiona=actiona[0]
            actionb=actionb[0]
            actionc=actionc[0]
            actiond=actiond[0]
            actione=actione[0]
            actionf=actionf[0]
            actiong=actiong[0]
            actionh=actionh[0]
            actioni=actioni[0]
            actionk=actionk[0]
            actionl=actionl[0]
            actionm=actionm[0]
            actionn=actionn[0]
            actiono=actiono[0]
            actionp=actionp[0]
            actionq=actionq[0]
            actionAA=np.array(actions_list[actiona])
            actionBB=np.array(actions_list[actionb])
            actionCC=np.array(actions_list[actionc])
            actionDD=np.array(actions_list[actiond])
            actionEE=np.array(actions_list[actione])
            actionFF=np.array(actions_list[actionf])
            actionGG=np.array(actions_list[actiong])
            actionHH=np.array(actions_list[actionh])
            actionII=np.array(actions_list[actioni])
            actionKK=np.array(actions_list[actionk])
            actionLL=np.array(actions_list[actionl])
            actionMM=np.array(actions_list[actionm])
            actionNN=np.array(actions_list[actionn])
            actionOO=np.array(actions_list[actiono])
            actionPP=np.array(actions_list[actionp])
            actionQQ=np.array(actions_list[actionq])
            stat1,reward1,done1,action_h1,bob_key1=env.step(actionAA)
            stat2,reward2,done2,action_h2,bob_key2=env1.step(actionBB)
            stat3,reward3,done3,action_h3,bob_key3=env2.step(actionCC)
            stat4,reward4,done4,action_h4,bob_key4=env3.step(actionDD)
            stat5,reward5,done5,action_h5,bob_key5=env4.step(actionEE)
            stat6,reward6,done6,action_h6,bob_key6=env5.step(actionFF)
            stat7,reward7,done7,action_h7,bob_key7=env6.step(actionGG)
            stat8,reward8,done8,action_h8,bob_key8=env7.step(actionHH)
            stat9,reward9,done9,action_h9,bob_key9=env8.step(actionII)
            stat10,reward10,done10,action_h10,bob_key10=env9.step(actionKK)
            stat11,reward11,done11,action_h11,bob_key11=env10.step(actionLL)
            stat12,reward12,done12,action_h12,bob_key12=env11.step(actionMM)
            stat13,reward13,done13,action_h13,bob_key13=env12.step(actionNN)
            stat14,reward14,done14,action_h14,bob_key14=env13.step(actionOO)
            stat15,reward15,done15,action_h15,bob_key15=env14.step(actionPP)
            stat16,reward16,done16,action_h16,bob_key16=env15.step(actionQQ)
            observation1=stat1[0]
            observation2=stat2[0]
            observation3=stat3[0]
            observation4=stat4[0]
            observation5=stat5[0]
            observation6=stat6[0]
            observation7=stat7[0]
            observation8=stat8[0]
            observation9=stat9[0]
            observation10=stat10[0]
            observation11=stat11[0]
            observation12=stat12[0]
            observation13=stat13[0]
            observation14=stat14[0]
            observation15=stat15[0]
            observation16=stat16[0]
            steps_ep+=1
            if done1:
                bob_key=bob_key1
            if done2:
                bob_key=bob_key2
            if done3:
                bob_key=bob_key3
            if done4:
                bob_key=bob_key4
            if done5:
                bob_key=bob_key5
            if done6:
                bob_key=bob_key6
            if done7:
                bob_key=bob_key7
            if done8:
                bob_key=bob_key8
            if done9:
                bob_key=bob_key9
            if done10:
                bob_key=bob_key10
            if done11:
                bob_key=bob_key11
            if done12:
                bob_key=bob_key12
            if done13:
                bob_key=bob_key13
            if done14:
                bob_key=bob_key14
            if done15:
                bob_key=bob_key15
            if done16:
                bob_key=bob_key16
            if done1==True or done2==True or done3==True or done4==True or done5==True or done6==True or done7==True or done8==True or done9==True or done10==True or done11==True or done12==True or done13==True or done14==True or done15==True or done16==True:
                steps_epi.append(steps_ep)
                if len(inp)==4 and len(bob_key)==len(inp):
                    inpus=''.join(str(x) for x in inp)
                    bob_keys=''.join(str(x) for x in bob_key[:len(inp)])
                    tp=np.array(LogicalStates4bit.loc[:,inpus]).T*np.array(LogicalStates4bit.loc[bob_keys,:])
                    Fidelity=abs(sum(tp))**2
                    total_fidelity.append(Fidelity)
                else:
                    total_fidelity.append(0)
                if reward1==1 or reward2==1 or reward3==1 or reward4==1 or reward5==1 or reward6==1 or reward7==1 or reward8==1 or reward9==1 or reward10==1 or reward11==1 or reward12==1 or reward13==1 or reward14==1 or reward15==1 or reward16==1:
                    count+=1
                    r=1
                    cum_rev+=r
                    cumulative_reward.append(cum_rev)
                    solved+=1
                    total_episodes.append(r)
                    break
                else:
                    r=-1
                    cum_rev+=r
                    cumulative_reward.append(cum_rev)
                    solved+=0
                    total_episodes.append(r)
                    break
    plt.figure(figsize=(13, 13))
    plt.plot(total_episodes)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Rewards')
    plt.grid(True,which="both",ls="--",c='gray')
    plt.title('The simulation has been solved the environment '+str(len(inpu))+' Proximal Policy Evaluation Rewards:{}'.format(solved/episodes))
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(cumulative_reward)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Rewards')
    plt.title('The simulation has been solved the environment '+str(len(inpu))+' Proximal Policy Evaluation cumulative:{}'.format(sum(cumulative_reward)))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(total_fidelity)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Fidelity')
    plt.title('The simulation has been solved the environment '+str(len(inpu))+' Proximal Policy Evaluation Fidelity:{}'.format(sum(total_fidelity)))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    plt.figure(figsize=(13, 13))
    plt.plot(steps_epi)
    plt.xlabel(f'Number of episodes')
    plt.ylabel('Rewards')
    plt.title('The number of steps:{}'.format(np.average(steps_epi)))
    plt.grid(True,which="both",ls="--",c='gray')
    plt.show()
    error=env.error_counter
    error1=env1.error_counter
    error2=env2.error_counter
    error3=env3.error_counter
    error4=env4.error_counter
    error5=env5.error_counter
    error6=env6.error_counter
    error7=env7.error_counter
    error8=env8.error_counter
    error9=env9.error_counter
    error10=env10.error_counter
    error11=env11.error_counter
    error12=env12.error_counter
    error13=env13.error_counter
    error14=env14.error_counter
    error15=env15.error_counter
    results=mannwhitney(total_episodes,error)
    results1=mannwhitney(total_episodes,error1)
    results2=mannwhitney(total_episodes,error2)
    results3=mannwhitney(total_episodes,error3)
    results4=mannwhitney(total_episodes,error4)
    results5=mannwhitney(total_episodes,error5)
    results6=mannwhitney(total_episodes,error6)
    results7=mannwhitney(total_episodes,error7)
    results8=mannwhitney(total_episodes,error8)
    results9=mannwhitney(total_episodes,error9)
    results10=mannwhitney(total_episodes,error10)
    results11=mannwhitney(total_episodes,error11)
    results12=mannwhitney(total_episodes,error12)
    results13=mannwhitney(total_episodes,error13)
    results14=mannwhitney(total_episodes,error14)
    results15=mannwhitney(total_episodes,error15)
    results.append([results1,results2,results3,results4,results5,results6,results7,results8,results9,results10,results11,results12,results13,results14,results15,'Reward:'+str(solved/episodes),'Cumulative:'+str(cumulative_reward[-1]),'Steps:'+str(np.mean(steps_epi)),'Fidelity:'+str(sum(total_fidelity))])
    return results

def load_weightsOne():
    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
    critic.load_weights(path+ '_criticOne0.h5')
    actor.load_weights(path+ '_actorOne0.h5')
def load_weightsTwo():
    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
    critic1.load_weights(path+ '_criticOne1.h5')
    actor1.load_weights(path+ '_actorOne1.h5')





def load_weightsOne():
    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
    critic.load_weights(path+ '_criticOne00.h5')
    actor.load_weights(path+ '_actorOne00.h5')
def load_weightsTwo():
    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
    critic1.load_weights(path+ '_criticOne01.h5')
    actor1.load_weights(path+ '_actorOne01.h5')
def load_weightsThree():
    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
    critic2.load_weights(path+ '_criticOne10.h5')
    actor2.load_weights(path+ '_actorOne10.h5')
def load_weightsFour():
    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
    critic3.load_weights(path+ '_criticOne11.h5')
    actor3.load_weights(path+ '_actorOne11.h5')
    

def load_weightsOne():
    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
    critic1.load_weights(path+ '_criticOne000.h5')
    actor1.load_weights(path+ '_actorOne000.h5')
def load_weightsTwo():
    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
    critic2.load_weights(path+ '_criticOne001.h5')
    actor2.load_weights(path+ '_actorOne001.h5')
def load_weightsThree():
    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
    critic3.load_weights(path+ '_criticOne010.h5')
    actor3.load_weights(path+ '_actorOne010.h5')
def load_weightsFour():
    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
    critic4.load_weights(path+ '_criticOne011.h5')
    actor4.load_weights(path+ '_actorOne011.h5')
def load_weightsFive():
    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
    critic5.load_weights(path+ '_criticOne100.h5')
    actor5.load_weights(path+ '_actorOne100.h5')
def load_weightsSix():
    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
    critic6.load_weights(path+ '_criticOne101.h5')
    actor6.load_weights(path+ '_actorOne101.h5')
def load_weightsSeven():
    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
    critic7.load_weights(path+ '_criticOne110.h5')
    actor7.load_weights(path+ '_actorOne110.h5')
def load_weightsEight():
    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
    critic8.load_weights(path+ '_criticOne111.h5')
    actor8.load_weights(path+ '_actorOne111.h5')


def load_weightsOne():
    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
    critic1.load_weights(path+ '_criticOne0000.h5')
    actor1.load_weights(path+ '_actorOne0000.h5')
def load_weightsTwo():
    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
    critic2.load_weights(path+ '_criticOne0001.h5')
    actor2.load_weights(path+ '_actorOne0001.h5')
def load_weightsThree():
    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
    critic3.load_weights(path+ '_criticOne0010.h5')
    actor3.load_weights(path+ '_actorOne0010.h5')
def load_weightsFour():
    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
    critic4.load_weights(path+ '_criticOne0011.h5')
    actor4.load_weights(path+ '_actorOne0011.h5')
def load_weightsFive():
    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
    critic5.load_weights(path+ '_criticOne0100.h5')
    actor5.load_weights(path+ '_actorOne0100.h5')
def load_weightsSix():
    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
    critic6.load_weights(path+ '_criticOne0101.h5')
    actor6.load_weights(path+ '_actorOne0101.h5')
def load_weightsSeven():
    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
    critic7.load_weights(path+ '_criticOne0110.h5')
    actor7.load_weights(path+ '_actorOne0110.h5')
def load_weightsEight():
    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
    critic8.load_weights(path+ '_criticOne0111.h5')
    actor8.load_weights(path+ '_actorOne0111.h5')
def load_weightsNine():
    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
    critic9.load_weights(path+ '_criticOne1000.h5')
    actor9.load_weights(path+ '_actorOne1000.h5')
def load_weightsTen():
    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
    critic10.load_weights(path+ '_criticOne1001.h5')
    actor10.load_weights(path+ '_actorOne1001.h5')
def load_weightsEleven():
    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
    critic11.load_weights(path+ '_criticOne1010.h5')
    actor11.load_weights(path+ '_actorOne1010.h5')
def load_weightsTwelve():
    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
    critic12.load_weights(path+ '_criticOne1011.h5')
    actor12.load_weights(path+ '_actorOne1011.h5')
def load_weightsThirteen():
    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
    critic13.load_weights(path+ '_criticOne1101.h5')
    actor13.load_weights(path+ '_actorOne1101.h5')
def load_weightsFourteen():
    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
    critic14.load_weights(path+ '_criticOne1110.h5')
    actor14.load_weights(path+ '_actorOne1110.h5')
def load_weightsFifteen():
    path= '/home/Optimus/Desktop/QuantumComputingThesis/'
    critic15.load_weights(path+ '_criticOne1111.h5')
    actor15.load_weights(path+ '_actorOne1111.h5')
#save_weights()
# =============================================================================
# Classical Channel     
# =============================================================================
observation_dimensions = 4
num_actions = 12
steps_per_epoch=15
# Initialize the buffer
buffer = Buffer(observation_dimensions, steps_per_epoch)

# Initialize the actor and the critic as keras models
observation_input = keras.Input(shape=(observation_dimensions,), dtype=tf.float32)
logits = mlp(observation_input, list(hidden_sizes) + [num_actions], tf.tanh, None)
# Initialize the policy and the value function optimizers
policy_optimizer = keras.optimizers.Adam(learning_rate=policy_learning_rate)
value_optimizer = keras.optimizers.Adam(learning_rate=value_function_learning_rate)
rewards_during_training=[]
value = tf.squeeze(mlp(observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1)
actor = keras.Model(inputs=observation_input, outputs=logits)
critic = keras.Model(inputs=observation_input, outputs=value)
actor,critic,r=proximalpo(1,actor,critic,False,False)
print(r,file=open('randomOneBitPPOTraining.txt','w'))
r=pposimulation(1, actor,False,False)
print(r,file=open('randomOneBitPPOTesting.txt','w'))
#value = tf.squeeze(mlp(observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1)
actor = keras.Model(inputs=observation_input, outputs=logits)
critic = keras.Model(inputs=observation_input, outputs=value)
actor,critic,r=proximalpo(2,actor,critic,False,False)
#sys.stdout.close()                # ordinary file object
#sys.stdout = temp 
print(r,file=open('randomTwoBitPPOTraining.txt','w'))
r=pposimulation(2, actor,False,False)
print(r,file=open('randomTwoBitPPOTesting.txt','w'))
#value = tf.squeeze(mlp(observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1)
actor = keras.Model(inputs=observation_input, outputs=logits)
critic = keras.Model(inputs=observation_input, outputs=value)
actor,critic,r=proximalpo(3,actor,critic,False,False)
print(r,file=open('randomThreeBitPPOTraining.txt','w'))
r=pposimulation(3, actor,False,False)
print(r,file=open('randomtThreeBitPPOTesting.txt','w'))
#value = tf.squeeze(mlp(observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1)
actor = keras.Model(inputs=observation_input, outputs=logits)
critic = keras.Model(inputs=observation_input, outputs=value)
actor,critic,r=proximalpo(4,actor,critic,False,False)
print(r,file=open('randomFourBitPPOraining.txt','w'))
r=pposimulation(4, actor,False,False)
print(r,file=open('randomFourBitPPOTesting.txt','w'))
# =============================================================================
# Quantum Channel     
# =============================================================================
#value = tf.squeeze(mlp(observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1)
actor = keras.Model(inputs=observation_input, outputs=logits)
critic = keras.Model(inputs=observation_input, outputs=value)
actor,critic,r=proximalpo(1,actor,critic,False,True)
print(r,file=open('randomOneQBitPPOTraining.txt','w'))
r=pposimulation(1, actor,False,True)
print(r,file=open('randomOneQBitPPOTesting.txt','w'))
#value = tf.squeeze(mlp(observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1)
actor = keras.Model(inputs=observation_input, outputs=logits)
critic = keras.Model(inputs=observation_input, outputs=value)
actor,critic,r=proximalpo(2,actor,critic,False,True)
#sys.stdout.close()                # ordinary file object
#sys.stdout = temp 
print(r,file=open('randomTwoQBitPPOTraining.txt','w'))
r=pposimulation(2, actor,False,True)
print(r,file=open('randomTwoQBitPPOTesting.txt','w'))
#value = tf.squeeze(mlp(observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1)
actor = keras.Model(inputs=observation_input, outputs=logits)
critic = keras.Model(inputs=observation_input, outputs=value)
actor,critic,r=proximalpo(3,actor,critic,False,True)
print(r,file=open('randomThreeQBitPPOTraining.txt','w'))
r=pposimulation(3, actor,False,True)
print(r,file=open('randomtThreeQBitPPOTesting.txt','w'))
#value = tf.squeeze(mlp(observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1)
actor = keras.Model(inputs=observation_input, outputs=logits)
critic = keras.Model(inputs=observation_input, outputs=value)
actor,critic,r=proximalpo(4,actor,critic,False,True)
print(r,file=open('randomFourQBitPPOraining.txt','w'))
r=pposimulation(4, actor,False,True)
print(r,file=open('randomFourQBitPPOTesting.txt','w'))










value = tf.squeeze(mlp(observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1)
actor = keras.Model(inputs=observation_input, outputs=logits)
critic = keras.Model(inputs=observation_input, outputs=value)
value = tf.squeeze(mlp(observation_input, list(hidden_sizes) + [1], tf.tanh, None), axis=1)
actor1 = keras.Model(inputs=observation_input, outputs=logits)
critic1 = keras.Model(inputs=observation_input, outputs=value)
actor,critic,r=proximalpo([0],actor,critic,True,True)
print(r,file=open('randomOneQBit[0]PPOTraining.txt','w'))
actor1,critic1,r=proximalpo([1],actor1,critic1,True,True)
print(r,file=open('randomOneQBit[1]PPOTraining.txt','w'))
#load_weightsOne()
#load_weightsTwo()
r=onebitsimulation(1,actor,actor1,True,True)
print(r,file=open('randomOneQBitMULTIPPOTestings.txt','w'))

actor = keras.Model(inputs=observation_input, outputs=logits)
critic = keras.Model(inputs=observation_input, outputs=value)
actor1 = keras.Model(inputs=observation_input, outputs=logits)
critic1 = keras.Model(inputs=observation_input, outputs=value)
actor2 = keras.Model(inputs=observation_input, outputs=logits)
critic2 = keras.Model(inputs=observation_input, outputs=value)
actor3 = keras.Model(inputs=observation_input, outputs=logits)
critic3 = keras.Model(inputs=observation_input, outputs=value)

actor,critic,r=proximalpo([0,0],actor,critic,True,False)
print(r,file=open('randomTwoQBit[0,0]PPOTraining.txt','w'))
actor1,critic1,r=proximalpo([0,1],actor1,critic1,True,False)
print(r,file=open('randomTwoQBit[0,1]PPOTraining.txt','w'))
actor2,critic2,r=proximalpo([1,0],actor2,critic2,True,False)
print(r,file=open('randomTwoQBit[1,0]PPOTraining.txt','w'))
actor3,critic3,r=proximalpo([1,1],actor3,critic3,True,False)
print(r,file=open('randomTwoQBit[1,1]PPOTraining.txt','w'))
#load_weightsOne()
#load_weightsTwo()
#load_weightsThree()
#load_weightsFour()
r=twobitsimulation(2,actor,actor1,actor2,actor3,True,False)
print(r,file=open('randomTwoQBitMULTIPPOTesting.txt','w'))

actor = keras.Model(inputs=observation_input, outputs=logits)
critic = keras.Model(inputs=observation_input, outputs=value)
actor1 = keras.Model(inputs=observation_input, outputs=logits)
critic1 = keras.Model(inputs=observation_input, outputs=value)
actor2 = keras.Model(inputs=observation_input, outputs=logits)
critic2 = keras.Model(inputs=observation_input, outputs=value)
actor3 = keras.Model(inputs=observation_input, outputs=logits)
critic3 = keras.Model(inputs=observation_input, outputs=value)
actor4 = keras.Model(inputs=observation_input, outputs=logits)
critic4 = keras.Model(inputs=observation_input, outputs=value)
actor5 = keras.Model(inputs=observation_input, outputs=logits)
critic5 = keras.Model(inputs=observation_input, outputs=value)
actor6 = keras.Model(inputs=observation_input, outputs=logits)
critic6 = keras.Model(inputs=observation_input, outputs=value)
actor7 = keras.Model(inputs=observation_input, outputs=logits)
critic7 = keras.Model(inputs=observation_input, outputs=value)

actor,critic,r=proximalpo([0,0,0],actor,critic,True,False)
print(r,file=open('randomThreeQBit[0,0,0]PPOTraining.txt','w'))
actor1,critic1,r=proximalpo([0,0,1],actor1,critic1,True,False)
print(r,file=open('randomThreeQBit[0,0,1]PPOTraining.txt','w'))
actor2,critic2,r=proximalpo([0,1,0],actor2,critic2,True,False)
print(r,file=open('randomThreeQBit[0,1,0]PPOTraining.txt','w'))
actor3,critic3,r=proximalpo([0,1,1],actor3,critic3,True,False)
print(r,file=open('randomThreeQBit[0,1,1]PPOTraining.txt','w'))
actor4,critic4,r=proximalpo([1,0,0],actor4,critic4,True,False)
print(r,file=open('randomThreeQBit[1,0,0]PPOTraining.txt','w'))
actor5,critic5,r=proximalpo([1,0,1],actor5,critic5,True,False)
print(r,file=open('randomThreeQBit[1,0,1]PPOTraining.txt','w'))
actor6,critic6,r=proximalpo([1,1,0],actor6,critic6,True,False)
print(r,file=open('randomThreeQBit[1,1,0]PPOTraining.txt','w'))
actor7,critic7,r=proximalpo([1,1,1],actor7,critic7,True,False)
print(r,file=open('randomThreeQBit[1,1,1]PPOTraining.txt','w'))
#load_weightsOne()
#load_weightsTwo()
#load_weightsThree()
#load_weightsFour()
#load_weightsFive()
#load_weightsSix()
#load_weightsSeven()
#load_weightsEight()
r=threebitsimulation(3,actor,actor1,actor2,actor3,actor4,actor5,actor6,actor7,True,False)
print(r,file=open('randomThreeQBitMULTIPPOTesting.txt','w'))


actor = keras.Model(inputs=observation_input, outputs=logits)
critic = keras.Model(inputs=observation_input, outputs=value)
actor1 = keras.Model(inputs=observation_input, outputs=logits)
critic1 = keras.Model(inputs=observation_input, outputs=value)
actor2 = keras.Model(inputs=observation_input, outputs=logits)
critic2 = keras.Model(inputs=observation_input, outputs=value)
actor3 = keras.Model(inputs=observation_input, outputs=logits)
critic3 = keras.Model(inputs=observation_input, outputs=value)
actor4 = keras.Model(inputs=observation_input, outputs=logits)
critic4 = keras.Model(inputs=observation_input, outputs=value)
actor5 = keras.Model(inputs=observation_input, outputs=logits)
critic5 = keras.Model(inputs=observation_input, outputs=value)
actor6 = keras.Model(inputs=observation_input, outputs=logits)
critic6 = keras.Model(inputs=observation_input, outputs=value)
actor7 = keras.Model(inputs=observation_input, outputs=logits)
critic7 = keras.Model(inputs=observation_input, outputs=value)
actor8 = keras.Model(inputs=observation_input, outputs=logits)
critic8 = keras.Model(inputs=observation_input, outputs=value)
actor9 = keras.Model(inputs=observation_input, outputs=logits)
critic9 = keras.Model(inputs=observation_input, outputs=value)
actor10 = keras.Model(inputs=observation_input, outputs=logits)
critic10 = keras.Model(inputs=observation_input, outputs=value)
actor11 = keras.Model(inputs=observation_input, outputs=logits)
critic11 = keras.Model(inputs=observation_input, outputs=value)
actor12 = keras.Model(inputs=observation_input, outputs=logits)
critic12 = keras.Model(inputs=observation_input, outputs=value)
actor13 = keras.Model(inputs=observation_input, outputs=logits)
critic13 = keras.Model(inputs=observation_input, outputs=value)
actor14 = keras.Model(inputs=observation_input, outputs=logits)
critic14 = keras.Model(inputs=observation_input, outputs=value)
actor15 = keras.Model(inputs=observation_input, outputs=logits)
critic15 = keras.Model(inputs=observation_input, outputs=value)
actor,critic,r=proximalpo([0,0,0,0],actor,critic,True,False)
print(r,file=open('randomFourBit[0,0,0,0]PPOTraining.txt','w'))
actor1,critic1,r=proximalpo([0,0,0,1],actor1,critic1,True,False)
print(r,file=open('randomFourBit[0,0,0,1]PPOTraining.txt','w'))
actor2,critic2,r=proximalpo([0,0,1,0],actor2,critic2,True,False)
print(r,file=open('randomFourBit[0,0,1,0]PPOTraining.txt','w'))
actor3,critic3,r=proximalpo([0,0,1,1],actor3,critic3,True,False)
print(r,file=open('randomFourBit[0,0,1,1]PPOTraining.txt','w'))
actor4,critic4,r=proximalpo([0,1,0,0],actor4,critic4,True,False)
print(r,file=open('randomFourBit[0,1,0,0]PPOTraining.txt','w'))
actor5,critic5,r=proximalpo([0,1,0,1],actor5,critic5,True,False)
print(r,file=open('randomFourBit[0,1,0,1]PPOTraining.txt','w'))
actor6,critic6,r=proximalpo([0,1,1,0],actor6,critic6,True,False)
print(r,file=open('randomFourBit[0,1,1,0]PPOTraining.txt','w'))
actor7,critic7,r=proximalpo([0,1,1,1],actor7,critic7,True,False)
print(r,file=open('randomFourBit[0,1,1,1]PPOTraining.txt','w'))
actor8,critic8,r=proximalpo([1,0,0,0],actor8,critic8,True,False)
print(r,file=open('randomFourBit[1,0,0,0]PPOTraining.txt','w'))
actor9,critic9,r=proximalpo([1,0,0,1],actor9,critic9,True,False)
print(r,file=open('randomFourBit[1,0,0,1]PPOTraining.txt','w'))
actor10,critic10,r=proximalpo([1,0,1,0],actor10,critic10,True,False)
print(r,file=open('randomFourBit[1,0,1,0]PPOTraining.txt','w'))
actor11,critic11,r=proximalpo([1,0,1,1],actor11,critic11,True,False)
print(r,file=open('randomFourBit[1,0,1,1]PPOTraining.txt','w'))
actor12,critic12,r=proximalpo([1,1,0,0],actor12,critic12,True,False)
print(r,file=open('randomFourBit[1,1,0,0]PPOTraining.txt','w'))
actor13,critic13,r=proximalpo([1,1,0,1],actor13,critic13,True,False)
print(r,file=open('randomFourBit[1,1,0,1]PPOTraining.txt','w'))
actor14,critic14,r=proximalpo([1,1,1,0],actor14,critic14,True,False)
print(r,file=open('randomFourBit[1,1,1,0]PPOTraining.txt','w'))
actor15,critic15,r=proximalpo([1,1,1,1],actor15,critic15,True,False)
print(r,file=open('randomFourBit[1,1,1,1]PPOTraining.txt','w'))
#load_weightsOne()
#load_weightsTwo()
#load_weightsThree()
#load_weightsFour()
#load_weightsFive()
#load_weightsSix()
#load_weightsEight()
#load_weightsNine()
#load_weightsTen()
#load_weightsEleven()
#load_weightsTwelve()
#load_weightsThirteen()
#load_weightsFourteen()
#load_weightsFifteen()
r=fourbitsimulation(4,actor,actor1,actor2,actor3,actor4,actor5,actor6,actor7,actor8,actor9,actor10,actor11,actor12,actor13,actor14,actor15,True,False)
print(r,file=open('randomFourBitMULTIPPOTesting.txt','w'))