#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 05:15:25 2023

@author: Optimus
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm 
from mpl_toolkits.mplot3d import axes3d
from scipy.stats import norm, multivariate_normal
import math

#%matplotlib inline
#%config InlineBackend.figure_format = 'retina'
#plt.rc('text', usetex=True)
#plt.rc('font', family='serif')
#plt.rcParams['text.latex.preamble'] ='\\usepackage{libertine}\n\\usepackage[utf8]{inputenc}'

import seaborn
seaborn.set(style='whitegrid')
seaborn.set_context('notebook')


sample1 = np.random.normal(0, 0.5, 1000)
sample2 = np.random.normal(1,1,500)

def plot_normal_sample(sample, mu, sigma):
    'Plots an histogram and the normal distribution corresponding to the parameters.'
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 100)
    plt.plot(x, norm.pdf(x, mu, sigma), 'b', lw=2)
    plt.hist(sample, 30, normed=True, alpha=0.2)
    plt.annotate('3$\sigma$', 
                     xy=(mu + 3*sigma, 0),  xycoords='data',
                     xytext=(0, 100), textcoords='offset points',
                     fontsize=15,
                     arrowprops=dict(arrowstyle="->",
                                    connectionstyle="arc,angleA=180,armA=20,angleB=90,armB=15,rad=7"))
    plt.annotate('-3$\sigma$', 
                     xy=(mu -3*sigma, 0), xycoords='data', 
                     xytext=(0, 100), textcoords='offset points',
                     fontsize=15,
                     arrowprops=dict(arrowstyle="->",
                                     connectionstyle="arc,angleA=180,armA=20,angleB=90,armB=15,rad=7"))
plt.figure(figsize=(11,4))
plt.subplot(121)
plot_normal_sample(sample1, 0, 0.5)
plt.title('Sample 1: $\mu=0$, $\sigma=0.5$')
plt.subplot(122)
plot_normal_sample(sample2, 1, 1)
plt.title('Sample 2: $\mu=1$, $\sigma=1$')
plt.tight_layout();

print('Sample 1; estimated mean:', sample1.mean(), ' and std. dev.: ', sample1.std())
print('Sample 2; estimated mean:', sample2.mean(), ' and std. dev.: ', sample2.std())

sample_2d = np.array(list(zip(sample1, np.ones(len(sample1))))).T
plt.scatter(sample_2d[0,:], sample_2d[1,:], marker='x')

cov=np.cov(sample_2d)

def rotate_sample(sample, angle=-45):
    'Rotates a sample by `angle` degrees.'
    theta = (angle/180.) * np.pi
    rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], 
                           [np.sin(theta), np.cos(theta)]])
    return sample.T.dot(rot_matrix).T
rot_sample_2d = rotate_sample(sample_2d)
np.cov(rot_sample_2d)


def autocovariance(Xi, N, k):
    Xs=np.average(Xi)
    aCov = 0.0
    for i in np.arange(0, N-k):
        aCov = (Xi[(i+k)]-Xs)*(Xi[i]-Xs)+aCov
    return  (1./(N))*aCov

autocov[i]=(autocovariance(My_wector, N, h))
