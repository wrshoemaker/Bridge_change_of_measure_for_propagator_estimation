# %%
%reset
import sys
import matplotlib
from scipy.optimize import curve_fit
import numpy as np
from numpy import exp as exp
from numpy import log as log
from numpy import sqrt as sqrt
import networkx as nx
import matplotlib.pyplot as plt
import time
import math
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.patches as patches


import utils



matplotlib.rcParams.update({'figure.autolayout': True})
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size':12})
matplotlib.rcParams['axes.linewidth']=1 #Mattia had 2 -> Grosor del marco (doble del standard)



def Plot_bonito(xlabel=r" $ x$",ylabel=r"$ y$",label_font_size=15,ticks_size=12,y_size=2.4,x_size=3.2):
    plt.figure(figsize=(x_size,y_size))
    plt.tick_params(labelsize=24)
    plt.xlabel(xlabel,fontsize=label_font_size)
    plt.ylabel(ylabel,fontsize=label_font_size)
    plt.xticks(fontsize=ticks_size)
    plt.yticks(fontsize=ticks_size)
    plt.locator_params(axis="both", nbins=5,tight=True)
def axis_bonito(ax,xlabel=r" $ x$",ylabel=r"$ y$",label_font_size=15,ticks_size=12):
    ax.set_xlabel(xlabel,fontsize=label_font_size)
    ax.set_ylabel(ylabel,fontsize=label_font_size)
    ax.tick_params(axis="x", labelsize=ticks_size)
    ax.tick_params(axis="y", labelsize=ticks_size)
    # For two column figure that fits in one column of a two-columns paper:
        #fig, (ax1,ax2) = plt.subplots(1, 2,figsize=(4.2,1.8))
    # For three rows figure without space between plots:
        #fig, (ax1,ax2,ax3) = plt.subplots(3, 1,sharex=True,figsize=(2,3.6))
        #fig.tight_layout()
       # #-----plots--------
        # ax1.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) #to remove x ticks in pdf file
        #plt.subplots_adjust(hspace=0)
#%% Functions





#%% Example usage
mu = 1.0
tau = 2.0
x0 = mu
dt = 0.01
n_steps = 100000

t, x = utils.simulate_ou_exact(mu, tau, x0, dt, n_steps, random_state=42)
Plot_bonito(xlabel=r"$t$",ylabel=r"$X_t$")
plt.plot(t,x)
plt.show();plt.close()

taus = np.linspace(0.5,5,1000)
ls = np.zeros(len(taus))
for ii,param in enumerate(taus):
    loglik = utils.ou_path_loglik(x, mu, param, dt)
    ls[ii] = loglik
Plot_bonito(xlabel=r"$\tau$",ylabel=r"$C+ \log\left( \rho(\tau)\right)$")
plt.plot(taus,ls)
plt.show();plt.close()
print("MLE(tau)=",taus[np.argmax(ls)])

mus = np.linspace(-5,5,1000)
ls = np.zeros(len(taus))
for ii,param in enumerate(mus):
    loglik = utils.ou_path_loglik(x, param, tau, dt)
    ls[ii] = loglik
Plot_bonito(xlabel=r"$\mu$",ylabel=r"$C+ \log\left( \rho(\mu)\right)$")
plt.plot(mus,ls)
plt.show();plt.close()
print("MLE(mu)=",mus[np.argmax(ls)])
# %%
