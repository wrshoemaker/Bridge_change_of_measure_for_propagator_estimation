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



# %% Additive OU process
D = 1.0  # Diffusion coefficient
mu = 1.0  # mean
k = 1.0  # strength of the restoring force (inverse of correlation time)

x0 = mu
xf = mu+0.1
tf = 1.0
t0 = 0.0
tfmt0 = tf - t0
target_prop = utils.propagator_OU_process(xf, tfmt0, x0, mu, k, D)


# %% Draw bridges, evaluate Radon-Nykodim derivatives and compute empirical propagator

Nbridges = 1000
dt_bridge = 0.01

tfmt0 = tf - t0  # make sure this is defined

Ls = np.zeros(Nbridges)

for i in range(Nbridges):
    ts_bridge, xs_bridge = utils.Fill_gaps_with_Wiener_bridges(dt_bridge, tf, t0, x0, xf)

    # OU drift
    b = -k * (xs_bridge - mu)

    # Girsanov exponent components (left-point rule)
    dx = np.diff(xs_bridge)
    dt = np.diff(ts_bridge)

    state_integral = (1.0 / D**2) * np.sum(b[:-1] * dx) #This could be computed exactly
    time_integral  = (1.0 / D**2) * np.sum(b[:-1]**2 * dt)

    log_weight = state_integral - 0.5 * time_integral
    weight = np.exp(log_weight)

    prop_WI = utils.propagator_WI_process(xf, tfmt0, x0, D)  # Wiener propagator
    Ls[i] = weight * prop_WI

empirical_prop = np.mean(Ls)

Plot_bonito(xlabel="Sample index", ylabel=r"$L$", x_size=4, y_size=3)
plt.scatter(np.arange(Nbridges), Ls, s=5, alpha=0.4, label="Samples",color="dodgerblue")
plt.axhline(target_prop, color='red', linestyle='--', linewidth=2, label="Target")
plt.axhline(empirical_prop, color='green', linestyle='--', linewidth=2, label="Estimator")
plt.legend(fontsize=10,frameon=False,loc="upper right")
plt.ylim(0,1.2)
plt.savefig("figures/OU_propagator_MC_samples.pdf",bbox_inches="tight",transparent=True)
plt.show();plt.close()

Plot_bonito(ylabel=r"$\rho(L)$", xlabel=r"$L$", x_size=4, y_size=3)
plt.hist(Ls, bins=50, density=True, alpha=0.4, color='dodgerblue', label='Histogram')
plt.axvline(target_prop, color='red', linestyle='--', linewidth=2, label="Target")
plt.axvline(empirical_prop, color='green', linestyle='--', linewidth=2, label="Empirical")

plt.legend(fontsize=10,frameon=False,loc="upper right")
plt.savefig("figures/OU_propagator_MC_histogram.pdf",bbox_inches="tight",transparent=True)
plt.show();plt.close()

print(f"Target OU propagator: {target_prop}")
print(f"Empirical OU propagator from Wiener bridges: {empirical_prop}")
print(f"Relative error: {(abs(empirical_prop - target_prop) / target_prop )*100:.4f} %")


# %%
