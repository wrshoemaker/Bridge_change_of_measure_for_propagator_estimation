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
matplotlib.rcParams.update({'figure.autolayout': True})
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
matplotlib.rcParams['mathtext.fontset'] = 'cm'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
matplotlib.rcParams.update({'font.size':12})
matplotlib.rcParams['axes.linewidth']=1 #Grosor del marco (doble del standard)
def Plot_bonito(xlabel=r" $ x$",ylabel=r"$ y$",label_font_size=15,ticks_size=12,y_size=2.4,x_size=3.2):
    plt.figure(figsize=(x_size,y_size))
    plt.tick_params(labelsize=24)
    plt.xlabel(xlabel,fontsize=label_font_size)
    plt.ylabel(ylabel,fontsize=label_font_size)
    plt.xticks(fontsize=ticks_size)
    plt.yticks(fontsize=ticks_size)
    plt.locator_params(axis="both", nbins=5,tight=True)
def axis_bonito(ax,xlabel=r" $ x$",ylabel=r"$ y$",label_font_size=12,ticks_size=10):
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
        #plt.subplots_adjust(hspace=0)

# %% Functions
def Fill_gaps_with_Wiener_bridges(dt, tf, t0=0, x0=0, xf=0):
    """
    Simulate a Wiener (Brownian) bridge on [t0, tf] from x0 to xf
    using a time step dt.

    Returns
    -------
    ts : np.ndarray
        Time grid from t0 to tf.
    xs : np.ndarray
        Bridge values X_t at those times.
    """
    if tf <= t0:
        raise ValueError("tf must be greater than t0")
    if dt <= 0:
        raise ValueError("dt must be positive")

    # Number of points
    npoints = int(np.round((tf - t0) / dt)) + 1
    ts = t0 + np.arange(npoints) * dt
    ts[-1] = tf  # force exact final time

    xs = np.zeros(npoints)
    xs[0] = x0
    xs[-1] = xf

    x = x0
    for i in range(1, npoints - 1):
        s = ts[i - 1]  # previous time
        t = ts[i]      # current time
        h = t - s      # time step (≈ dt)

        # Brownian bridge conditional law:
        # X_t | X_s = x, X_tf = xf ~ N(mean, var) with
        # mean = x + (h / (tf - s)) * (xf - x)
        # var  = h * (tf - t) / (tf - s)
        denom = (tf - s)
        mean = x + h * (xf - x) / denom
        var = h * (tf - t) / denom
        std = np.sqrt(var)

        x = mean + std * np.random.normal()
        xs[i] = x

    return ts, xs


# %%
tf = 20
Plot_bonito(ylabel=r"$X_t$",xlabel=r"$t$")
ts = np.linspace(0,tf,200)
dt = ts[1]-ts[0]
points = len(ts)
xs = np.zeros(points)
N = 50
for jj in range(N):
    x = 0
    for ii in range(1,points):
        x += np.random.normal()*sqrt(dt)
        xs[ii] = x
    plt.plot(ts,xs,alpha=0.3)
plt.plot(ts,2.0*sqrt(ts) ,color="black",ls="--",label=r"$\pm\sqrt{t}$")
plt.plot(ts,-2.0*sqrt(ts),color="black",ls="--")
plt.scatter([0],[0],color="black",zorder=100)
plt.legend(frameon=False,fontsize=10)
plt.savefig("figures/Wiener_path.pdf",bbox_inches="tight")
plt.show();plt.close()

Plot_bonito(ylabel=r"$X_t$",xlabel=r"$t$")


N = 50
for jj in range(N):
    ts,xs =  Fill_gaps_with_Wiener_bridges(dt,tf,t0=0,xf=0,x0=0)    
    plt.plot(ts,xs,alpha=0.3)
plt.plot(ts,2.0*sqrt(ts) ,color="black",ls="--")
plt.plot(ts,-2.0*sqrt(ts),color="black",ls="--")
plt.scatter([0,20],[0,0],color="black",zorder=100)
plt.savefig("figures/Wiener_bridge_path.pdf",bbox_inches="tight")
plt.show();plt.close()
# %%
