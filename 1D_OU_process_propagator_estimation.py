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

# %% Functions
def gaussian(x, mean, variance):
    """Evaluate the Gaussian distribution at x with given mean and variance."""
    coeff = 1.0 / np.sqrt(2 * np.pi * variance)
    exponent = -((x - mean) ** 2) / (2 * variance)
    return coeff * np.exp(exponent)

def evaluate_stationary_distribution(x, model, mu,K,D):
    """
    Evaluate either a Gaussian (if model == 'OU') or Gamma (if model == 'DE') distribution.

    Parameters:
    - x: scalar or array of input values
    - model: 'OU' for Gaussian, 'DE' for Gamma
    - param1: mean (for OU) or shape (for DE)
    - param2: variance (for OU) or rate (for DE)
    """
    if model == 'OU':
        param1 = mu
        param2 = D**2/K/2
        return gaussian(x, param1, param2)
    elif model == 'DE':
        param1 = 2*K*mu/D**2 #shape
        param2 = 2*K/D**2 #rate
        return gamma_distribution(x, param1, param2)
    else:
        raise ValueError("Unknown model. Use 'OU' for Gaussian or 'DE' for Gamma.")

def propagator_OU_process(x, t, x0, mu, K, D):
    """
    Compute the propagator (transition probability density) of the Ornstein-Uhlenbeck process.

    Parameters:
    - x: scalar or array of positions where to evaluate the propagator
    - t: time
    - x0: initial position
    - mu: mean reversion level
    - K: strength of the restoring force
    - D: diffusion coefficient

    Returns:
    - Propagator evaluated at x and time t.
    """
    mean = mu + (x0 - mu) * np.exp(-K * t)
    variance = (D**2 / (2 * K)) * (1 - np.exp(-2 * K * t))
    return gaussian(x, mean, variance)

def propagator_WI_process(x, t, x0, D):
    """
    Compute the propagator (transition probability density) of the Wiener (Brownian motion) process.

    Parameters:
    - x: scalar or array of positions where to evaluate the propagator
    - t: time
    - x0: initial position
    - D: diffusion coefficient

    Returns:
    - Propagator evaluated at x and time t.
    """
    mean = x0
    variance = D**2 * t
    return gaussian(x, mean, variance)

def Fill_gaps_with_Wiener_bridges(dt, tf, t0=0, x0=0, xf=0):
    """
    Simulate a Wiener bridge on [t0, tf] from x0 to xf
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

def Fill_gaps_with_Brownian_bridges(dt, tf, t0=0.0, x0=0.0, xf=0.0, D=1.0):
    """
    Simulate a Brownian bridge on [t0, tf] from x0 to xf
    with diffusion coefficient D, using time step dt.

    dX_t = D dW_t

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
    if D <= 0:
        raise ValueError("D must be positive")

    # Number of points
    npoints = int(np.round((tf - t0) / dt)) + 1
    ts = t0 + np.arange(npoints) * dt
    ts[-1] = tf  # force exact final time

    xs = np.zeros(npoints)
    xs[0] = x0
    xs[-1] = xf

    x = x0
    for i in range(1, npoints - 1):
        s = ts[i - 1]   # previous time
        t = ts[i]       # current time
        h = t - s       # time step

        # Brownian bridge conditional law
        denom = (tf - s)

        mean = x + h * (xf - x) / denom
        var  = (D**2) * h * (tf - t) / denom
        std  = np.sqrt(var)

        x = mean + std * np.random.normal()
        xs[i] = x

    return ts, xs


# %% Additive OU process
D = 0.1  # Diffusion coefficient
mu = 1.0  # mean
k = 1.0  # strength of the restoring force (inverse of correlation time)

x0 = mu
tf = 1.0
t0 = 0.0
tfmt0 = tf - t0
mean = mu + (x0 - mu) * np.exp(-k * tfmt0)
var = (D**2 / (2 * k)) * (1 - np.exp(-2 * k * tfmt0))
xf = mean+ np.sqrt(var) * np.random.normal()

target_prop = propagator_OU_process(xf, tfmt0, x0, mu, k, D)


# %% Draw bridges, evaluate Radon-Nykodim derivatives and compute empirical propagator

Nbridges = 1000
dt_bridge = 0.01

tfmt0 = tf - t0  # make sure this is defined

Ls = np.zeros(Nbridges)

for i in range(Nbridges):
    ts_bridge, xs_bridge = Fill_gaps_with_Brownian_bridges(dt_bridge, tf, t0, x0, xf,D)

    # OU drift
    b = -k * (xs_bridge - mu)

    # Girsanov exponent components (left-point rule)
    dx = np.diff(xs_bridge)
    dt = np.diff(ts_bridge)

    state_integral = (1.0 / D**2) * np.sum(b[:-1] * dx) #This could be computed exactly
    time_integral  = (1.0 / D**2) * np.sum(b[:-1]**2 * dt)

    log_weight = state_integral - 0.5 * time_integral
    weight = np.exp(log_weight)

    prop_WI = propagator_WI_process(xf, tfmt0, x0, D)  # Wiener propagator
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
