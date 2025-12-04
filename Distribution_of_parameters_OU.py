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
#%% Functions

def simulate_ou_exact(mu,tau, x0, dt, n_steps, random_state=None):
    """
    Simulate an Ornstein–Uhlenbeck process using the exact transition:

        dX_t = -(X_t - mu)/tau dt + dW_t

    Exact transition:
        X_{t+dt} | X_t ~ N(mean, var) with
        mean = mu + (X_t - mu) * exp(-dt/tau)
        var  = (tau/2) * (1 - exp(-2*dt/tau))

    Parameters
    ----------
    mu : float
        Long-term mean.
    tau : float
        Relaxation time (tau > 0).
    x0 : float
        Initial value X_0 (treated as fixed).
    dt : float
        Time step.
    n_steps : int
        Number of steps to simulate.
    random_state : int or None
        Seed for reproducibility.

    Returns
    -------
    t : ndarray, shape (n_steps + 1,)
        Time grid.
    x : ndarray, shape (n_steps + 1,)
        Simulated OU trajectory.
    """
    rng = np.random.default_rng(random_state)
    t = np.linspace(0.0, n_steps * dt, n_steps + 1)
    x = np.empty(n_steps + 1, dtype=float)
    x[0] = x0

    exp_term = np.exp(-dt / tau)
    var = (tau / 2.0) * (1.0 - exp_term ** 2)
    std = np.sqrt(var)

    for n in range(n_steps):
        mean = mu + (x[n] - mu) * exp_term
        x[n + 1] = rng.normal(loc=mean, scale=std)

    return t, x


def ou_path_loglik(path, mu, tau, dt, include_prior_x0=False):
    """
    Compute the log-likelihood of an OU path under fixed (mu, tau),
    using the exact OU transition density for constant time step dt.

    Model: dX_t = -(X_t - mu)/tau dt + dW_t, diffusion coefficient = 1.

    Observations at times 0, dt, 2dt, ..., N*dt.
    By default, X_0 is treated as fixed (conditional likelihood).

    Parameters
    ----------
    path : ndarray, shape (N+1,)
        Observed trajectory values [X_0, X_dt, ..., X_{N*dt}].
    mu : float
        OU mean parameter.
    tau : float
        OU time scale parameter (tau > 0).
    dt : float
        Time step between consecutive observations.
    include_prior_x0 : bool, default False
        If True, include stationary prior for X_0:
        X_0 ~ N(mu, tau/2).

    Returns
    -------
    loglik : float
        Log-likelihood of the path under (mu, tau).
    """
    path = np.asarray(path)
    x0 = path[0]
    x_prev = path[:-1]
    x_next = path[1:]

    exp_term = np.exp(-dt / tau)
    mean = mu + (x_prev - mu) * exp_term
    var = (tau / 2.0) * (1.0 - exp_term ** 2)

    log_two_pi = np.log(2.0 * np.pi)
    loglik_increments = -0.5 * (
        log_two_pi + np.log(var) + ((x_next - mean) ** 2) / var
    )

    loglik = np.sum(loglik_increments)

    if include_prior_x0:
        prior_var = tau / 2.0
        loglik_prior_x0 = -0.5 * (
            log_two_pi
            + np.log(prior_var)
            + (x0 - mu) ** 2 / prior_var
        )
        loglik += loglik_prior_x0

    return loglik

#%% Example usage
mu = 1.0
tau = 2.0
x0 = mu
dt = 0.01
n_steps = 100000

t, x = simulate_ou_exact(mu, tau, x0, dt, n_steps, random_state=42)
Plot_bonito(xlabel=r"$t$",ylabel=r"$X_t$")
plt.plot(t,x)
plt.show();plt.close()

taus = np.linspace(0.5,5,1000)
ls = np.zeros(len(taus))
for ii,param in enumerate(taus):
    loglik = ou_path_loglik(x, mu, param, dt)
    ls[ii] = loglik
Plot_bonito(xlabel=r"$\tau$",ylabel=r"$C+ \log\left( \rho(\tau)\right)$")
plt.plot(taus,ls)
plt.show();plt.close()
print("MLE(tau)=",taus[np.argmax(ls)])

mus = np.linspace(-5,5,1000)
ls = np.zeros(len(taus))
for ii,param in enumerate(mus):
    loglik = ou_path_loglik(x, param, tau, dt)
    ls[ii] = loglik
Plot_bonito(xlabel=r"$\mu$",ylabel=r"$C+ \log\left( \rho(\mu)\right)$")
plt.plot(mus,ls)
plt.show();plt.close()
print("MLE(mu)=",mus[np.argmax(ls)])
# %%
