# %%
%reset
import sys
import matplotlib
from scipy.optimize import curve_fit
import numpy as np
from numpy import exp as exp
from numpy import log as log
from numpy import sqrt as sqrt
from scipy.special import gamma  # Γ function
from scipy.special import iv  # modified Bessel I_v
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

def gaussian(x, mean, variance):
    """Evaluate the Gaussian distribution at x with given mean and variance."""
    coeff = 1.0 / np.sqrt(2 * np.pi * variance)
    exponent = -((x - mean) ** 2) / (2 * variance)
    return coeff * np.exp(exponent)

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

def propagator_DE_process(x, t, x0, mu, k, D):
    """
    Exact transition density (propagator) of the DE (a.k.a. CIR) process

        dX_t = k (mu - X_t) dt + D sqrt(X_t) dW_t

    Parameters
    ----------
    x : float or np.ndarray
        Final state(s), x >= 0
    t : float
        Time increment (t > 0)
    x0 : float
        Initial state (x0 >= 0)
    k : float
        Mean-reversion strength
    mu : float
        Long-term mean
    D : float
        Diffusion coefficient

    Returns
    -------
    pdf : float or np.ndarray
        Transition density p(x, t | x0)
    """
    x = np.asarray(x)

    c = 2 * k / (D**2 * (1 - np.exp(-k * t)))
    u = c * x0 * np.exp(-k * t)
    v = c * x
    q = 2 * k * mu / D**2 - 1

    pdf = np.zeros_like(x, dtype=float)
    mask = x >= 0

    pdf[mask] = (
        c
        * np.exp(-(u + v[mask]))
        * (v[mask] / u) ** (q / 2)
        * iv(q, 2 * np.sqrt(u * v[mask]))
    )

    return pdf

def h_I(y,D,theta):
    """Inverse of Lamperti transform in gamma model model"""
    if theta==1:
        return exp(y*D)
    else:
        dum  = (1.0 - theta)
        dum2 = 2.0 / dum
        return (y* dum * D /2.0)**dum2

def propagator_gamma_process_BCM(t0,tf, xf,x0, mu, k, D,theta,Nbridges=1000,dt_bridge=0.001):
    """Compute propagator of the 1D gamma process via importance sampling using bridges of Wiener process
    If theta=0, the process reduces to the DE (CIR) process and its propagator is computed exactly
    """
    tfmt0 = tf - t0

    if theta >= 2.0*k*mu/D**2:
        raise ValueError("The Feller condition is not satisfied: theta > 2*k*mu/D^2")
        
    if theta == 0:
        empirical_prop = propagator_DE_process(xf, tfmt0, x0, mu, k, D)
        return empirical_prop
    
    #Make Lamperti transformation of data to use bridges of Wiener process and compute Jacobian of transformation and Radon-Nikodym derivative between bridge and its undonditioned version (propagator)
    if theta==1:
        y0 = log(x0)  / D
        yf = log(xf)  / D
        J  = 1.0 / xf / D 
    else:
        dum  = (1.0 - theta)
        dum2 = dum / 2.0
        dum3 = -(1.0+theta) / 2.0
        y0 = 2.0*x0**dum2 / dum / D 
        yf = 2.0*xf**dum2 / dum / D 
        J  = xf**dum3 / D
    prop_WI = propagator_WI_process(yf, tfmt0, y0, D=1)  # Wiener propagator

    Ls = np.zeros(Nbridges)

    dum = (1.0 + theta)/2.0
    for i in range(Nbridges): #For every bridge, compute the Radon-Nykodym derivative between unconditioned processes
        ts_bridge, ys_bridge = Fill_gaps_with_Brownian_bridges(dt_bridge, tf, t0, y0, yf, D=1)

        # Transformed drift
        xs_bridge = h_I(ys_bridge,D,theta)
        if theta==1:
            Bs = xs_bridge * D
            Bs_der = D 
        else:
            Bs = xs_bridge**dum * D
            Bs_der = xs_bridge**(dum-1.0) * D * dum
        b  = - k * (xs_bridge - mu)*xs_bridge/ Bs - 0.5*Bs_der

        # Girsanov exponent components (left-point rule)
        dy = np.diff(ys_bridge)
        dt = np.diff(ts_bridge)

        state_integral  =  np.sum(b[:-1] * dy) #This could be computed exactly
        time_integral   =  np.sum(b[:-1]**2.0 * dt)
        # bridge_integral =  np.sum(b[:-1] * beta * dt)

        # log_weight = state_integral - 0.5 * time_integral - bridge_integral
        log_weight = state_integral - 0.5 * time_integral 
        weight = np.exp(log_weight)
        Ls[i] = weight

    empirical_prop = np.mean(Ls)*J* prop_WI
    Ls = Ls*J* prop_WI
    return empirical_prop

def Update_system_general_gamma_model(x0,t0,tf, dt_int,theta,mu,k,D):
    """Update system from x0 to x=x(tf) using Milstein method"""

    steps=int((tf-t0)/dt_int)
    sqdt = sqrt(dt_int)
    D2 = D*D
    dtheta = (theta+1.0)/2.0
    x = x0
    t = t0
    for ii in range(steps):
        t=t+dt_int
        u=np.random.normal()
        xth = x**(theta)
        milstein = D2*dtheta*xth*dt_int/2.0
        drift = k*(mu-x)*xth*dt_int 
        noise = D*(x**(dtheta))*sqdt
        xnew=x+drift+noise*u+milstein*(u*u-1.0)
        x = xnew
    return x
# %%
D = 0.1  # Diffusion coefficient
mu = 1.0  # mean
k = 1.0  # strength of the restoring force (inverse of correlation time)
tf = 1.0
t0 = 0.0
tfmt0 = tf - t0
x0 = mu
theta = 0.5  # Non-linearity parameter

#Classical MC
realiz = 10000
dt_int = tfmt0/100
theta = 0.5
xfs = np.zeros(realiz)
for ii in range(realiz):
    x = Update_system_general_gamma_model(x0,t0,tf, dt_int,theta,mu,k,D)
    xfs[ii]=x

#Bridge change of measure
N_bridges = 1000
x_targets = np.linspace(0.75,1.25,10)
props = np.zeros(len(x_targets))
for ii, xf in enumerate(x_targets):
    empirical_prop = propagator_gamma_process_BCM(t0,tf, xf,x0, mu, k, D,theta,N_bridges,dt_bridge=0.001)
    props[ii] = empirical_prop

Plot_bonito(xlabel=r"$ x$", ylabel=r"$ p(x,t|x_0)$", x_size=4, y_size=3)
bins = np.linspace(xfs.min(),xfs.max(),50)
plt.hist(xfs,bins=bins,density=True,label="Classic MC",alpha=0.4,color='dodgerblue')
plt.yscale('log')

plt.scatter(x_targets,props,color='red',label='Bridge estimation',s=25)
plt.legend(fontsize=10,frameon=False,loc="upper right")
plt.xlim(0.7,1.4)
plt.ylim(1e-4,1e3)
plt.savefig("figures/theta_"+str(theta)+"_propagator_comparison_MC_bridges.pdf",bbox_inches="tight",transparent=True)
plt.show();plt.close()


# %%
