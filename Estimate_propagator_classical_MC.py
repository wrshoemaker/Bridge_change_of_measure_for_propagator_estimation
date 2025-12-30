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
from scipy.special import iv  # modified 
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

# %%
# subroutine Update_system_dem_exact(next_t)
#         !Update system from x(t)=x to x(next_t), x and t are public variables
#         !This is the exact scheme for the DEM model
#         !K=0 will not work here, as the limit beta/(eb-1) (see below) is not properly computed (it is possible to use Update_system_dem_approx instead).
#         !See Dornic, I., Chaté, H., & Munoz, M. A. (2005). Integration of langevin equations with multiplicative noise and the viability of field theories for absorbing phase transitions. Physical Review Letters, 94(10), 18–21. https://doi.org/10.1103/PhysRevLett.94.100601
#         implicit none
#         double precision, intent(in) :: next_t
#         double precision :: lambda,nu,alpha,beta,D2,dt,eb
#         double precision :: dran_gamma
#         integer*8 :: steps,ii,n
#         integer*8 :: iran_poisson

#         dt = next_t-t
#         D2 = D*D
#         beta = -K
#         alpha = mu*K
#         eb = exp(beta*dt)
#         lambda = 2.0d0*beta/D2/(eb-1.0d0)
#         nu = 2.0d0*alpha/D2-1.0d0
#         n = iran_poisson(lambda*x*eb)
#         ! print *, "alpha gamma = ",n+nu+1.0d0,"n=",n,"alpha=",alpha,"beta=",beta,"lambda=",lambda
#         x = dran_gamma(n+nu+1.0d0)/lambda
#         t = next_t

#     end subroutine Update_system_dem_exact

#%% Draw trajectories general gamma model

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

def Estimate_propagator_standard_MC(x0,t0,tf,dt_int,realiz,x_target,dx,theta,mu,k,D):
    th = dx/2.0
    count = 0
    for ii in range(realiz):
        x =  Update_system_general_gamma_model(x0,t0,tf, dt_int,theta,mu,k,D)
        if (x>(x_target-th))and(x<=(x_target+th)):
            count+=1
    prop = count/realiz/dx
    return prop

D = 1.0  # Diffusion coefficient
mu = 1.0  # mean
k = 1.0  # strength of the restoring force (inverse of correlation time)

x0 = mu
tf = 1.0
t0 = 0.0
tfmt0 = tf - t0

xf = x0 + 0.1

target_prop = propagator_DE_process(xf, tfmt0, x0, mu, k, D)

realiz = 10000
dt_int = tfmt0/100
theta = 0
xfs = np.zeros(realiz)
for ii in range(realiz):
    x = Update_system_general_gamma_model(x0,t0,tf, dt_int,theta,mu,k,D)
    xfs[ii]=x
bins = np.linspace(xfs.min(),xfs.max(),50)
plt.hist(xfs,bins=bins,density=True)
true_props = propagator_DE_process(bins, tfmt0, x0, mu, k, D)
plt.plot(bins, true_props)
plt.show();plt.close()
# %%
realiz = 100000
Estimate_propagator_standard_MC(x0,t0,tf,dt_int,realiz,x0+0.1,0.001,theta,mu,k,D)
# %%
