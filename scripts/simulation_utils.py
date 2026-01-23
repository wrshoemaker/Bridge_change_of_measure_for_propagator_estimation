import os
import random
import copy
#import config
import sys
import numpy
import math

import scipy.stats as stats
from scipy.spatial import distance
from scipy.special import digamma, gamma, erf


def simulate_bdm_trajectory_dornic(t_total, n_reps, k, mu, D, delta_t = 1, x_0=None, epsilon=None):

    # Uses the convolution derived by Dornic et al to generate a trajctory of a migration-birth-drift SDE
    # sampling occurs *daily* like in the human gut timeseries
    # https://doi.org/10.1103/PhysRevLett.94.100601

    # delta_t = time between sampling events (1 day) *NOT* gridpoints
    # we are exactly simulating the FPE, not approximating it via Euler


    # expected value of the stationary distribution

    if t_total == None:
        delta_t = (1/k)*epsilon
        t_total = (1/k)/epsilon
        n_observations = math.ceil(t_total/delta_t)

    else:
        n_observations = t_total

        
    if x_0 == None:
        x_0 = mu

    # redefine variables for conveinance
    # first term, constant
    alpha = k*mu
    beta = -1*k
    sigma = D

    x_matrix = numpy.zeros(shape=(n_observations+1, n_reps))
    x_matrix[0,:] = x_0

    lambda_ = (2*beta)/((sigma**2) * (numpy.exp(beta*delta_t)-1) )
    mu_ = -1 + (2*alpha)/(sigma**2)
    
    for i in range(n_observations):
        poisson_rate_i = lambda_*x_matrix[i,:]*numpy.exp(beta*delta_t)
        poisson_rv_i = stats.poisson.rvs(poisson_rate_i)
        gamma_rate_i = mu_ + 1 + poisson_rv_i
        # rescale at each timepoint because this value is used as the initial condition for the next round of sampling.
        x_matrix[i+1,:] = (stats.gamma.rvs(gamma_rate_i)/lambda_)
        

    return x_matrix

