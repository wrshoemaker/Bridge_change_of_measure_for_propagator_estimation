import sys
import numpy as np
from numpy import exp as exp
from numpy import log as log
from numpy import sqrt as sqrt
from scipy.optimize import curve_fit
from scipy.special import gamma as gamma_func
from scipy.special import iv  # modified Bessel I_v
from scipy.stats import gamma
from scipy.special import logsumexp








# %% Functions
def gaussian(x, mean, variance):
    """Evaluate the Gaussian distribution at x with given mean and variance."""
    coeff = 1.0 / np.sqrt(2 * np.pi * variance)
    exponent = -((x - mean) ** 2) / (2 * variance)
    return coeff * np.exp(exponent)


def gamma_distribution(x, shape, scale):
    """Evaluate the Gamma distribution at x with given shape (k) and scale (theta)."""
    coeff = 1.0 / (gamma_func(shape) * scale**shape)
    return coeff * x**(shape - 1) * np.exp(-x / scale)


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


def evaluate_stationary_distribution(x, model, mu, K, D):
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




def simulate_ou_exact(mu, tau, x0, dt, n_steps, random_state=None):
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



def BDM_Lamperti(ts_data):
    arr = ts_data      
    # Check for negative values
    if np.any(arr < 0):
        raise ValueError("Input array contains negative values. Square root is not defined for negative numbers.")    
    result = 2 * np.sqrt(arr)
    return result   


def SLM_Lamperti(ts_data):
    arr = ts_data      
    # Check for negative values
    if np.any(arr < 0):
        raise ValueError("Input array contains negative values. Log is not defined for negative numbers.")    
    result = np.log(arr)
    return result


#def lamperti_transformation(x, lambda=0):

def log_propagators_target_process(log_rho_array, data, model='BDM'):
    valid_models = {"BDM", "SLM"}

    data = data[1:]

    if model not in valid_models:
        raise ValueError(f"Invalid model: {model}. Must be one of: {', '.join(valid_models)}")
    elif model == 'BDM':
        h_prime = 1 / np.sqrt(data)
        log_h_prime = np.log(h_prime)
        return log_h_prime + log_rho_array
    
    elif model == 'SLM':
        h_prime = 1 / data
        log_h_prime = np.log(h_prime)
        return log_h_prime + log_rho_array
    





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

def propagator_gamma_process_BCM(t0,tf, x0, xf, mu, k, D,theta,Nbridges=1000,dt_bridge=0.001):
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


def sampling_propagator_gamma_process_BCM(time, sampled_n, n_reads_total, mu, D, k, theta, n_bridges = 1000):

    alpha = 2*k/(D**2)
    beta = ((2*k*mu)/(D**2)) - theta
    n_samples = len(sampled_n)

    # get gamma parameters
    shape = beta + sampled_n
    # rate = 1/scale
    scale = 1/(alpha + n_reads_total)

    # get gamma rvs for each sample
    x_underline = [gamma.rvs(shape[_idx], scale=scale[_idx], size=n_bridges) for _idx in range(n_samples)]

    # calculate probability density for each sample
    pdf_val = [gamma.pdf(x_underline[_idx], a=shape[_idx], scale=scale[_idx]) for _idx in range(n_samples)]
    
    
    log_normalized_l = 0
    for t_idx in range(n_samples-1):
        
        t0 = time[t_idx]
        t1 = time[t_idx+1]
        x_underline_t0 = x_underline[t_idx]
        x_underline_t1 = x_underline[t_idx+1]

        empirical_prop_t = propagator_gamma_process_BCM(t0, t1, x_underline_t0 , x_underline_t1, mu, k, D, theta, n_bridges, dt_bridge=0.001)
        
        # propagator_gamma_process_BCM = ratio of probabilities

        # log likelihood for each iteration.
        # average over all iterations
        #log_normalized_l += np.mean(np.log(empirical_prop_t) - np.log(pdf_val[t_idx + 1]))

        # the loglikelihood is the log over the integral(i.e., sum)
        #log_normalized_l += np.log(np.mean(empirical_prop_t / pdf_val[t_idx + 1]))
        # faster
        log_r = np.log(empirical_prop_t) - np.log(pdf_val[t_idx + 1])
        log_normalized_l += logsumexp(log_r) - np.log(log_r.size)


        # log mean ratio is log marginal likelihood
        # mean log ratio = ELBO. Always biased downards, favors parameters that reduce varaince of ratio.
        # ==> why we're getting bettern LLs for small, wrong k?


    return log_normalized_l
