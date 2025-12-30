# Bridge Change of Measure for Propagator Estimation

This repository contains research code and numerical experiments for propagator (transition density) estimation of stochastic processes using change-of-measure techniques and bridge constructions. The focus is on comparing "gamma models" i.e., 1d SDEs that share the same gamma stationary distribution

$$dX_t = k X_t^{\theta} (\mu - X_t)\,dt + D X_t^{\frac{\theta+1}{2}}\,dW_t.$$

When the Feller condition is fulfilled,

$$ \theta < 2k \mu / D^2, $$

Then the stationary distribution is a gamma,

$$ \rho(x) = \frac{\left(\frac{k}{2 D^2}\right)^{\frac{2k\mu}{D^2} - \theta}}{\Gamma\left(\frac{2k\mu}{D^2} - \theta\right)} x^{\frac{2k\mu}{D^2} - \theta - 1} e^{-\frac{2kx}{D^2}}. $$
