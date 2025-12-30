# Bridge Change of Measure for Propagator Estimation

This repository contains research code and numerical experiments for **propagator (transition density) estimation** of stochastic processes using **change-of-measure techniques** and **bridge constructions**.  

The focus is on comparing *gamma models*, i.e. one-dimensional stochastic differential equations (SDEs) that share the same **Gamma stationary distribution**:

$$
dX_t = k X_t^{\theta} (\mu - X_t)\,dt + D X_t^{\frac{\theta+1}{2}}\,dW_t.
$$

When the **Feller condition** is fulfilled,

$$
\theta < \frac{2k\mu}{D^2},
$$

the stationary distribution is a Gamma distribution given by

$$
\rho(x) =
\frac{\left(\frac{k}{2D^2}\right)^{\frac{2k\mu}{D^2} - \theta}}
{\Gamma\!\left(\frac{2k\mu}{D^2} - \theta\right)}
x^{\frac{2k\mu}{D^2} - \theta - 1}
e^{-\frac{2kx}{D^2}}.
$$

## Instructions
In order to run the .py codes, a folder named "figures" has to be created in order to store outcomes.

## License

This project is shared for **academic and research purposes**. It is free to use, redistribute, modify, and share for research purposes, provided that proper credit is given to the authors through citation of:

Aguilar, Javier, Miguel A. Muñoz, and Sandro Azaele.  
*The Limits of Inference in Complex Systems: When Stochastic Models Become Indistinguishable.*  
arXiv preprint arXiv:2509.24977 (2025).
