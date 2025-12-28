# %% Draw bridges, evaluate Radon-Nykodim derivatives and compute empirical propagator

Nbridges = 1000
dt_bridge = 0.01

tfmt0 = tf - t0  # make sure this is defined

Ls = np.zeros(Nbridges)

for i in range(Nbridges):
    ts_bridge, xs_bridge = Fill_gaps_with_Brownian_bridges(dt_bridge, tf, t0, yo, yf,D)

    # OU drift
    b = -k * (xs_bridge - mu)

    # Girsanov exponent components (left-point rule)
    dx = np.diff(xs_bridge)
    dt = np.diff(ts_bridge)

    state_integral = (1.0 / D**2) * np.sum(b[:-1] * dx) #This could be computed exactly
    time_integral  = (1.0 / D**2) * np.sum(b[:-1]**2 * dt)

    log_weight = state_integral - 0.5 * time_integral
    weight = np.exp(log_weight)

    prop_WI = propagator_WI_process(yf, tfmt0, yo, D)  # Wiener propagator
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
