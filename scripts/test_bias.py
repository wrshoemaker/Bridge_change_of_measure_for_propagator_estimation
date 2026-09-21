
import numpy as np
import utils
import simulation_utils

def compare_parameter_grid_with_mc_repeats(
    time,
    total_t,
    delta_t,
    n_samples,
    n_reads_total,
    mu,
    D_true,
    k_true,
    theta,
    ep_all,
    n_datasets=20,
    n_mc_repeats=10,
    n_bridges=1000,
    simulate_fn=None,
    ll_fn=None):
    
    # Diagnose whether wrong optima come from finite-data effects or MC likelihood noise/bias

    if simulate_fn is None:
        raise ValueError("simulate_fn must be provided")
    if ll_fn is None:
        raise ValueError("ll_fn must be provided")

    param_grid = [(ep_D * D_true, ep_k * k_true, ep_D, ep_k) for ep_D in ep_all for ep_k in ep_all]

    dataset_results = []
    true_wins = []

    n_reads_total_vec = np.repeat(n_reads_total, n_samples)

    for dataset_idx in range(n_datasets):

        print(dataset_idx)
        
        #simulate one dataset from the true parameters
        true_x = simulate_fn(total_t, 1, k_true, mu, D_true, delta_t=delta_t)[1:, 0]
        sampled_n = np.random.binomial(n_reads_total_vec, true_x)

        # evaluate each parameter pair multiple times on same dataset
        ll_by_param = {}

        for D_test, k_test, ep_D, ep_k in param_grid:
            ll_reps = []

            for rep in range(n_mc_repeats):
                np.random.seed(10_000 * dataset_idx + rep)

                ll_val = utils.sampling_propagator_gamma_process_BCM(time, sampled_n, n_reads_total_vec, mu, D_test, k_test, theta, n_bridges=n_bridges)
                ll_reps.append(ll_val)

            ll_reps = np.asarray(ll_reps, dtype=float)

            ll_by_param[(ep_D, ep_k)] = {
                "D": D_test,
                "k": k_test,
                "ll_reps": ll_reps,
                "ll_mean": np.mean(ll_reps),
                "ll_std": np.std(ll_reps, ddof=1) if len(ll_reps) > 1 else 0.0}

        # locate true parameter
        true_key = None
        for ep_D in ep_all:
            for ep_k in ep_all:
                if np.isclose(ep_D, 1.0) and np.isclose(ep_k, 1.0):
                    true_key = (ep_D, ep_k)
                    break
            if true_key is not None:
                break

        if true_key is None:
            raise ValueError("ep_all must contain 1.0 so the true parameter is in the grid.")

        # best parameter by mean estimated LL
        best_key = max(ll_by_param, key=lambda key: ll_by_param[key]["ll_mean"])

        true_ll_mean = ll_by_param[true_key]["ll_mean"]
        best_ll_mean = ll_by_param[best_key]["ll_mean"]

        dataset_results.append({
            "dataset_idx": dataset_idx,
            "sampled_n": sampled_n.copy(),
            "true_x": true_x.copy(),
            "ll_by_param": ll_by_param,
            "true_key": true_key,
            "best_key": best_key,
            "true_ll_mean": true_ll_mean,
            "best_ll_mean": best_ll_mean,
            "true_minus_best": true_ll_mean - best_ll_mean,
        })

        true_wins.append(best_key == true_key)

    summary = {
        "n_datasets": n_datasets,
        "n_mc_repeats": n_mc_repeats,
        "true_win_rate": np.mean(true_wins),
        "dataset_results": dataset_results,
    }


    # aggregate over datasets for each parameter
    param_summary = {}
    for ep_D in ep_all:
        for ep_k in ep_all:
            ll_means = np.array([
                ds["ll_by_param"][(ep_D, ep_k)]["ll_mean"]
                for ds in dataset_results
            ], dtype=float)

            ll_stds = np.array([
                ds["ll_by_param"][(ep_D, ep_k)]["ll_std"]
                for ds in dataset_results
            ], dtype=float)

            param_summary[(ep_D, ep_k)] = {
                "mean_over_datasets": np.mean(ll_means),
                "std_over_datasets": np.std(ll_means, ddof=1) if len(ll_means) > 1 else 0.0,
                "mean_mc_std": np.mean(ll_stds),
            }

    summary["param_summary"] = param_summary
    
    return summary


mu = 0.01
D = 0.2
k = 3
theta = 0

# composite parameters
alpha = 2*k/(D**2)
beta = ((2*k*mu)/(D**2)) - theta

# DE simulation params 
delta_t = 1
total_t = 1000
n_samples = int(total_t/delta_t)
time = np.arange(0, total_t, step=delta_t)
n_pairs = len(time) - 1
#ep_all = np.logspace(-1, 1, num=10)
#ep_all = np.logspace(-1, 1, num=10)
ep_all = [0.1, 1, 10]
#ep_all = []
n_bridges = 1000
#n_reads_total_set = 1000
#n_reads_total_set = [10000, 1000, 100]

results = compare_parameter_grid_with_mc_repeats(
    time=time,
    total_t=total_t,
    delta_t=delta_t,
    n_samples=n_samples,
    n_reads_total=1000,
    mu=mu,
    D_true=D,
    k_true=k,
    theta=theta,
    ep_all=ep_all,
    n_datasets=5,
    n_mc_repeats=3,
    n_bridges=n_bridges,
    simulate_fn=simulation_utils.simulate_bdm_trajectory_dornic,
    ll_fn=utils.sampling_propagator_gamma_process_BCM)



for key, val in results["param_summary"].items():
    print(
        key,
        "mean LL over datasets =", val["mean_over_datasets"],
        "mean MC std =", val["mean_mc_std"],
    )
