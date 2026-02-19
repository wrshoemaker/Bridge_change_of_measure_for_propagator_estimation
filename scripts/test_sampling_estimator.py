import sys
from scipy.optimize import curve_fit
from scipy.stats import gamma

import numpy as np
from numpy import exp as exp
from numpy import log as log
from numpy import sqrt as sqrt

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import pickle
import config

import simulation_utils
import utils


test_dict_path = '%stest_dict.pickle' % config.data_directory


# parameters of DE model
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

#n_reads_total_set = 10000
#n_reads_total = np.repeat(n_reads_total_set, n_samples)

# inference params
# number of draws from gamma
#n_gamma_rvs = 100
n_bridges = 1000


n_iter = 10
ep_all = np.logspace(-1, 1, num=10)

n_reads_total_set = [10000, 1000, 100]


def run_simulation():

    output_dict = {}
    for n_reads_total_ in n_reads_total_set:
        output_dict[n_reads_total_] = {}
        output_dict[n_reads_total_]['sim_results'] = {}
        for ep_D in ep_all:
            
            output_dict[n_reads_total_]['sim_results'][ep_D] = {}
            
            for ep_k in ep_all:
                output_dict[n_reads_total_]['sim_results'][ep_D][ep_k] = []


    for n_reads_total_ in n_reads_total_set:

        print(n_reads_total_)

        n_reads_total = np.repeat(n_reads_total_, n_samples)

        for n in range(n_iter):

            # simulate the data
            true_x = simulation_utils.simulate_bdm_trajectory_dornic(total_t, 1, k, mu, D, delta_t = delta_t)[1:,0]
            sampled_n = np.random.binomial(n_reads_total, true_x)

            # goal: minimize the negative log-likelihood 
            log_normalized_l = utils.sampling_propagator_gamma_process_BCM(time, sampled_n, n_reads_total, mu, D, k, theta, n_bridges=n_bridges)

            # go through range of options
            for ep_D_idx, ep_D in enumerate(ep_all):
                for ep_k_idx, ep_k in enumerate(ep_all):

                    log_normalized_l_ep = utils.sampling_propagator_gamma_process_BCM(time, sampled_n, n_reads_total, mu, ep_D*D, ep_k*k, theta, n_bridges=n_bridges)

                    print(log_normalized_l)
                    output_dict[n_reads_total_]['sim_results'][ep_D][ep_k].append(log_normalized_l - log_normalized_l_ep)


    sys.stderr.write("Saving dictionary...\n")
    with open(test_dict_path, 'wb') as outfile:
        pickle.dump(output_dict, outfile, protocol=pickle.HIGHEST_PROTOCOL)
    sys.stderr.write("Done!\n")



def plot_simulation(n_reads_total):

    test_dict = pickle.load(open(test_dict_path, "rb"))

    Z = np.empty((len(ep_all), len(ep_all)))

    for ep_D_idx, ep_D in enumerate(ep_all):
        for ep_k_idx, ep_k in enumerate(ep_all):

            delta_l = np.asarray(test_dict[n_reads_total]['sim_results'][ep_D][ep_k])
            delta_l = delta_l[np.isfinite(delta_l)]

            if len(delta_l) > 0:
                mean_delta_l = np.mean(delta_l)
            else:
                mean_delta_l = np.nan

            Z[ep_k_idx, ep_D_idx] = mean_delta_l


    #cmap = plt.cm.Blues.copy()
    cmap = plt.cm.RdBu.copy()
    cmap.set_bad(color="black")

    #print(np.nanmin(Z))
    #vabs = max(abs(np.nanmin(Z)), abs(np.nanmax(Z)))
    vabs = 5000

    norm = TwoSlopeNorm(vmin=-vabs, vcenter=0.0, vmax=vabs)

    fig, ax = plt.subplots()

    #im = ax.imshow(Z, origin="lower", aspect="equal", cmap=cmap, vmin=np.nanmin(Z), vmax=np.nanmax(Z))

    im = ax.imshow(Z, origin="lower", aspect="equal", cmap=cmap, norm=norm)
    #print(np.nanmin(Z), np.nanmax(Z))
    
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(r'$ \bar{\ell}_{\mathrm{false}} - \bar{\ell}_{\mathrm{true}} $')

    ax.set_xticks(np.arange(len(ep_all)))
    ax.set_yticks(np.arange(len(ep_all)))
    ax.set_xticklabels(np.round(ep_all, 2))
    ax.set_yticklabels(np.round(ep_all, 2))

    #ax.axhline(y=1)

    ax.set_title(r'$\mu \cdot N_{\mathrm{reads}} = $' + str(round(mu * n_reads_total)), fontsize=11)


    idx = np.nanargmin(Z)
    row, col = np.unravel_index(idx, Z.shape)
    ax.plot(col, row, 'ko', markersize=12)



    ax.set_xlabel(r'$\frac{k_{\mathrm{false}}}{k_{\mathrm{true}}}$', fontsize=12)
    ax.set_ylabel(r'$\frac{D_{\mathrm{false}}}{D_{\mathrm{true}}}$', fontsize=12)
    
    fig.subplots_adjust(hspace=0.25, wspace=0.25)
    fig_name = "%ssimulation_heatmap_%d.png" % (config.analysis_directory, n_reads_total)
    fig.savefig(fig_name, format='png', bbox_inches = "tight", pad_inches = 0.3, dpi = 600)
    plt.close()    


    #Z = np.array([[test_dict[n_reads_total]['sim_results'][i][j] for j in ep_all] for i in ep_all])
    #for ep_D_idx, ep_D in enumerate(ep_all):
    #    for ep_k_idx, ep_k in enumerate(ep_all):







if __name__ == "__main__":

    #run_simulation()

    for n_reads_total in n_reads_total_set:

        plot_simulation(n_reads_total)