# TODO: goal is to write a script that automatically goes to the directory of interest 
# and generates figures for the BO training in progress

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse


def plot_progress(metric, cum_costs, results_folder, trial, max_posterior_mean = None):
    # metrics: dictionary of {metric_name: list of metric values}

    k, v = metric
    fig, ax1 = plt.subplots()

    a1, = ax1.plot(cum_costs, v, color = 'g', label = 'best objective', marker = 'o')
    ax1.set_xlabel('resources consumed')
    ax1.set_ylabel('best objective value achieved')

    if max_posterior_mean is not None:
        a2, = ax1.plot(cum_costs, max_posterior_mean, color = 'm', label = 'max posterior mean', marker = 'o')
        ax1.set_ylabel('maximum posterior mean')
    
    plt.title(k + '\n final objective value achieved: {:.2f}, resources consumed: {:.2f}'.format(v[-1].item(), cum_costs[-1]))
    p = [a1, a2]
    ax1.legend(p, [p_.get_label() for p_ in p], loc= 'upper left')

    fig.savefig(results_folder + 'visualization_trial_' + str(trial))


def plot_acqf_vals_and_fidelities(acqf_vals, sampled_fidelites, results_folder, trial):

    fig, ax1 = plt.subplots()

    a1, = ax1.plot(acqf_vals.numpy(), color = 'g', label = 'acquisition function value', marker = 'o')
    ax1.set_xlabel('Iterations of BO')
    ax1.set_ylabel('Acquisition function value')
    ax1.axhline(c = 'grey', linestyle = '--')
    
    ax2 = ax1.twinx()
    a2, = ax2.plot(sampled_fidelites, color = 'b', label = 'sampled fidelities', marker = 'o')
    ax2.set_ylabel('sampled fidelities')

    ax1.set_title('Acquisition function values and sampled fidelities')
    p = [a1, a2]
    ax1.legend(p, [p_.get_label() for p_ in p], loc= 'upper left')

    fig.savefig(results_folder + 'acqf_vals_trial_' + str(trial))









# DEPRECATED

def plot_monitoring_figs(problem, algo, trial):

    #objective_vals = np.squeeze(np.load('/home/yz685/SGD_diagnostics/experiment_problems/results/' + \
    #    problem + '/' + algo + '/objective_at_X/objective_at_X_' + trial + '.npy') , 1)

    log_best_so_far = np.load('/home/yz685/SGD_diagnostics/experiment_problems/results/' + \
        problem + '/' + algo + '/log_best_so_far_' + trial + '.npy') 

    acqf_vals = np.load('/home/yz685/SGD_diagnostics/experiment_problems/results/' + \
        problem + '/' + algo + '/acqf_vals_' + trial + '.npy') 

    print(log_best_so_far)
    print(acqf_vals)

    #plt.figure(figsize = (, 5))

    #ax1 = plt.subplot(311)
    #ax1.plot(objective_vals)
    #ax1.set_title('Objective values over BO iterations (including initial samples)')

    ax2 = plt.subplot(312)
    ax2.plot(log_best_so_far)
    ax2.set_title('Best value so far over BO iterations')

    ax3 = plt.subplot(313)
    ax3.plot(acqf_vals)
    ax3.set_title('Acquisition function values')

    plt.tight_layout(pad=1.0)


    plt.savefig('/home/yz685/SGD_diagnostics/experiment_problems/results/' + \
        problem + '/' + algo + '/' + 'visualization_trial_' + trial + '.pdf')