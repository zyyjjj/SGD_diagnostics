# TODO: goal is to write a script that automatically goes to the directory of interest 
# and generates figures for the BO training in progress

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse



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

"""

def parse_hp_args():
    parser = argparse.ArgumentParser(description='specify the experiment and trial to monitor')
    parser.add_argument('-p', '--problem', help='experiment problem')
    parser.add_argument('-a', '--algo', help = 'algorithm')
    parser.add_argument('-t', '--trial', help = 'trial number')

    args = parser.parse_args()

    return args
"""

if __name__ == '__main__':


    plot_monitoring_figs('Mnist_Mlp', 'EI', '1')

    """
    args = parse_hp_args()


    objective_vals = np.squeeze(np.load('/home/yz685/SGD_diagnostics/experiment_problems/results/' + \
        args.problem + '/' + args.algo + '/objective_at_X/objective_at_X_' + args.trial + '.npy') , 1)

    log_best_so_far = np.load('/home/yz685/SGD_diagnostics/experiment_problems/results/' + \
        args.problem + '/' + args.algo + '/log_best_so_far_' + args.trial + '.npy') 

    acqf_vals = np.load('/home/yz685/SGD_diagnostics/experiment_problems/results/' + \
        args.problem + '/' + args.algo + '/acqf_vals_' + args.trial + '.npy') 


    #plt.figure(figsize = (, 5))

    ax1 = plt.subplot(311)
    ax1.plot(objective_vals)
    ax1.set_title('Objective values over BO iterations (including initial samples)')

    ax2 = plt.subplot(312)
    ax2.plot(log_best_so_far)
    ax2.set_title('Best value so far over BO iterations')

    ax3 = plt.subplot(313)
    ax3.plot(acqf_vals)
    ax3.set_title('Acquisition function values')

    plt.tight_layout(pad=1.0)


    plt.savefig('/home/yz685/SGD_diagnostics/experiment_problems/results/' + \
        args.problem + '/' + args.algo + '/' + 'visualization_trial_' + args.trial + '.pdf')
"""