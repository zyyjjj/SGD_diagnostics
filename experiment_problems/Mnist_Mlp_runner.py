from functools import partial
from Mnist_Mlp import HPs_to_VARY, MultiFidelity_PARAMS, problem_evaluate
import sys
import argparse
import torch
sys.path.append('..')
from src.experiment_manager import experiment_manager
import pdb

# Basic skeleton for running an experiment

problem_name = 'Mnist_Mlp'

return_metrics = {
    'val_acc': 'max',
    # 'norm_batch_grad': 'last',
    # 'norm_aux_batch_grad': 'last',
    # 'diff_sq_norm_main_aux_batch_grads': 'last',
    'running_avg_of_cosine_sim_batch_grad': 'last',
    # 'norm_change_batch_grad': 'last',
    'denoise_signal_1': 'last',
    'norm_of_running_avg_of_batch_grad': 'last',
    'epoch_accum_grad': 'last',
    'val_loss_improvement': 'last'
}
# TODO: consider pruning the set of return metrics, leaving fewer than right now


first_trial = last_trial = int(sys.argv[1])

# if len(sys.argv) == 3:
#     first_trial = int(sys.argv[1])
#     last_trial =  int(sys.argv[2])
# elif len(sys.argv) == 2:
#     first_trial = int(sys.argv[1])
#     last_trial =  int(sys.argv[1])

algo = sys.argv[-2] # could be 'EI', 'KG'
is_multitask = bool(int(sys.argv[-1])) # 1 if multi-task; 0 if single-task
DEBUG = False

if not is_multitask:
    # only return the one main metric
    key0, val0 = list(return_metrics.items())[0]
    problem_evaluate_fixed_metrics = partial(problem_evaluate, return_metrics = {key0: val0}, designs = HPs_to_VARY, debug = DEBUG)
else:
    problem_evaluate_fixed_metrics = partial(problem_evaluate, return_metrics = return_metrics, designs = HPs_to_VARY, debug = DEBUG)
    problem_name += '_MT'

if torch.cuda.is_available():
    torch.cuda.set_device("cuda:0")
tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}

experiment_manager(
    problem = problem_evaluate_fixed_metrics,
    problem_name = problem_name,    
    param_ranges = HPs_to_VARY,
    algo = algo,
    first_trial = first_trial,
    last_trial = last_trial,
    n_initial_pts = 1, # TODO: change to 10 after debugging
    n_bo_iter = 50,
    restart = False, # TODO: change to True after debugging 
    verbose = True,
    is_multitask = is_multitask,
    multifidelity_params = MultiFidelity_PARAMS,
    **tkwargs
)

# pdb.pm()