from functools import partial
from Cifar10_Cnn import HPs_to_VARY, problem_evaluate
from experiment_manager import experiment_manager

# Basic skeleton for running an experiment

problem_name = 'Cifar10_Cnn'

return_metrics = {'val_acc': 'max'}

problem_evaluate_fixed_metrics = partial(problem_evaluate, return_metrics = return_metrics)

algo = 'EI'

experiment_manager(
    problem = problem_evaluate_fixed_metrics,
    problem_name = problem_name,    
    input_dim = 5,
    param_ranges=HPs_to_VARY,
    algo = algo,
    first_trial = 1,
    last_trial = 1,
    n_initial_pts = 10,
    n_bo_iter = 50,
    restart = True,
)