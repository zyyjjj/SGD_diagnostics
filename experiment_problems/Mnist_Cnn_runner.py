from functools import partial
from Mnist_Cnn import HPs_to_VARY, problem_evaluate
import sys
sys.path.append('..')
from src.experiment_manager import experiment_manager

# Basic skeleton for running an experiment

problem_name = 'Mnist_Cnn'

return_metrics = {'val_acc': 'max'}

problem_evaluate_fixed_metrics = partial(problem_evaluate, return_metrics = return_metrics)

algo = 'EI'

if len(sys.argv) == 3:
    first_trial = int(sys.argv[1])
    last_trial =  int(sys.argv[2])
elif len(sys.argv) == 2:
    first_trial = int(sys.argv[1])
    last_trial =  int(sys.argv[1])

experiment_manager(
    problem = problem_evaluate_fixed_metrics,
    problem_name = problem_name,    
    param_ranges=HPs_to_VARY,
    algo = algo,
    first_trial = first_trial,
    last_trial = last_trial,
    n_initial_pts = 10,
    n_bo_iter = 50,
    restart = True,
    verbose = True
)