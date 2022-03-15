from functools import partial
from synthetic_test import problem_evaluate, Input_RANGES, MultiFidelity_PARAMS
import sys
import torch
sys.path.append('..')
from src.experiment_manager import experiment_manager

problem_name = 'synthetic_test'

first_trial = last_trial = int(sys.argv[1])
algo = sys.argv[2] # could be 'EI', 'KG'
is_multitask = bool(int(sys.argv[3])) # 1 if multi-task; 0 if single-task

problem_evaluate_fixed_outputs = partial(problem_evaluate, is_multitask = is_multitask)
if is_multitask:
    problem_name += '_MT'

if torch.cuda.is_available():
    torch.cuda.set_device("cuda:0")
tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}

experiment_manager(
    problem = problem_evaluate_fixed_outputs,
    problem_name = problem_name,    
    param_ranges = Input_RANGES,
    algo = algo,
    first_trial = first_trial,
    last_trial = last_trial,
    n_initial_pts = 5,
    n_bo_iter = 50,
    restart = False,
    verbose = True,
    is_multitask = is_multitask,
    use_additive_kernel = None,
    multifidelity_params = MultiFidelity_PARAMS,
    **tkwargs
)
