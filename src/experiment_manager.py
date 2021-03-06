# Largely inspired by Raul's code
# https://github.com/RaulAstudillo06/BOFN/blob/b3be80821d16a8501d09c6141367ae5d2944a695/bofn/bofn_trial.py


from typing import Callable
from .BO_trial import BO_trial
import os, sys

def experiment_manager(
    problem: Callable,
    problem_name: str,
    # input_dim: int,
    param_ranges: dict,
    algo: str,
    first_trial: int, 
    last_trial: int,
    n_initial_pts: int,
    n_bo_iter: int,
    restart: bool,
    verbose: bool,
    is_multitask: bool,
    kernel_name = None,
    multifidelity_params = None,
    checkpoint_fidelities = None,
    **tkwargs
) -> None:

    # Get script directory
    script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    results_folder = script_dir + "/results/" + problem_name + "/" + algo + "/"

    print('working on problem {}, is_multitask = {}'.format(problem_name, is_multitask))

    if not os.path.exists(results_folder) :
        os.makedirs(results_folder)
    if not os.path.exists(results_folder + "runtimes/"):
        os.makedirs(results_folder + "runtimes/")
    if not os.path.exists(results_folder + "X/"):
        os.makedirs(results_folder + "X/")
    if not os.path.exists(results_folder + "output_at_X/"):
        os.makedirs(results_folder + "output_at_X/")

    for trial in range(first_trial, last_trial + 1):
        BO_trial(
            problem_evaluate = problem,
            problem_name=problem_name,
            # input_dim=input_dim,
            param_ranges=param_ranges,
            algo=algo,
            n_initial_pts=n_initial_pts,
            n_bo_iter=n_bo_iter,
            trial=trial,
            restart=restart,
            verbose = verbose,
            is_multitask = is_multitask,
            kernel_name = kernel_name,
            multifidelity_params = multifidelity_params,
            checkpoint_fidelities = checkpoint_fidelities,
            **tkwargs
        )