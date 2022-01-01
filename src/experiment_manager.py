# Largely inspired by Raul's code
# https://github.com/RaulAstudillo06/BOFN/blob/b3be80821d16a8501d09c6141367ae5d2944a695/bofn/bofn_trial.py


from typing import Callable
from BO_trial import BO_trial
import os, sys


def experiment_manager(
    problem_name: str,
    algo: str,
    first_trial: int, 
    last_trial: int,
    n_init_evals: int,
    n_bo_iter: int,
    restart: bool,
    problem: Callable,
    active_input_indices: List[List[Optional[int]]],
    input_dim: int
) -> None:

    # Get script directory
    script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    results_folder = script_dir + "/results/" + problem + "/" + algo + "/"

    if not os.path.exists(results_folder) :
        os.makedirs(results_folder)
    if not os.path.exists(results_folder + "runtimes/"):
        os.makedirs(results_folder + "runtimes/")
    if not os.path.exists(results_folder + "X/"):
        os.makedirs(results_folder + "X/")
    if not os.path.exists(results_folder + "network_output_at_X/"):
        os.makedirs(results_folder + "network_output_at_X/")
    if not os.path.exists(results_folder + "objective_at_X/"):
        os.makedirs(results_folder + "objective_at_X/")

    for trial in range(first_trial, last_trial + 1):
        BO_trial(
            problem = problem,
            problem_name=problem_name,
            input_dim=input_dim,
            active_input_indices=active_input_indices,
            algo=algo,
            n_init_evals=n_init_evals,
            n_bo_iter=n_bo_iter,
            trial=trial,
            restart=restart,
        )
            