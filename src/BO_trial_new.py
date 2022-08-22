import os
import random
import sys
from typing import Callable, Dict, List, Tuple, Type
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
import torch
from botorch.acquisition import PosteriorMean
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.fit import fit_gpytorch_model
from botorch.models import SingleTaskGP
from botorch.models.cost import AffineFidelityCostModel
from botorch.models.transforms import Normalize, Standardize
from botorch.models.gpytorch import GPyTorchModel
from botorch.optim import optimize_acqf
from botorch.optim.initializers import gen_one_shot_kg_initial_conditions
from gpytorch.kernels import IndexKernel, MaternKernel, ProductKernel, ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
# TODO: import a list of kernels that I can call by string











"""
INPUT_RANGES = {"x": ["uniform", [0.0, 1.0]], "iteration_fidelity": ["uniform", [0.25, 1]]}

MULTIFIDELITY_PARAMS = {
    # which dimension of the input is for fidelity
    "fidelity_dim": 1,
    # key = fidelity_dim, value = target fidelity we want to optimize at
    "target_fidelities": {1: 1.0, 2: 0.0},
    # key = fidelity_dim, value = the increase in cost per one unit of fidelity
    "fidelity_weights": {1: 1.0},
    # fixed cost of evaluation; cost of evaluating at fidelity s = fixed_cost + fidelity_weights value * s \propto s
    "fixed_cost": 0.0,
}

tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cpu"),  # torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

# fidelity values at which to save intermediate outputs
CHECKPOINT_FIDELITIES = [0.25, 0.5, 1]
num_checkpoints = len(CHECKPOINT_FIDELITIES)


"""



def problem_evaluate(X: torch.Tensor, checkpoint_fidelities: List, **tkwargs) -> torch.Tensor:
    """
    Evaluates synthetic multi-task multi-fidelity problem

    Args:
        X: `num_samples x input_dim` tensor
        checkpoint_fidelities: list of numbers between 0 and 1, which are fidelity values where we save intermediate outputs

    Returns:
        `num_samples x 2` tensor of outputs, where column 0 and 1 contain values for task 0 and 1, respectively
    """

    input_shape = X.shape
    outputs = []
    max_iters = 20

    for i in range(input_shape[0]):
        x = X[i][0].item()
        s = X[i][1].item()
        y = 1 - (x - 0.5) ** 2

        checkpoints = []
        for frac_fid in checkpoint_fidelities:
            checkpoints.append(int(max_iters * s * frac_fid))

        for checkpoint in checkpoints:
            print("evaluating at checkpoint {}".format(checkpoint))
            if checkpoint < max_iters - 1:
                task_1 = torch.randn(1)
            else:
                task_1 = torch.tensor([y])
            task_0 = torch.tensor([y])

            outputs.append(torch.cat((task_0, task_1)))

    return torch.stack(outputs).to(**tkwargs)


def expand_intermediate_fidelities_wtask(
    X: torch.Tensor, checkpoint_fidelities: List
) -> torch.Tensor:
    """
    Expands the input tensor, whose last column is a task index,
    and second last column is a fidelity value,
    to include rows with the same design at intermediate fidelity values,
    determined by `checkpoint_fidelities`.

    Args:
        X: `trials x (input_dim + 2)` shape tensor,
            where the last column is a task index,
            and the second last column is a fidelity value
        checkpoint_fidelities: list of numbers indicating the fractions of the fidelity value where we want to make intermediate observations
    Returns:
        `(trials x len(checkpoint_fidelities)) x (input_dim + 1)` shape tensor

    Example: 1 trial, 2-dim design, 1-dim fidelity, checkpoint_fidelities = [0.25, 0.5, 1]
    Input: [[0.5, 0.5, 0.8, 0]]
    Output: [[0.5, 0.5, 0.2, 0], [0.5, 0.5, 0.4, 0], [0.5, 0.5, 0.8, 0]]
    """

    num_fidelities = len(checkpoint_fidelities)

    # last dim is task; this holds for processing candidates returned by the acquisition function
    fidelity_dim = -2

    designs = X[:, :fidelity_dim].repeat_interleave(num_fidelities, 0)

    fids = torch.kron(X[:, fidelity_dim], torch.tensor(checkpoint_fidelities)).unsqueeze(1)

    print("fidelity checkpoints", fids)

    # does not include the task column; task column to be added in process_multitask_data()
    return torch.cat((designs, fids), dim=1)


def expand_intermediate_fidelities_notask(
    X: torch.Tensor, checkpoint_fidelities: List
) -> torch.Tensor:
    """
    Expands the input tensor, whose last column is a fidelity value,
    to include rows with the same design at intermediate fidelity values,
    determined by `checkpoint_fidelities`.

    Args:
        X: `trials x (input_dim + 1)` shape tensor, where the last column is a fidelity value
        checkpoint_fidelities: list of numbers indicating the fractions of the fidelity value where we want to make intermediate observations
    Returns:
        `(trials x len(checkpoint_fidelities)) x (input_dim + 1)` shape tensor

    Example: 1 trial, 2-dim design, 1-dim fidelity, checkpoint_fidelities = [0.25, 0.5, 1]
    Input: [[0.5, 0.5, 0.8]]
    Output: [[0.5, 0.5, 0.2], [0.5, 0.5, 0.4], [0.5, 0.5, 0.8]]
    """
    num_fidelities = len(checkpoint_fidelities)

    fidelity_dim = -1

    designs = X[:, :fidelity_dim].repeat_interleave(num_fidelities, 0)

    fids = torch.kron(X[:, fidelity_dim], torch.tensor(checkpoint_fidelities)).unsqueeze(1)

    print("fidelity checkpoints", fids)
    result = torch.cat((designs, fids), dim=1)
    # does not include the task column; task column to be added in process_multitask_data()
    return result


def process_multitask_data(X: torch.Tensor, y: torch.Tensor, num_checkpoints: int, add_last_col_X: bool = False):
    """
    Transform (multi-dim input, multi-dim output) data into (multi-dim input, single-dim output) data

    Args:
        X: `(num_trials * num_checkpoints) x (input_dim + 1)`
                or `(num_trials * num_checkpoints) x (input_dim + 2)` shape tensor
        y: `(num_trials * num_checkpoints) x output_dim` shape tensor
        num_checkpoints:
        add_last_col_X:

    Returns:
        new_X: `(num_trials * num_checkpoints * output_dim) x (input_dim + 2)` shape tensor
        new_y: `(num_trials * num_checkpoints * output_dim) x 1` shape tensor
    """

    num_trials = y.shape[0] // num_checkpoints
    num_outputs = y.shape[-1]

    X_repeat = X.repeat_interleave(num_outputs, 0)
    task_idx_repeat = torch.arange(0, num_outputs).unsqueeze(1).repeat(num_checkpoints, 1).repeat(num_trials, 1)

    if add_last_col_X:
        new_X = torch.cat((X_repeat, task_idx_repeat), 1)
    else:
        new_X = torch.cat((X_repeat[:, :-1], task_idx_repeat), 1)

    new_y = y.flatten().unsqueeze(1)

    return new_X, new_y


def fit_GP_model(X, y, num_outputs, **tkwargs):
    """
    Fit a GP model based on given data

    Args:
        X:
        y:
        num_outputs:

    Returns:
        SingleTaskGP model

    """

    covar_module = ProductKernel(
        MaternKernel(active_dims=torch.arange(0, X.shape[-1] - 1)),
        IndexKernel(
            active_dims=torch.tensor([X.shape[-1] - 1]),
            num_tasks=num_outputs,
        ),
    )
    print(f"Fitting GP model on {X.shape} and {y.shape}.")
    model = SingleTaskGP(
        X,
        y,
        covar_module=covar_module,
        #input_transform=Normalize(d=X.shape[-1]),
        #outcome_transform=Standardize(m=y.shape[-1]),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)

    return model.to(**tkwargs)

def generate_initial_samples(n_samples: int, param_ranges: Dict, seed: int = None) -> torch.Tensor:
    """
    generate initial input from specified domain

    Args:
        n_samples:
        param_ranges:

    Returns:
        initial_X:
    """
    if seed is not None:
        torch.manual_seed(seed)

    initial_X = torch.Tensor()

    for k, ranges in param_ranges.items():
        if ranges[0] == "uniform":
            sample = torch.FloatTensor(n_samples, 1).uniform_(ranges[1][0], ranges[1][1])
            initial_X = torch.cat((initial_X, sample), dim=1)

        elif ranges[0] == "int":
            sample = torch.randint(ranges[1][0], ranges[1][1] + 1, (n_samples, 1))
            initial_X = torch.cat((initial_X, sample), dim=1)

        elif ranges[0] == "discrete":
            vals = ranges[1]
            sample = torch.Tensor(random.choices(vals, k=n_samples))
            initial_X = torch.cat((initial_X, torch.unsqueeze(sample, 1)), dim=1)

    print(f"initial_X = {initial_X}")
    return initial_X


def get_param_bounds(param_ranges: Dict) -> Tuple[torch.Tensor, List]:

    num_params = len(param_ranges)
    bounds = torch.empty(2, num_params)

    # also return the is_int feature to be passed into Matern kernel
    is_int = []

    for i, ranges in enumerate(param_ranges.values()):
        bounds[0,i] = min(ranges[1])
        bounds[1,i] = max(ranges[1])

        if ranges[0] in ['discrete', 'int']:
            is_int.append(True)
        else:
            is_int.append(False)

    return bounds.double(), is_int


def get_mfkg(
    model: Type[GPyTorchModel],
    bounds: torch.Tensor,
    cost_aware_utility: Type[InverseCostWeightedUtility],
    project: Callable,
    fidelity_dim: int,
    **tkwargs
):

    """
    Create qMultiFidelityKnowledgeGradient.

    Args:
        model:
        bounds:
        cost_aware_utility:
        project: function that maps input to one at the target fidelity
        fidelity_dim: dimension of the input vector that is the fidelity value

    Returns:
        qMultiFidelityKnowledgeGradient
    """

    # fix features to fidelity = 1, task = 0, since we care about the main task performance at the highest fidelity
    curr_val_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=fidelity_dim + 2,
        columns=[fidelity_dim, 2], # do not put negative indices here
        values=[1.0, 0.0],
    )
    _bounds = bounds[:, :-2]
    curr_val_acqf.to(**tkwargs)

    # get the largest mean (of the main task at highest fidelity) under the current posterior,
    # optimizing with respect to the designs only
    best_point, current_value = optimize_acqf(
        acq_function=curr_val_acqf,
        bounds=_bounds,
        q=1,
        num_restarts=10,
        raw_samples=1024,
        options={"batch_limit": 10, "maxiter": 200},
    )

    print("current max posterior mean before sampling new points: {}".format(current_value))

    # return the KG, the expected increase in best expected value conditioned on q more samples
    return qMultiFidelityKnowledgeGradient(
        model=model,
        num_fantasies=128,
        current_value=current_value,
        cost_aware_utility=cost_aware_utility,
        project=project,
    )


def optimize_acqf_and_suggest_new_pt(
    model: Type[GPyTorchModel],
    bounds: torch.Tensor,
    param_ranges: Dict,
    trial: int,
    is_int: List[bool] = None,
    cost_aware_utility: Type[InverseCostWeightedUtility] = None,
    project: Callable = None,
    fidelity_dim: int = None,
    **tkwargs
):

    acqf = get_mfkg(model, bounds, cost_aware_utility, project, fidelity_dim)

    current_max_posterior_mean = acqf.current_value

    # generate KG initial conditions with fidelity fixed to 1 and task fixed to 0 (if multi-task)
    # note that bounds is still the full set of bounds, including those that were fixed during get_mfkg()
    # that is, the candidates are of the form (design, fidelity, task), but we ignore the task
    X_init = gen_one_shot_kg_initial_conditions(
        acq_function=acqf,
        bounds=bounds,
        q=1,
        num_restarts=20,  # default value is 20
        raw_samples=1024,  # default value is 1024
        options={
            "num_inner_restarts": 20,
            "raw_inner_samples": 1024,  # default is 20 and 1024
            "batch_limit": 5,  # if this is not specified, raw_inner_samples posterior computations happen at the same time, leading to OOM
        },
    )
    candidates, acqf_val = optimize_acqf(
        acq_function=acqf,
        bounds=bounds,
        q=1,
        num_restarts=10,
        raw_samples=512,
        batch_initial_conditions=X_init,
        options={"batch_limit": 5, "maxiter": 200},
    )

    if len(acqf_val.size()) == 0:
        acqf_val = acqf_val.unsqueeze(0)

    for i in range(candidates.shape[-1]):
        if is_int[i]:
            candidates[..., i] = torch.round(candidates[..., i])

    print("optimize MultiFidelityKG, get candidates ", candidates, ", acqf_val ", acqf_val)

    # candidates has shape q x (input_dim + 2) -- both fidelity and task are included
    return candidates, acqf_val, current_max_posterior_mean



def plot_model(model, task=0.0):
    locs = torch.linspace(1.0, 5.0, 100).view(100, -1)
    test_X = torch.concat((locs, torch.tensor([[1.0, task]]).repeat(100, 1)), dim=-1).unsqueeze(-2)
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    means = model.posterior(test_X).mean.detach().numpy().flatten()
    sd = np.sqrt(model.posterior(test_X).variance.detach().numpy().flatten())
    ub = means + 1.96 * sd
    lb = means - 1.96 * sd
    ax.plot(locs, means)
    ax.fill_between(locs.squeeze(), lb, ub, alpha=0.2)
    plt.title(f"task = {task}")
    plt.show()


def run_BO(
    problem_evaluate: Callable,
    param_ranges: Dict,
    n_initial_pts: int,
    n_bo_iter: int,
    trial: int,
    multifidelity_params: Dict,
    checkpoint_fidelities: List,
):

    # Get script directory
    script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    results_folder = script_dir + "/"

    print("results folder", results_folder)

    # specify cost model and cost-scaled utility
    cost_model = AffineFidelityCostModel(
        fidelity_weights=multifidelity_params["fidelity_weights"], fixed_cost=multifidelity_params["fixed_cost"]
    )
    cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

    # specify which dim(s) of input are fidelities, what the target fidelities are
    fidelity_dim = multifidelity_params["fidelity_dim"]
    target_fidelities = multifidelity_params["target_fidelities"]
    def project(X):
        return project_to_target_fidelity(X=X, target_fidelities=target_fidelities)

    # X has shape (n_initial_pts) * (input_dim + 1)
    X = generate_initial_samples(n_initial_pts, param_ranges, trial)
    # y has shape (num_trials * num_checkpoints) x num_outputs
    y = problem_evaluate(X, checkpoint_fidelities = checkpoint_fidelities)
    num_outputs = y.shape[-1]

    # this expands X to incorporate a column for fidelity, with a row for each fidelity checkpoint
    # expanded X has shape (n_initial_pts * num_checkpoints) * (input_dim + 1)
    X = expand_intermediate_fidelities_notask(X, checkpoint_fidelities)

    init_batch_id = 1
    log_best_so_far = []
    cum_costs = [
        cost_model(X[num_checkpoints - 1 :: num_checkpoints]).sum().item()
    ]  # evaluate cost of sampling initial X
    param_ranges["task_idx"] = ["int", [0, num_outputs - 1]]
    # initial X does not contain the task column, so set add_last_col_X to True
    X, y = process_multitask_data(X, y, num_checkpoints, add_last_col_X=True)
    max_posterior_mean = [y[::num_outputs][(num_checkpoints - 1) :: num_checkpoints].max().item()]

    bounds, is_int = get_param_bounds(param_ranges)

    # dictionary for logging data
    out = {}
    sampled_fidelities = []
    acqf_vals = torch.Tensor().cpu()

    for iter in range(init_batch_id, n_bo_iter + 1):

        print("starting BO iteration ", iter)
        print(X)
        print(y)
        model = fit_GP_model(X, y, num_outputs, **tkwargs)
        plot_model(model, task=0.0)
        plot_model(model, task=1.0)
        new_pt, acqf_val, current_max_posterior_mean = optimize_acqf_and_suggest_new_pt(
            model, bounds, param_ranges, trial, is_int, cost_aware_utility, project, fidelity_dim
        )

        max_posterior_mean.append(current_max_posterior_mean)
        sampled_fidelities.append(new_pt[0][fidelity_dim].item())

        new_y = problem_evaluate(new_pt, checkpoint_fidelities = checkpoint_fidelities)
        print("evaluation of newly sampled point {}".format(new_y))

        # last dimension of new_pt is the task column
        new_pt = expand_intermediate_fidelities_wtask(new_pt, checkpoint_fidelities, last_dim_is_task=True)
        new_pt, new_y = process_multitask_data(new_pt, new_y, num_checkpoints, add_last_col_X=True)

        acqf_vals = torch.cat((acqf_vals, acqf_val))

        if cost_model is not None:
            cum_costs.append(cum_costs[-1] + cost_model(new_pt)[-1].item())

        print("cumulative cost", cum_costs)

        X = torch.cat((X, new_pt), dim=0)
        y = torch.cat((y, new_y), dim=0)

        # only log the best value of task 0 at the highest fidelity sampled, not the intermediate ones
        log_best_so_far = (
            y[::num_outputs][(num_checkpoints - 1) :: num_checkpoints].cummax(0).values[n_initial_pts - 1 :]
        )

        print("Finished iteration {}, best value so far is {}".format(iter, log_best_so_far[-1].item()))

        # save results in dictionary
        out["X"] = X
        out["Y"] = y
        out["best_so_far"] = log_best_so_far
        out["acqf_vals"] = acqf_vals
        out["cum_costs"] = cum_costs

        torch.save(out, results_folder + "trial_" + str(trial))  # TODO: make kernel name an input

        title = "best objective value"

        # plot_progress([title, log_best_so_far], cum_costs, results_folder, trial, max_posterior_mean=max_posterior_mean)
        # plot_acqf_vals_and_fidelities(acqf_vals, sampled_fidelities, results_folder, trial)

    
    return out


if __name__ == "__main__":

    INPUT_RANGES = {"x": ["uniform", [0.0, 1.0]], "iteration_fidelity": ["uniform", [0.25, 1]]}

    MULTIFIDELITY_PARAMS = {
        # which dimension of the input is for fidelity
        "fidelity_dim": 1,
        # key = fidelity_dim, value = target fidelity we want to optimize at
        "target_fidelities": {1: 1.0, 2: 0.0},
        # key = fidelity_dim, value = the increase in cost per one unit of fidelity
        "fidelity_weights": {1: 1.0},
        # fixed cost of evaluation; cost of evaluating at fidelity s = fixed_cost + fidelity_weights value * s \propto s
        "fixed_cost": 0.0,
    }

    tkwargs = {
        "dtype": torch.double,
        "device": torch.device("cpu"),  # torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    # fidelity values at which to save intermediate outputs
    CHECKPOINT_FIDELITIES = [0.25, 0.5, 1]
    num_checkpoints = len(CHECKPOINT_FIDELITIES)

    output = run_BO(
        problem_evaluate=problem_evaluate,
        param_ranges=deepcopy(INPUT_RANGES),
        n_initial_pts=5,
        n_bo_iter=5,
        trial=0,
        multifidelity_params=MULTIFIDELITY_PARAMS,
        checkpoint_fidelities=CHECKPOINT_FIDELITIES,
    )