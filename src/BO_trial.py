import torch
import torch.nn as nn
import torch.nn.functional as F
#import torch.profiler
from torch.utils.data import DataLoader, random_split
from torch.optim import SGD, Adagrad, Adam
import torchvision
import torchvision.transforms as transforms
import neptune.new as neptune
import random
import numpy as np
import pdb, time, argparse, itertools, copy
import sys, os
sys.path.append('..')
from utils.parse_hp_args import parse_hp_args
from utils.train_nn import fit, accuracy
from utils.callback import *
#from experiment_problems import problem_evaluate

from botorch.models import SingleTaskGP, FixedNoiseGP, ModelListGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.optim import optimize_acqf, optimize_acqf_mixed
from botorch.utils import standardize

from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, MaternKernel
from gpytorch.priors.torch_priors import GammaPrior


from models.ModifiedMaternKernel import ModifiedMaternKernel

def BO_trial(
        problem_evaluate: Callable,
        problem_name: str,
        input_dim: int,
        param_ranges: dict,
        algo: str,
        n_initial_pts: int,
        n_bo_iter: int,
        trial: int,
        restart: bool
        ):

    X = None
    y = []

    bounds, is_int = get_param_bounds(param_ranges)

    if restart:
        # check if there is saved data available
        try: 
            # get saaved data
            
            # init_batch_id = len(hist_best_obs_vals)
            pass
        except:
            # generate initial data
            # TODO: check: there probably will be tensor shape issues here
            X = generate_initial_samples(n_initial_pts, param_ranges, trial)
            y = problem_evaluate(X)
            init_batch_id = 1

    # fit initial model
    model, mll = fit_GP_model(X, y, is_int)    

    best_all_trials = []

    for iter in range(init_batch_id, n_bo_iter+1):

        best = np.inf # or soemthing else
        best_log = []
        fit_gpytorch_model(mll)

        # define acqf
        if algo == 'EI':
            acqf = ExpectedImprovement(model, best)

        new_pt = suggest_new_pt(algo, X, y, bounds)
        new_y = problem_evaluate(new_pt)

        X = torch.cat([X, new_pt])
        y = torch.cat([y, new_y])

        # log the best-performing ones so far (including random)
        best_log.append(new_y)

        # re-fit model
        model, mll = fit_GP_model(X, y)    

        # if verbose: print stuff to STDOUT
        # also, save the newest X and y
        # update init_batch_id to track how many iters so far

    best_all_trials.append(best_log)


def fit_GP_model(X, y, is_int):

    _, aug_batch_shape = get_batch_dimensions(X,y)

    # the modified matern kernel rounds the values for integer variables 
    # before computing the kernel output
    covar_module = ScaleKernel(
        ModifiedMaternKernel(
            is_int,
            nu = 2.5,
            ard_num_dims = X.shape[-1],
            batch_shape = aug_batch_shape,
            lengthscale_prior = GammaPrior(3.0, 6.0),
        ),
        batch_shape = aug_batch_shape,
        outputscale_prior=GammaPrior(2.0, 0.15)
    )

    model = SingleTaskGP(X, y, covar_module)

    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    # TODO: figure this out
        # load state dict if it is passed
        # if state_dict is not None:
        # model.load_state_dict(state_dict)
 
    return model, mll


def suggest_new_pt(algo, X, y, bounds):
    if algo == 'EI':
        # fit GP model based on X, y
        model, _ = fit_GP_model(X, y)
        best = y.max().item()
        # define acq function
        acqf = ExpectedImprovement(model, best)

    # for integer ranges, rather than fixed_features_list,
    # maybe a better way is to treat it as a real number between 0 and 1
    # and round to the nearest integer

    candidates, _ = optimize_acqf(
        acq_function = acqf,
        bounds = bounds,
        q = 1, # TODO: figure out the following three
        num_restarts = 10,
        raw_samples = 10
    )

    return candidates


# This can go to utils/

def generate_initial_samples(n_samples, param_ranges, seed):

    if seed is not None:
        # TODO: set seed the righ way
        pass

    initial_X = torch.Tensor()

    for k, ranges in param_ranges.items():
        if ranges[0] == 'uniform':
            sample = torch.FloatTensor(n_samples, 1).uniform_(ranges[1][0], ranges[1][1])
            initial_X = torch.cat((initial_X, sample))
        elif ranges[0] == 'discrete':
            vals = ranges[1]
            sample = torch.Tensor(random.choices(vals, k = n_samples))
            initial_X = torch.cat((initial_X, torch.unsqueeze(sample, 1)))
        elif ranges[0] == 'int':
            sample = torch.randint(ranges[1][0], ranges[1][1]+1, (n_samples, 1))
            initial_X = torch.cat((initial_X, sample))
    
    return initial_X


# This can also go to utils

def get_param_bounds(param_ranges):
    
    num_params = len(param_ranges)
    bounds = torch.empty(2, num_params)
    fixed_features_list = []
    # TODO: also return the is_int feature to be passed into Matern kernel
    is_int = []

    for i, ranges in enumerate(param_ranges.values()):
        bounds[0,i] = min(ranges[1])
        bounds[1,i] = max(ranges[1])

        if ranges[1] in ['discrete', 'int']:
            is_int.append(1)
        else:
            is_int.append(0)
    
    return bounds, is_int
    

# copied from BoTorch 
# https://botorch.org/v/0.3.2/api/_modules/botorch/models/gpytorch.html#BatchedMultiOutputGPyTorchModel.get_batch_dimensions
def get_batch_dimensions(
    train_X: torch.Tensor, train_Y: torch.Tensor
) -> Tuple[torch.Size, torch.Size]:
    r"""Get the raw batch shape and output-augmented batch shape of the inputs.

    Args:
        train_X: A `n x d` or `batch_shape x n x d` (batch mode) tensor of training
            features.
        train_Y: A `n x m` or `batch_shape x n x m` (batch mode) tensor of
            training observations.

    Returns:
        2-element tuple containing

        - The `input_batch_shape`
        - The output-augmented batch shape: `input_batch_shape x (m)`
    """
    input_batch_shape = train_X.shape[:-2]
    aug_batch_shape = input_batch_shape
    num_outputs = train_Y.shape[-1]
    if num_outputs > 1:
        aug_batch_shape += torch.Size([num_outputs])

    return input_batch_shape, aug_batch_shape




"""
On bounds of the search space:
# TODO: specify hp config search space -- how to do?
        # first, specify bounds, sth like
        bounds = torch.tensor([[0.0] * problem.dim, [1.0] * problem.dim], **tkwargs)

# TODO: also simulate a random policy
# TODO: write evaluate(); implement more sophisticated early stopping; then look at KG
"""