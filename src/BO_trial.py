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
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels.matern_kernel import MaternKernel
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler

from botorch.optim import optimize_acqf, optimize_acqf_mixed
from botorch.utils import standardize




def BO_trial(
        problem_evaluate: Callable,
        problem_name: str,
        input_dim: int,
        #active_input_indices: List[List[Optional[int]]],
        algo: str,
        n_initial_pts: int,
        n_bo_iter: int,
        trial: int,
        restart: bool
        ):

    X = None
    y = []

    if restart:
        # check if there is saved data available
        try: 
            # get saaved data
            # init_batch_id = len(hist_best_obs_vals)
            pass
        except:
            # generate initial data
            # or wrap this into a function and put into utils
            for _ in range(n_initial_pts):
                # TODO: figure out how to sample hp config from search space
                sampled_hp_config = generate_initial_samples()
                outcome = problem_evaluate(sampled_hp_config)
                X = np.concatenate([X, sampled_hp_config], axis=0)
                y.append(outcome)
            init_batch_id = 1

    # fit initial model
    model, mll = fit_GP_model(X, y)    

    best_all_trials = []

    for iter in range(init_batch_id, n_bo_iter+1):

        best = np.inf # or soemthing else
        best_log = []
        fit_gpytorch_model(mll)

        # define acqf
        if algo == 'EI':
            acqf = ExpectedImprovement(model, best)

        new_pt = suggest_new_pt(algo, X, y)
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


def fit_GP_model(X, y):

    model = SingleTaskGP(X, y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)

    # TODO: figure this out
        # load state dict if it is passed
        # if state_dict is not None:
        # model.load_state_dict(state_dict)

    return model, mll


def suggest_new_pt(algo, X, y):
    if algo == 'EI':
        # fit GP model based on X, y
        model, _ = fit_GP_model(X, y)
        best = y.max().item()
        # define acq function
        acqf = ExpectedImprovement(model, best)
    
    # call optimize_acquisition_function_and_get_suggested_point()
    # or optimize_acqf_mixed()
    # return new point

def generate_initial_samples(n_samples, param_ranges, seed = None):
    # TODO: generate initial samples of hp_config, make them tensors?
    # if you make them tensors, need to encode which dim is which hyperparam
    
    # TODO: seed?

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


"""
On bounds of the search space:
# TODO: specify hp config search space -- how to do?
        # first, specify bounds, sth like
        bounds = torch.tensor([[0.0] * problem.dim, [1.0] * problem.dim], **tkwargs)

# TODO: also simulate a random policy
# TODO: write evaluate(); implement more sophisticated early stopping; then look at KG
"""