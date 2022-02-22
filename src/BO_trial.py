from typing import Callable
import torch
import random
import numpy as np
import pdb, time, argparse, itertools, copy
import sys, os
from collections import defaultdict
sys.path.append('../')

from botorch.models import SingleTaskGP, FixedNoiseGP, ModelListGP, KroneckerMultiTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition.acquisition import AcquisitionFunction
from botorch.acquisition.analytic import ExpectedImprovement
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.optim import optimize_acqf, optimize_acqf_mixed
from botorch.utils import standardize

from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel
from gpytorch.priors.torch_priors import GammaPrior

from .models.kernels import ModifiedMaternKernel
from .utils.multitask_GP import *

def BO_trial(
        problem_evaluate: Callable,
        problem_name: str,
        #input_dim: int,
        param_ranges: dict,
        algo: str,
        n_initial_pts: int,
        n_bo_iter: int,
        trial: int,
        restart: bool,
        verbose: bool
        ):

    # Get script directory
    script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    results_folder = script_dir + "/results/" + problem_name + "/" + algo + "/"

    X = None
    y = []
    acqf_vals = torch.Tensor().cpu()

    bounds, is_int = get_param_bounds(param_ranges)
    print('is_int', is_int)

    if restart:
        # check if there is saved data available
        try: 
            # get saaved data
            X = torch.tensor(np.load(results_folder + 'X/X_' + str(trial) + '.npy'))
            y = torch.tensor(np.load(results_folder + 'objective_at_X/objective_at_X_' + str(trial) + '.npy'))

            runtimes = list(np.load(results_folder + 'runtimes/runtimes_' + str(trial) + '.npy'))
            log_best_so_far = list(np.load(results_folder + 'log_best_so_far_' + str(trial) + '.npy'))

            init_batch_id = len(log_best_so_far)
            
        except:
            # generate initial data
            X = generate_initial_samples(n_initial_pts, param_ranges, trial)
            y = problem_evaluate(X)
            init_batch_id = 1

            np.save(results_folder + 'X/X_' + str(trial) + '.npy', X)
            np.save(results_folder + 'objective_at_X/objective_at_X_' + str(trial) + '.npy', y)

    print('loaded / generated data for {} BO iteration(s)'.format(init_batch_id))

    log_best_so_far = []
    runtimes = []

    # TODO: make a defaultdict(list)

    print('before BO start, X shape, y shape'.format(X.shape, y.shape))

    for iter in range(init_batch_id, n_bo_iter+1):

        print('starting BO iteration ', init_batch_id)

        start_time = time.time()

        # suggest_new_pt() calls fit_GP_model()
        new_pt, acqf_val = suggest_new_pt(algo, X, y, bounds, is_int, param_ranges, trial)
        new_y = problem_evaluate(new_pt)

        acqf_vals = torch.cat((acqf_vals, acqf_val))
        print('acqf vals', acqf_vals)

        runtimes.append(time.time() - start_time)

        X = torch.cat((X, new_pt), dim = 0)
        y = torch.cat((y, new_y), dim = 0)

        best_so_far = y.max().item()

        # log the best-performing ones so far (including random)
        log_best_so_far.append(best_so_far)

        if verbose:
            print('Finished iteration {}, best value so far is {}'.format(iter, best_so_far))

        np.save(results_folder + 'X/X_' + str(trial) + '.npy', X)
        np.save(results_folder + 'objective_at_X/objective_at_X_' + str(trial) + '.npy', y)
        np.save(results_folder + 'runtimes/runtimes_' + str(trial) + '.npy', runtimes)
        np.save(results_folder + 'log_best_so_far_' + str(trial) + '.npy', log_best_so_far)
        np.save(results_folder + 'acqf_vals_' + str(trial) + '.npy', acqf_vals)


def fit_GP_model(X, y, is_int, multitask = False):

    #_, aug_batch_shape = get_batch_dimensions(X,y)
    #print(aug_batch_shape)

    # the modified matern kernel rounds the values for integer variables 
    # before computing the kernel output
    # TODO: What kind of kernel to use is something we want to revisit later [P1]
    
    covar_module = ScaleKernel(
        ModifiedMaternKernel(
            is_int,
            nu = 2.5,
            ard_num_dims = X.shape[-1],
            lengthscale_prior = GammaPrior(3.0, 6.0),
        ),
        outputscale_prior=GammaPrior(2.0, 0.15)
    )

    if not multitask:
        model = SingleTaskGP(X, y, covar_module = covar_module)
    else:
        model = KroneckerMultiTaskGP(X, y)

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)

    # TODO: figure this out
        # load state dict if it is passed
        # if state_dict is not None:
        # model.load_state_dict(state_dict)


 
    return model


def suggest_new_pt(algo, X, y, bounds, is_int, param_ranges, trial):
    print('suggesting new point')

    if algo == 'random':
        return generate_initial_samples(1, param_ranges, seed = trial)

    elif algo == 'EI':
        # fit GP model based on X, y
        model = fit_GP_model(X, y, is_int)
        best = y.max().item()
        # define acq function
        # TODO: change to q-ExpectedImprovement
        acqf = qExpectedImprovement(model, best)
        print('use EI, fit GP, best value is {}'.format(best))
    
    elif algo == 'KG':
        # TODO: 
        # fit a multi output GP model
        # set acqf to be qMultiFidelityKG
        print('use multifidelityKG, fit GP')
        pass

    # TODO: can you just plug in a different optimizer herE?? work on this later
    candidates, acqf_val = optimize_acqf(
        acq_function = acqf,
        bounds = bounds,
        q = 1, # TODO: figure out the following three
        num_restarts = 10,
        raw_samples = 512
    )

    print('candidates', candidates, candidates.shape)
    print('acqf_val', acqf_val, acqf_val.shape)

    if len(acqf_val.size()) == 0:
        acqf_val = acqf_val.unsqueeze(0)

    return candidates, acqf_val


# This can go to utils/

def generate_initial_samples(n_samples, param_ranges, seed=None):

    if seed is not None:
        torch.manual_seed(seed)

    initial_X = torch.Tensor()

    for k, ranges in param_ranges.items():
        if ranges[0] == 'uniform':
            sample = torch.FloatTensor(n_samples, 1).uniform_(ranges[1][0], ranges[1][1])
            initial_X = torch.cat((initial_X, sample), dim = 1)
        
        elif ranges[0] == 'int':
            sample = torch.randint(ranges[1][0], ranges[1][1]+1, (n_samples, 1))
            initial_X = torch.cat((initial_X, sample), dim = 1)

        elif ranges[0] == 'discrete':
            vals = ranges[1]
            sample = torch.Tensor(random.choices(vals, k = n_samples))
            initial_X = torch.cat((initial_X, torch.unsqueeze(sample, 1)), dim = 1)
    
    return initial_X


# This can also go to utils

def get_param_bounds(param_ranges):
    
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
    
    return bounds, is_int
    

# copied from BoTorch 
# https://botorch.org/v/0.3.2/api/_modules/botorch/models/gpytorch.html#BatchedMultiOutputGPyTorchModel.get_batch_dimensions
def get_batch_dimensions(train_X, train_Y):
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