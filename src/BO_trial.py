from concurrent.futures import process
from typing import Callable
from matplotlib import projections
import torch
import random
import numpy as np
import pdb, time, argparse, itertools, copy
import sys, os
from collections import defaultdict
sys.path.append('../')

from botorch.models import SingleTaskGP, KroneckerMultiTaskGP, MultiTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler, IIDNormalSampler
from botorch.optim import optimize_acqf, optimize_acqf_mixed
from botorch.optim.initializers import gen_one_shot_kg_initial_conditions
from botorch.utils import standardize
from botorch.acquisition.objective import LinearMCObjective, ScalarizedPosteriorTransform
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.acquisition import PosteriorMean
from botorch.models.cost import AffineFidelityCostModel
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.models.kernels.exponential_decay import ExponentialDecayKernel

from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, MaternKernel, IndexKernel
from gpytorch.priors.torch_priors import GammaPrior

from .models.kernels import ModifiedMaternKernel, InputFidelityTaskProductKernel, InvertedExponentialDecayKernel
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
        verbose: bool,
        is_multitask = True,
        use_additive_kernel = False,
        multifidelity_params = None,
        **tkwargs
        ):

    # Get script directory
    script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    results_folder = script_dir + "/results/" + problem_name + "/" + algo + "/"

    print('starting trial {} for {}, saving results to {}'.format(trial, problem_name, results_folder))
    print('use additive kernel = {}'.format(use_additive_kernel))

    X = None
    y = []
    acqf_vals = torch.Tensor().cpu()

    if multifidelity_params is not None:
        cost_model = AffineFidelityCostModel(
            fidelity_weights = multifidelity_params['fidelity_weights'], 
            fixed_cost = multifidelity_params['fixed_cost'])
        cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)
        def project(X):
            return project_to_target_fidelity(X=X, target_fidelities=multifidelity_params['target_fidelities'])
        fidelity_dim = multifidelity_params['fidelity_dim']
    else:
        cost_model = AffineFidelityCostModel(fidelity_weights = {-1: 0}, fixed_cost = 1) # uniform cost 1 for all inputs
        cost_aware_utility = None
        project = None
        fidelity_dim = None

    if restart:
        # check if there is saved data available
        try: 
            # get saved data
            X = torch.tensor(np.load(results_folder + 'X/X_' + str(trial) + '.npy'))
            y = torch.tensor(np.load(results_folder + 'output_at_X/output_at_X_' + str(trial) + '.npy'))

            runtimes = list(np.load(results_folder + 'runtimes/runtimes_' + str(trial) + '.npy'))
            log_best_so_far = list(np.load(results_folder + 'log_best_so_far_' + str(trial) + '.npy'))
            cum_costs = list(np.load(results_folder + 'cum_costs_' + str(trial) + '.npy'))

            init_batch_id = len(log_best_so_far)
            
            print('load saved data for trial {}, starting from iteration {}'.format(trial, init_batch_id))

        except:
            # generate initial data
            X = generate_initial_samples(n_initial_pts, param_ranges, trial)
            y = problem_evaluate(X)
            init_batch_id = 1

            np.save(results_folder + 'X/X_' + str(trial) + '.npy', X)
            np.save(results_folder + 'output_at_X/output_at_X_' + str(trial) + '.npy', y)

            log_best_so_far = []
            runtimes = []
            cum_costs = [cost_model(X).sum()] # evaluate cost of sampling initial X

    else:
        # generate initial data
        X = generate_initial_samples(n_initial_pts, param_ranges, trial)
        print(X)
        y = problem_evaluate(X)
        init_batch_id = 1

        np.save(results_folder + 'X/X_' + str(trial) + '.npy', X)
        np.save(results_folder + 'output_at_X/output_at_X_' + str(trial) + '.npy', y)

        log_best_so_far = []
        runtimes = []
        cum_costs = [cost_model(X).sum()] # evaluate cost of sampling initial X
        
    print('loaded / generated data for {} BO iteration(s)'.format(init_batch_id))
    print('initial samples X and y: ', X, y)
    print('before BO start, X shape, y shape'.format(X.shape, y.shape))

    num_outputs = y.shape[-1]
    # if multi-output, process X and y to add task dimension to X
    if is_multitask:
        param_ranges['task_idx'] = ['int', [0, num_outputs]]
        X, y = process_multitask_data(X, y, add_last_col_X = True)

    bounds, is_int = get_param_bounds(param_ranges)
    print('bounds', bounds)
    print('is_int', is_int)

    weights = torch.cat((torch.tensor([1]), torch.zeros(num_outputs-1)))
    objective = ScalarizedPosteriorTransform(weights)

    print('number of outputs: {}; weights for linear objective: {}'.format(num_outputs, weights))

    for iter in range(init_batch_id, n_bo_iter+1):

        print('starting BO iteration ', iter)

        start_time = time.time()

        # suggest_new_pt() calls fit_GP_model()
        new_pt, acqf_val = optimize_acqf_and_suggest_new_pt(
            algo, X, y, objective, bounds, param_ranges, trial, is_multitask, use_additive_kernel, is_int, 
            cost_aware_utility, project, fidelity_dim, num_outputs)
        # this should suggest a task-agnostic point (input, fidelity) since we observe all tasks

        new_y = problem_evaluate(new_pt)

        if is_multitask:
            new_pt, new_y = process_multitask_data(new_pt, new_y)

        acqf_vals = torch.cat((acqf_vals, acqf_val))

        if cost_model is not None:
            cum_costs.append(cost_model(new_pt).sum())

        runtimes.append(time.time() - start_time)

        X = torch.cat((X, new_pt), dim = 0)
        y = torch.cat((y, new_y), dim = 0)
        print('shape of X and y after concatenating new data point: ', X.shape, y.shape)

        if not is_multitask:
            log_best_so_far = y.cummax(0).values
        else:
            log_best_so_far = y[::num_outputs].cummax(0).values

            
        if verbose:
            print('Finished iteration {}, best value so far is {}'.format(iter, log_best_so_far[-1].item()))

        np.save(results_folder + 'X/X_' + str(trial) + '.npy', X)
        np.save(results_folder + 'output_at_X/output_at_X_' + str(trial) + '.npy', y)
        np.save(results_folder + 'runtimes/runtimes_' + str(trial) + '.npy', runtimes)
        np.save(results_folder + 'log_best_so_far_' + str(trial) + '.npy', log_best_so_far)
        np.save(results_folder + 'acqf_vals_' + str(trial) + '.npy', acqf_vals)
        np.save(results_folder + 'cum_costs_' + str(trial) + '.npy', cum_costs)

        title = 'best objective value for ' + problem_name + ' with ' + algo
        if is_multitask:
            title += ' (multitask)'

        # TODO: make the horizontal axis be the cumulative fidelities so far
        plot_progress({title: log_best_so_far}, results_folder, trial)

def fit_GP_model(X, y, is_multitask, use_additive_kernel, is_int=None, num_outputs = None):

    # the modified matern kernel rounds the values for integer variables 
    # before computing the kernel output
    # TODO: What kind of kernel to use is something we want to revisit later [P1]

    pdb.set_trace()

    if not is_multitask:
        if use_additive_kernel:
            covar_module = MaternKernel(active_dims = torch.arange(0, X.shape[-1]-1)) + \
                         ExponentialDecayKernel(active_dims = torch.tensor([X.shape[-1]-1]))
        else:
            covar_module = MaternKernel(active_dims = torch.arange(0, X.shape[-1]-1)) * \
                         ExponentialDecayKernel(active_dims = torch.tensor([X.shape[-1]-1]))
        
        model = SingleTaskGP(X, y, covar_module=covar_module)

    else:
        # X = (design, fidelity, task); y = (output) 
        # fit a single output GP

        # change to single task GP with my custom kernel on {inputs} x fidelity x task
        # option 1: matern on inputs, exponentially decaying kernel on fidelity, index kernel on task
        # option 2: MISO kernel on {inputs} x task, exponentially decaying kernel on fidelity

        if use_additive_kernel:
            covar_module = MaternKernel(active_dims = torch.arange(0, X.shape[-1]-2)) + \
                            ExponentialDecayKernel(active_dims = torch.tensor([X.shape[-1]-2])) + \
                            IndexKernel(active_dims = torch.tensor([X.shape[-1]-1]), num_tasks=num_outputs)
        else:
            covar_module = MaternKernel(active_dims = torch.arange(0, X.shape[-1]-2)) * \
                            ExponentialDecayKernel(active_dims = torch.tensor([X.shape[-1]-2])) * \
                            IndexKernel(active_dims = torch.tensor([X.shape[-1]-1]), num_tasks=num_outputs)

        model = SingleTaskGP(X, y, covar_module = covar_module)

        # TODO: Later, explore kernels that deal with integers better

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)

    # TODO: figure this out
        # load state dict if it is passed
        # if state_dict is not None:
        # model.load_state_dict(state_dict)
 
    return model


def optimize_acqf_and_suggest_new_pt(
    algo, X, y, objective, bounds, param_ranges, trial, is_multitask, use_additive_kernel, 
    is_int=None, cost_aware_utility = None, project = None, fidelity_dim = None, num_outputs = None):

    """ General steps for a non-random-sampling algorithm:
    1. define and fit GP model
    2. define sampler for evaluating the acqf
    3. construct acqf
    4. optimize acqf (w fixed feature if including fidelity)
    """
    
    print('suggesting new point')

    if algo == 'random':
        return generate_initial_samples(1, param_ranges, seed = trial)

    elif algo == 'EI':
        # fit GP model based on X, y
        model = fit_GP_model(X, y, is_multitask, use_additive_kernel, is_int)
        # sampler = SobolQMCNormalSampler(num_samples=64)
        sampler = IIDNormalSampler(num_samples=64)
        if not is_multitask:
            best = y.max().item()
        else:
            best = objective(y).max().item() # TODO; change this, though not super urgent as I'm primarily using KG now
        
        # define acq function
        acqf = qExpectedImprovement(
            model = model, 
            best_f = best, 
            sampler = sampler, 
            objective = objective
        )
        print('use EI, fit GP, best value is {}'.format(best))

        candidates, acqf_val = optimize_acqf(
            acq_function = acqf,
            bounds = bounds,
            q = 1, # TODO: figure out the following three
            num_restarts = 10,
            raw_samples = 512
            )

    elif algo == 'KG':

        # fit a multi output GP model
        model = fit_GP_model(X, y, is_multitask, use_additive_kernel, is_int, num_outputs)
        # TODO: can I output the task and fidelity kernels here?
        
        acqf = get_mfkg(model, objective, bounds, cost_aware_utility, project, fidelity_dim, is_multitask)
        
        X_init = gen_one_shot_kg_initial_conditions(
            acq_function = acqf,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=512,
        )

        candidates, acqf_val = optimize_acqf(
            acq_function = acqf,
            bounds=bounds,
            q=1,
            num_restarts=10,
            raw_samples=512,
            batch_initial_conditions=X_init,
            options={"batch_limit": 5, "maxiter": 200},
        )

    if len(acqf_val.size()) == 0:
        acqf_val = acqf_val.unsqueeze(0)
    
    # TODO: check correctness: key step of rounding integer entries of suggested candidate
    for i in range(candidates.shape[-1]):
        if is_int[i]:
            candidates[..., i] = torch.round(candidates[..., i])

    print('optimize MultiFidelityKG, get candidates ', candidates, ', acqf_val ', acqf_val)

    return candidates, acqf_val


def generate_initial_samples(n_samples, param_ranges, seed=None):

    if seed is not None:
        torch.manual_seed(seed)

    initial_X = torch.Tensor()

    for k, ranges in param_ranges.items():

        #if k == 'iteration_fidelity':
        #    initial_X = torch.cat((initial_X, torch.ones(n_samples, 1)), dim = 1)
        #    continue

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
    
    return bounds.float(), is_int
    

# TODO: make the x-axis represent the cumulative fidelities sampled so far
def plot_progress(metric, results_folder, trial):
    # metrics: dictionary of {metric_name: list of metric values}

    for k,v in metric.items():
        plt.plot(v)
        plt.title(k)
        plt.savefig(results_folder + 'visualization_trial_' + str(trial))


def get_mfkg(model, objective, bounds, cost_aware_utility, project, fidelity_dim, is_multitask):

    print('fidelity dim', fidelity_dim)

    if is_multitask:
        # get the largest mean under the current posterior
        curr_val_acqf = FixedFeatureAcquisitionFunction(
            acq_function=PosteriorMean(model),
            d=fidelity_dim + 2, 
            columns=[fidelity_dim, -1],
            values=[1, 0], # fix to fidelity = 1, task = 0 
        )
        _bounds = bounds[:,:-2]
    else:
        curr_val_acqf = FixedFeatureAcquisitionFunction(
            acq_function=PosteriorMean(model),
            d=fidelity_dim + 1, 
            columns=[fidelity_dim],
            values=[1], # fix to fidelity = 1
        )
        _bounds = bounds[:,:-1]
    

    # pdb.set_trace()
    _, current_value = optimize_acqf(
        acq_function=curr_val_acqf,
        bounds = _bounds,
        q=1,
        num_restarts=10, # TODO: change to 10 after debugging
        raw_samples=512, # TODO: change to 1024 after debugging
        options={"batch_limit": 10, "maxiter": 200},
        # options={"batch_limit": 10, "maxiter": 10},
    )
        
    # return the KG, the expected increase in best expected value conditioned on one more sample
    return qMultiFidelityKnowledgeGradient(
        model=model,
        num_fantasies=128,
        current_value=current_value,
        cost_aware_utility=cost_aware_utility,
        project=project,
        #posterior_transform=objective
    )

def process_multitask_data(X,y, add_last_col_X = False):
    # the goal is to transform (multi-dim input, multi-dim output) data into (multi-dim input, single-dim output) data
    # input shapes: X: num_trials x (design_dim + 1) or num_trials x (design_dim + 2) ; 
    #               y: num_trials x output_dim
    # target shapes: X: (num_trials * output_dim) x (design_dim + 2) ; y: num_trials x 1

    num_trials = y.shape[0]
    num_outputs = y.shape[-1]
    
    X_repeat = X.repeat_interleave(num_outputs, 0)
    task_idx_repeat = torch.arange(0,num_outputs).unsqueeze(1).repeat(num_trials,1)

    if add_last_col_X:
        # X shape: num_trials x (design_dim + 1)
        new_X = torch.cat((X_repeat, task_idx_repeat), 1)
    else:
        # X shape: num_trials x (design_dim + 2), but the last column is a dummy task column, not meaningful
        new_X = torch.cat((X_repeat[:, :-1], task_idx_repeat), 1)
    
    print('expanded X', new_X)

    new_y = y.flatten().unsqueeze(1)
    print('expanded y', new_y)
    
    return new_X, new_y