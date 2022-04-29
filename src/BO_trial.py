from typing import Callable
import torch
import gpytorch
import random
import numpy as np
import pdb, time, argparse, itertools, copy
import sys, os
from collections import defaultdict

sys.path.append('../')

from botorch.models import SingleTaskGP, KroneckerMultiTaskGP, MultiTaskGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition.knowledge_gradient import qKnowledgeGradient, qMultiFidelityKnowledgeGradient
from botorch.acquisition.monte_carlo import qExpectedImprovement, qNoisyExpectedImprovement
from botorch.sampling.samplers import SobolQMCNormalSampler, IIDNormalSampler
from botorch.optim import optimize_acqf, optimize_acqf_mixed
from botorch.optim.initializers import gen_one_shot_kg_initial_conditions
from botorch.acquisition.objective import LinearMCObjective, ScalarizedPosteriorTransform
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.acquisition import PosteriorMean
from botorch.models.cost import AffineFidelityCostModel
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.models.kernels.exponential_decay import ExponentialDecayKernel

from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.kernels import ScaleKernel, MaternKernel, IndexKernel, ProductKernel
from gpytorch.priors.torch_priors import GammaPrior

from .models.kernels import ModifiedIndexKernel, IndicatorKernel
from .utils.plotting import plot_progress, plot_acqf_vals_and_fidelities
from .utils.multi_task_fidelity_utils import get_fidelity_covariance, print_kernel_hyperparams, process_multitask_data, expand_intermediate_fidelities, \
    get_task_covariance, get_task_fidelity_covariance

# TODO: import a list of kernels that I can call by string

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
        kernel_name = 'matern_expdecay_index_product',
        multifidelity_params = None,
        checkpoint_fidelities = None,
        **tkwargs
        ):

    # Get script directory
    script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
    results_folder = script_dir + "/results/" + problem_name + "/" + algo + "/"

    print('starting trial {} for {}, saving results to {}'.format(trial, problem_name, results_folder))

    X = None
    y = []
    acqf_vals = torch.Tensor().cpu()
    sampled_fidelities = []

    num_checkpoints = len(checkpoint_fidelities)

    if multifidelity_params is not None:
        cost_model = AffineFidelityCostModel(
            fidelity_weights = multifidelity_params['fidelity_weights'], 
            fixed_cost = multifidelity_params['fixed_cost'])
        cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)
        fidelity_dim = multifidelity_params['fidelity_dim']
        target_fidelities = multifidelity_params['target_fidelities']
        if not is_multitask:
            def project(X):
                return project_to_target_fidelity(X=X, target_fidelities=target_fidelities)
        else:
            target_fidelities[fidelity_dim+1] = 0 # define the target task fidelity to be 0 (main task)
            def project(X):
                return project_to_target_fidelity(X=X, target_fidelities=target_fidelities)
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
            cum_costs = [cost_model(X).sum().item()] # evaluate cost of sampling initial X

    else:
        # generate initial data
        # X has shape (n_initial_pts) * (design_dim + 1)
        X = generate_initial_samples(n_initial_pts, param_ranges, trial)
        # y has shape (num_trials * num_checkpoints) x num_outputs
        y = problem_evaluate(X)
        # then, incorporate intermediate fidelities into X
        # expanded X has shape (n_initial_pts * num_checkpoints) * (design_dim + 1)
        X = expand_intermediate_fidelities(X, checkpoint_fidelities, last_dim_is_task = False)

        init_batch_id = 1

        np.save(results_folder + 'X/X_' + str(trial) + '.npy', X)
        np.save(results_folder + 'output_at_X/output_at_X_' + str(trial) + '.npy', y)

        log_best_so_far = []
        runtimes = []
        cum_costs = [cost_model(X[num_checkpoints-1 : : num_checkpoints]).sum().item()] # evaluate cost of sampling initial X
        
    print('loaded / generated data for {} BO iteration(s)'.format(init_batch_id))
    print('before BO start, X shape, y shape'.format(X.shape, y.shape))

    # if multi-output, process X and y to add task dimension to X
    num_outputs = y.shape[-1]
    if is_multitask:
        param_ranges['task_idx'] = ['int', [0, num_outputs-1]]
        # initial X does not contain the task column, so set add_last_col_X to True
        X, y = process_multitask_data(X, y, num_checkpoints, add_last_col_X = True)
        max_posterior_mean = [y[::num_outputs][(num_checkpoints-1)::num_checkpoints].max().item()]
    else:
        max_posterior_mean = [y.max().item()]

    bounds, is_int = get_param_bounds(param_ranges)
    print('bounds', bounds)
    print('is_int', is_int)

    weights = torch.cat((torch.tensor([1]), torch.zeros(num_outputs-1)))
    objective = ScalarizedPosteriorTransform(weights)

    print('number of outputs: {}; weights for linear objective: {}'.format(num_outputs, weights))

    # dictionary for logging data
    out = {}

    for iter in range(init_batch_id, n_bo_iter+1):
        
        print('starting BO iteration ', iter)

        start_time = time.time()

        # this calls fit_GP_model() inside
        new_pt, acqf_val, current_max_posterior_mean = optimize_acqf_and_suggest_new_pt(
            algo, X, y, objective, bounds, param_ranges, trial, is_multitask, kernel_name, is_int, 
            cost_aware_utility, project, fidelity_dim, num_outputs, num_checkpoints)

        max_posterior_mean.append(current_max_posterior_mean)
        sampled_fidelities.append(new_pt[0][fidelity_dim].item())

        new_y = problem_evaluate(new_pt)
        print('evaluation of newly sampled point {}'.format(new_y))
        print('shape of evaluation of newly sampled point {}'.format(new_y.shape))
        print('shape of newly sampled point before checkpoint-fidelity expansion {}'.format(new_pt.shape))

        if is_multitask:
            # last dimension of new_pt is the task column
            new_pt = expand_intermediate_fidelities(new_pt, checkpoint_fidelities, last_dim_is_task = True)
            print('shape of newly sampled point after checkpoint-fidelity expansion {}'.format(new_pt.shape))

            new_pt, new_y = process_multitask_data(new_pt, new_y, num_checkpoints, add_last_col_X=True)
            print('shape of newly sampled point and evaluation after multi-task expansion {}'.format(new_pt.shape, new_y.shape))
        else:
            new_pt = expand_intermediate_fidelities(new_pt, checkpoint_fidelities, last_dim_is_task = False)
            print('shape of newly sampled point after checkpoint-fidelity expansion {}'.format(new_pt.shape))

        acqf_vals = torch.cat((acqf_vals, acqf_val))

        if cost_model is not None:
            cum_costs.append(cum_costs[-1] + cost_model(new_pt)[-1].item())
        
        print('cumulative cost', cum_costs)

        runtimes.append(time.time() - start_time)

        X = torch.cat((X, new_pt), dim = 0)
        y = torch.cat((y, new_y), dim = 0)
        print('shape of X and y after concatenating new data point: ', X.shape, y.shape)

        # only log the best value of task 0 at the highest fidelity sampled, not the intermediate ones
        if not is_multitask:
            log_best_so_far = y[::num_checkpoints].cummax(0).values[n_initial_pts-1:]
        else:
            log_best_so_far = y[::num_outputs][(num_checkpoints-1)::num_checkpoints].cummax(0).values[n_initial_pts-1:]
            
        if verbose:
            print('Finished iteration {}, best value so far is {}'.format(iter, log_best_so_far[-1].item()))

        # save results in dictionary
        out['X'] = X
        out['Y'] = y
        out['runtimes'] = runtimes
        out['best_so_far'] = log_best_so_far
        out['acqf_vals'] = acqf_vals
        out['cum_costs'] = cum_costs

        torch.save(out, results_folder + 'trial_' + str(trial) + '_' + kernel_name) # TODO: make kernel name an input

        title = 'best objective value for ' + problem_name + ' with ' + algo
        if is_multitask:
            title += ' (multitask)'

        plot_progress([title, log_best_so_far], cum_costs, results_folder, trial, max_posterior_mean = max_posterior_mean)
        plot_acqf_vals_and_fidelities(acqf_vals, sampled_fidelities, results_folder, trial)


def fit_GP_model(X, y, is_multitask, kernel_name, is_int=None, num_outputs = None):

    # TODO: What kind of kernel to use is something we want to revisit later [P1]

    if not is_multitask:
        # if use_additive_kernel:
        #     covar_module = ScaleKernel(MaternKernel(active_dims = torch.arange(0, X.shape[-1]-1))) + \
        #                  ScaleKernel(ExponentialDecayKernel(active_dims = torch.tensor([X.shape[-1]-1])))
        # else:
        #     covar_module = ProductKernel(ScaleKernel(MaternKernel(active_dims = torch.arange(0, X.shape[-1]-1))), 
        #                  ExponentialDecayKernel(active_dims = torch.tensor([X.shape[-1]-1])) )

        # model = SingleTaskGP(X, y, covar_module=covar_module)
        # TODO: deal with the single task case later
        pass

    else:
        # change to single task GP with my custom kernel on {inputs} x fidelity x task
        # option 1: matern on inputs, exponentially decaying kernel on fidelity, index kernel on task
        # option 2: MISO kernel on {inputs} x task, exponentially decaying kernel on fidelity

        if kernel_name == 'matern_expdecay_index_product':
            covar_module = ProductKernel(
                MaternKernel(active_dims = torch.arange(0, X.shape[-1]-2)),
                ExponentialDecayKernel(
                    active_dims = torch.tensor([X.shape[-1]-2]), 
                    lengthscale_prior=GammaPrior(3.0, 6.0),
                    # offset_prior=GammaPrior(3.0, 6.0),
                    # power_prior=GammaPrior(3.0, 6.0)
                    ),
                ModifiedIndexKernel(
                    active_dims = torch.tensor([X.shape[-1]-1]), 
                    num_tasks=num_outputs,
                    # prior=GammaPrior(3.0, 6.0)
                    )
                )
            # Note to self: the reason I got rid of the prior is b/c if I apply them then fid kernel params are constantly at 0
        elif kernel_name == 'matern_index_product':
            covar_module = ProductKernel(
                MaternKernel(active_dims = torch.arange(0, X.shape[-1]-1)),
                ModifiedIndexKernel(
                    active_dims = torch.tensor([X.shape[-1]-1]), 
                    num_tasks = num_outputs,
                    )
                )
        elif kernel_name == 'matern_index_additive':
            covar_module = ProductKernel(
                MaternKernel(active_dims = torch.arange(0, X.shape[-1]-2)),
                ModifiedIndexKernel(
                    active_dims = torch.tensor([X.shape[-1]-1]), 
                    num_tasks=3,
                    # prior=GammaPrior(3.0, 6.0)
                    )
                ) + ProductKernel(
                    IndicatorKernel(X.shape[-1]-1), 
                    MaternKernel(active_dims = torch.arange(0, X.shape[-1]-1))
                )
        elif kernel_name == 'matern_expdecay_index_additive':
            covar_module = ScaleKernel(MaternKernel(active_dims = torch.arange(0, X.shape[-1]-2))) + \
                            ScaleKernel(ExponentialDecayKernel(active_dims = torch.tensor([X.shape[-1]-2]))) + \
                            ScaleKernel(IndexKernel(active_dims = torch.tensor([X.shape[-1]-1]), num_tasks=num_outputs))
        else:
            print('kernel name is not recognized')
            
        model = SingleTaskGP(X, y, covar_module = covar_module) # TODO: check how the two ways of defining kernels differ

        # TODO: Later, explore kernels that deal with integers better

    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_model(mll)

    # TODO: saving (design, fidelity) is likely required for freeze-thaw
        # load state dict if it is passed
        # if state_dict is not None:
        # model.load_state_dict(state_dict)
 
    return model

def optimize_acqf_and_suggest_new_pt(
    algo, X, y, objective, bounds, param_ranges, trial, is_multitask, kernel_name, 
    is_int=None, cost_aware_utility = None, project = None, 
    fidelity_dim = None, num_outputs = None, num_fidelities = None, **kwargs):

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
        model = fit_GP_model(X, y, is_multitask, kernel_name, is_int)
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
        model = fit_GP_model(X, y, is_multitask, kernel_name, is_int, num_outputs)

        # TODO: instead of printing, save these in a dictionary
        # if is_multitask:
            # print('task-fidelity covariance matrix', get_task_fidelity_covariance(model, X, num_outputs, num_fidelities))
            # print('task covariance matrix', get_task_covariance(model, X, num_outputs))
            # print('fidelity covariance matrix', get_fidelity_covariance(model))

        acqf = get_mfkg(model, objective, bounds, cost_aware_utility, project, fidelity_dim, is_multitask)

        current_max_posterior_mean = acqf.current_value

        # TODO: does fixing the task to be 0 affect the acqf optimization?
        # I think so, because it enforces evaluating the main task
        # But then the algorithm doesn't know that other tasks will be observed as well, 
        # so the between-task correlation learned so far won't enter the decision process.

        # Sidenote: setting fixed_features here or not shouldn't matter b/c there's the project() operator already
        if is_multitask:
            fixed_features = {fidelity_dim + 1: 0} # fixed task to be 0
        else:
            fixed_features = None
        
        # generate KG initial conditions with fidelity fixed to 1 and task fixed to 0 (if multi-task)
        # note that bounds is still the full set of bounds, including those that were fixed during get_mfkg()

        X_init = gen_one_shot_kg_initial_conditions(
            acq_function = acqf,
            bounds=bounds,
            fixed_features = fixed_features,
            q=1,
            num_restarts=10, # default value is 20
            raw_samples=512, # default value is 1024
            options = {
                'num_inner_restarts': 10, 
                'raw_inner_samples': 512, # default is 20 and 1024
                'batch_limit': 5 # if this is not specified, raw_inner_samples posterior computations happen at the same time, leading to OOM
                } 
        )
   
        candidates, acqf_val = optimize_acqf(
            acq_function = acqf,
            bounds = bounds, 
            q = 1,
            num_restarts = 10, 
            raw_samples = 512,
            fixed_features = fixed_features,
            batch_initial_conditions = X_init,
            options={"batch_limit": 5, "maxiter": 200}, # TODO: see if decreasing maxiter helps
        )
        
    if len(acqf_val.size()) == 0:
        acqf_val = acqf_val.unsqueeze(0)
    
    for i in range(candidates.shape[-1]):
        if is_int[i]:
            candidates[..., i] = torch.round(candidates[..., i])

    print('optimize MultiFidelityKG, get candidates ', candidates, ', acqf_val ', acqf_val)

    # delete model to free memory
    del model, acqf

    # candidates has shape q x (design_dim + 2) -- both fidelity and task are included
    return candidates, acqf_val, current_max_posterior_mean


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
    

def get_mfkg(model, objective, bounds, cost_aware_utility, project, fidelity_dim, is_multitask):

    if is_multitask:
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
    
    # get the largest mean (of the main task at highest fidelity) under the current posterior,
    # optimizing with respect to the designs only
    _, current_value = optimize_acqf(
        acq_function=curr_val_acqf,
        bounds = _bounds,
        q=1,
        num_restarts=10, 
        raw_samples=128, 
        options={"batch_limit": 10, "maxiter": 200},
    )

    print('current max posterior mean before sampling new points: {}'.format(current_value))
        
    # return the KG, the expected increase in best expected value conditioned on q more samples
    return qMultiFidelityKnowledgeGradient(
        model=model,
        num_fantasies=64, # sized down from 128, see how it affects memory
        current_value=current_value,
        cost_aware_utility=cost_aware_utility,
        project=project,
    )


# TODO: Next is to understand how to correctly inspect the task-fidelity covariance!
