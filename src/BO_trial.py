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
from experiment_problems import problem_evaluate

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
        problem: Callable,
        problem_name: str,
        input_dim: int,
        active_input_indices: List[List[Optional[int]]],
        algo: str,
        n_initial_pts: int,
        n_bo_iter: int,
        trial: int,
        restart: bool
        ):
    # runs loop

    X = None
    y = []

    # TODO: revisit the <problem> construction: make it a class, or a function?
    # goal is to have it input hp config and output training performance

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
                sampled_hp_config = None
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

    best_all_trials.append(best_log)


def fit_GP_model(X, y):

    model = SingleTaskGP(X, y)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return model, mll


def suggest_new_pt(algo, X, y):
    if algo == 'EI':
        # fit GP model based on X, y
        model, _ = fit_GP_model(X, y)
        best = y.max().item()
        # define acq function
        acqf = ExpectedImprovement(model, best)
    
    # call optimize_acquisition_function_and_get_suggested_point()
    # return new point



















# --- below is deprecated



class MyMethod():

    # TODO: add a function to the Learner class that returns a "terminal" performance metric(s) of some sort

    def __init__(self, problem, obj_func, path, num_trials, n_initial_pts, acqf_key, algo_key, search_space): # more attributes
        # learner: maps input (e.g., hp config) to output(s) (e.g., validation loss / acc)
        # path: where BO iteration data are saved

        self.problem = problem
        self.obj_func = obj_func
        self.path = path
        self.num_trials = num_trials
        self.n_initial_pts = n_initial_pts
        
        if acqf_key == 'EI':
            self.acqf = ExpectedImprovement()
        # TODO: enable more (+ custom) acquisition function types
        elif acqf_key == 'KG':
            pass

        # also specify what BO algorithm to use
        # if algo_key == 'something':

        self.X = None
        self.y = []

        # TODO: specify hp config search space -- how to do?
        # first, specify bounds, sth like
        bounds = torch.tensor([[0.0] * problem.dim, [1.0] * problem.dim], **tkwargs)

        # TODO: used to have 3 categories: base_config, self.opt_kwargs, self.loss_fn_kwargs
        # include architecture config as 4th? or include in base_config
    

    def get_initial_data(self):
        # get initial data for fitting the GP and likelihood parameters
        
        # either here, or in run(), enable restarting from available saved data


        for _ in range(self.n_initial_pts):
            # TODO: figure out how to sample hp config from search space
            sampled_hp_config = None
            outcome = self.problem_evaluate(sampled_hp_config)
            self.X = np.concatenate([self.X, sampled_hp_config], axis=0)
            self.y.append(outcome)
        
        
    def fit_model(self):

        model = SingleTaskGP(self.X, self.y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        # TODO: figure this out
        # load state dict if it is passed
        # if state_dict is not None:
        # model.load_state_dict(state_dict)
        
        return mll, model


    def evaluate_new_pt(self, new_pt):
        # something like this
        return self.problem.evaluate(new_pt)
        
        
    def suggest_new_pt(self):
        # given existing (X, Y), call optimize_acqf, return suggested values (And predictions?)

        candidates, _ = optimize_acqf_mixed(
            self.acqf,

        )



        pass

    def run(self, MC_SAMPLES):

        # first get initial data: X, y
        self.get_initial_data()

        # then fit an initial GP and mll using initial data
        mll, model = self.fit_model()
        best_all_trials = []
        qmc_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES) # or something else

        for trial_idx in self.num_trials:

            best = np.inf # or soemthing else
            best_log = []
            fit_gpytorch_model(mll)

            # define acqf
            if self.acqf_key == 'EI':
                acqf = qExpectedImprovement(model, best, qmc_sampler, self.obj_func)

            new_pt = self.suggest_new_pt(acqf)
            new_y = self.evaluate_new_pt(new_pt)
            X = torch.cat([X, new_pt]); y = torch.cat([y, new_y])

            # log the best-performing ones so far (including random)
            best_log.append(new_y)

            # re-fit model
            mll, model = self.fit_model(X, y)

            # if verbose: print stuff to STDOUT
            # also, save the newest X and y

        best_all_trials.append(best_log)


    # TODO: also simulate a random policy
    # TODO: write evaluate(); implement more sophisticated early stopping; then look at KG
