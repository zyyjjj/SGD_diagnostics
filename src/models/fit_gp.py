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
from utils.learner import Learner
from utils.callback import *

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


class MyMethod():

    # TODO: add a function to the Learner class that returns a "terminal" performance metric(s) of some sort

    def __init__(self, Learner, obj_func, path, num_trials, n_initial_pts, acqf_key, search_space): # more attributes
        # learner: maps input (e.g., hp config) to output(s) (e.g., validation loss / acc)
        # path: where BO iteration data are saved

        self.Learner = Learner
        self.obj_func = obj_func
        self.path = path
        self.num_trials = num_trials
        self.n_initial_pts = n_initial_pts
        if acqf_key == 'EI':
            self.acqf = ExpectedImprovement()
        # TODO: enable more (+ custom) acquisition function types
        elif acqf_key == 'KG':
            pass

        self.X = None
        self.y = []
    

    def get_initial_data(self):
        # get initial data for fitting the GP and likelihood parameters
        
        # either here, or in run(), enable restarting from available saved data


        for _ in range(self.n_initial_pts):
            # TODO: figure out how to sample hp config from search space
            sampled_hp_config = None
            learner = self.Learner(sampled_hp_config)
            outcome = learner.run_fit() # this could be one or multiple values
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


    def evaluate_new_pt(self, learner, new_pt):
        # something like this
        return learner.evaluate(new_pt)
        
        
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
