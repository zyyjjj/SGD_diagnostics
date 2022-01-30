import math
import torch
import gpytorch
import os
from matplotlib import pyplot as plt


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=2
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=2, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

def fit_multitask_GP_model(train_x, train_y, num_tasks, training_iterations=20, initial_lr=0.1):

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks)
    model = MultitaskGPModel(train_x, train_y, likelihood)

    # set GP model and likelihood into training mode; prepare to optimize their params
    model.train()
    likelihood.train()

    # Use the adam optimizer
    # Note that model.params() include GaussianLikelihood parameters
    # because model() is defined usign likelihood
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)  

    # "Negative loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
        optimizer.step()
    
    model.eval()
    likelihood.eval()

    return model

# TODO:
# design an acquisition function, taking into consideration
    # multiple tasks, each task with potentially multiple fidelities
    # some observations come for free, some at extra cost
# optimize acqf and suggest new *point*


if __name__ == "__main__":

    # TODO: inputs are to be replaced by hp configs
    train_x = torch.linspace(0, 1, 100)

    # TODO: outputs are to be replaced by performance metrics
    train_y = torch.stack([
        torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2,
        torch.cos(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * 0.2,
    ], -1)

    # also TODO: enable specifying num_tasks in MultitaskGPModel()
    model = fit_multitask_GP_model(train_x, train_y, 2)