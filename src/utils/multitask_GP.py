import math
import random
from re import L
import torch
import gpytorch
import os
import pdb
from matplotlib import pyplot as plt

# THIS IS based on a Gpytorch tutorial on multi output GPs

class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, num_tasks, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks
        )
        # the MultitaskKernel of GPytorch uses the product between input and task kernels (i.e., ICM)
        # TODO: fully understand this; then try different kernels (LMC, ...)
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.MaternKernel(), num_tasks, rank=2
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

def fit_multitask_GP_model(train_x, train_y, num_tasks, training_iterations=20, initial_lr=0.1):

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks)
    model = MultitaskGPModel(train_x, train_y, num_tasks, likelihood)

    # set GP model and likelihood into training mode; prepare to optimize their params
    model.train()
    likelihood.train()

    # Use the adam optimizer
    # Note that model.params() include GaussianLikelihood parameters
    # because model() is defined using likelihood
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)  

    # marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iterations):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        if i % 5 == 0:
            print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
        optimizer.step()
    
    model.eval()
    likelihood.eval()

    return model, likelihood


def test_function_noisy(x,s,noise_level=0.2):

    output = torch.stack([
        (x * torch.exp(-x) + (10-x)*torch.exp(x-10)) * torch.sqrt(s) * 10 + torch.randn(x.size()) * noise_level,
        (x * torch.exp(-x) + (10-x)*torch.exp(x-10)) * 10 * 1/(2*torch.sqrt(s)) + torch.randn(x.size()) * noise_level
    ], -1)

    return output

def cost_at_fidelity(s):
    # can try more sophisticated ones later
    # but cost \propto s should capture a lot of the real-world scenarios
    return s
    

# TODO:
# design an acquisition function, taking into consideration
    # multiple tasks, each task with potentially multiple fidelities
    # some observations come for free, some at extra cost
# optimize acqf and suggest new *point*
# what are numerical methods for dealing with large scale inference? low rank approximations?


def multitask_BO_trial(function, param_ranges={'x':[0,10], 's': [0,1]}, n_initial_pts=10, n_bo_iter=20, **kwargs):
    
    xs = torch.FloatTensor(n_initial_pts, 1).uniform_(param_ranges['x'][0], param_ranges['x'][1])
    ss = torch.FloatTensor(n_initial_pts, 1).uniform_(param_ranges['s'][0], param_ranges['s'][1]/2)
    train_x_s = torch.cat((xs, ss), dim=1)

    print(train_x_s)

    cum_cost = 0

    train_y = []
    for i in range(n_initial_pts):
        train_y.append(function(train_x_s[i][0], train_x_s[i][1]))
        cum_cost += cost_at_fidelity(train_x_s[i][1])
    train_y = torch.stack(train_y)

    max_xs=[]
    true_vals = []
    max_means=[]
    cum_cost = [cum_cost]

    for iter in range(n_bo_iter):
        model, likelihood = fit_multitask_GP_model(train_x_s, train_y, 2, training_iterations = 30)
        # print tentative optimum so far
        max_x, max_mean = get_posterior_full_fid_max(model, likelihood)
        true_val = function(max_x, torch.Tensor([1]), noise_level=0)[0,0].item()
        max_xs.append(max_x.item())
        true_vals.append(true_val)
        max_means.append(max_mean.item())

        # randomly sample new point (x,s), incurring cost
        # refit GP, record observed function value, record cost
        new_x = torch.FloatTensor(1).uniform_(param_ranges['x'][0], param_ranges['x'][1])
        new_s = torch.FloatTensor(1).uniform_(param_ranges['s'][0], param_ranges['s'][1])

        cum_cost.append(cum_cost[-1] + cost_at_fidelity(new_s.item()) )

        new_y = function(new_x, new_s)
        
        train_x_s = torch.cat((train_x_s, torch.cat((new_x, new_s)).unsqueeze(0)))
        
        train_y = torch.cat((train_y, new_y))

        print(train_x_s.shape, train_y.shape)
        print(max_xs, max_means)


    true_maximizer, true_max_val = get_true_full_fid_max(function)

    f, (x_ax, true_vals_ax, posterior_mean_ax, cost_ax) = plt.subplots(4,1, figsize=(10, 10))
    x_ax.plot(max_xs)
    #x_ax.axhline(true_maximizer.item(), color='r')
    x_ax.set_title('Optimal x over iterations')
    true_vals_ax.plot(true_vals)
    true_vals_ax.axhline(true_max_val.item(), color='r')
    true_vals_ax.set_title('True objective value at highest fidelity at optimal xs')
    posterior_mean_ax.plot(max_means)
    posterior_mean_ax.set_title('Optimal objective posterior mean at highest fidelity over iterations')
    cost_ax.plot(cum_cost)
    cost_ax.set_title('Cumulative cost over iterations')

    plt.savefig('test_mtmf.pdf')

    return model, likelihood


def get_posterior_full_fid_max(model, likelihood, param_ranges = [0,10], num_test_pts = 101):
    test_x = torch.linspace(param_ranges[0], param_ranges[1], num_test_pts).unsqueeze(1)
    test_s = torch.ones(num_test_pts).unsqueeze(1)

    test_x_s = torch.cat((test_x, test_s),dim=1)

    predictions = likelihood(model(test_x_s))
    mean = predictions.mean
    max_mean = torch.max(mean[:, 0])
    maximizer = test_x[torch.argmax(mean[:,0])]

    return maximizer, max_mean

def get_true_full_fid_max(function, param_ranges = [0,10], num_test_pts = 101):
    test_x = torch.linspace(param_ranges[0], param_ranges[1], num_test_pts).unsqueeze(1)
    test_s = torch.ones(num_test_pts).unsqueeze(1)

    #test_x_s = torch.cat((test_x, test_s),dim=1)

    vals = function(test_x, test_s, noise_level=0)
    max_val = torch.max(vals[:, 0])
    maximizer = test_x[torch.argmax(vals[:,0])]

    return maximizer, max_val

if __name__ == "__main__":

    multitask_BO_trial(test_function_noisy, n_bo_iter=20)
    
    """

    fidelity = 0.5

    # TODO: inputs are to be replaced by hp configs
    train_x = torch.linspace(0.1, 20, 21)

    # TODO: outputs are to be replaced by performance metrics
    train_y = test_function_noisy(train_x, s=fidelity, noise_level = 0.2)

    model, likelihood = fit_multitask_GP_model(train_x, train_y, 2, training_iterations = 30)
    
    f, (y1_ax, y2_ax) = plt.subplots(1, 2, figsize=(8, 3))

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = torch.linspace(0.1,20,101)
        # return marginal distribution p(y|test_x) under the posterior model
        predictions = likelihood(model(test_x))
        print(predictions)
        mean = predictions.mean
        true = test_function_noisy(test_x,1, noise_level=0)
        lower, upper = predictions.confidence_region()

    print('shapes of train_x and train_y:', train_x.shape, train_y.shape)

     
    # Plot training data as black stars
    y1_ax.scatter(train_x.detach().numpy(), train_y[:, 0].detach().numpy(), color='grey', marker='*')
    # Predicted and true means
    y1_ax.plot(test_x.numpy(), true[:, 0].numpy(), 'r')
    y1_ax.plot(test_x.numpy(), mean[:, 0].numpy(), 'cyan')
    # Shade in confidence
    y1_ax.fill_between(test_x.numpy(), lower[:, 0].numpy(), upper[:, 0].numpy(), alpha=0.3)
    y1_ax.legend(['Observed Data', 'True value at s=1', 'GP Mean at s={}'.format(fidelity), 'Confidence'])
    y1_ax.set_title('Inferred f0')

    # Plot training data as black stars
    y2_ax.scatter(train_x.detach().numpy(), train_y[:, 1].detach().numpy(), color='grey', marker='*')
    # Predicted and true means
    y2_ax.plot(test_x.numpy(), true[:, 1].numpy(), 'r')
    y2_ax.plot(test_x.numpy(), mean[:, 1].numpy(), 'cyan')
    # Shade in confidence
    y2_ax.fill_between(test_x.numpy(), lower[:, 1].numpy(), upper[:, 1].numpy(), alpha=0.3)
    y2_ax.legend(['Observed Data', 'True value at s=1', 'GP Mean at s={}'.format(fidelity), 'Confidence'])
    y2_ax.set_title('Inferred f1')

    plt.savefig('test.pdf')

    """