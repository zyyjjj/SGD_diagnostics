import math
import torch
import gpytorch
import os
from matplotlib import pyplot as plt


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, num_tasks, likelihood):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks
        )
        # the MultitaskKernel of GPytorch uses the product between input and task kernels
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
    # because model() is defined usign likelihood
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)  

    # marginal log likelihood
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

    return model, likelihood


def test_noisy(x,s,noise_level=0.2):
    output = torch.stack([
        x/(1+x) + (1-s) * torch.sin(2*math.pi * x)/x + torch.randn(x.size()) * noise_level,
        1+1/((1+x)**2) + (1-s) * (2*math.pi * torch.cos(2*math.pi*x)*x - torch.sin(2*math.pi*x))/(x**2) + torch.randn(x.size()) * noise_level
    ], -1)

    return output

# Feb 11 TODO
# make the test function more complicated
# make the first component be the one we want to maximize
# set the second component to a noisy evaluation of the gradient of the first component wrt x
# use a GP on (x, task, single fidelity) to model the objective; try different kernels (one big, or product)
# set cost to be higher for higher fidelities
# forget acquisition function now, implement random search first 
# immediate goal: does the GP recover the function structure well?


# TODO:
# design an acquisition function, taking into consideration
    # multiple tasks, each task with potentially multiple fidelities
    # some observations come for free, some at extra cost
# optimize acqf and suggest new *point*
# what are numerical methods for dealing with large scale inference? low rank approximations?


if __name__ == "__main__":

    # TODO: inputs are to be replaced by hp configs
    train_x = torch.linspace(0.1, 20, 501)

    # TODO: outputs are to be replaced by performance metrics
    train_y = test_noisy(train_x,0.3,0.1)

    model, likelihood = fit_multitask_GP_model(train_x, train_y, 2, training_iterations = 30)

    f, (y1_ax, y2_ax) = plt.subplots(1, 2, figsize=(8, 3))

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        test_x = torch.linspace(0.1,20,101)
        # return marginal distribution p(y|test_x) under the posterior model
        predictions = likelihood(model(test_x))
        print(predictions)
        mean = predictions.mean
        true = test_noisy(test_x,1, noise_level=0)
        lower, upper = predictions.confidence_region()

    print('shapes of train_x and train_y:', train_x.shape, train_y.shape)

    # Plot training data as black stars
    y1_ax.scatter(train_x.detach().numpy(), train_y[:, 0].detach().numpy(), color='grey', marker='*')
    # Predicted and true means
    y1_ax.plot(test_x.numpy(), true[:, 0].numpy(), 'r')
    y1_ax.plot(test_x.numpy(), mean[:, 0].numpy(), 'cyan')
    # Shade in confidence
    y1_ax.fill_between(test_x.numpy(), lower[:, 0].numpy(), upper[:, 0].numpy(), alpha=0.3)
    #y1_ax.set_ylim([-3, 3])
    y1_ax.legend(['Observed Data', 'Noiseless', 'Mean', 'Confidence'])
    y1_ax.set_title('Observed Values (Likelihood)')

    # Plot training data as black stars
    y2_ax.scatter(train_x.detach().numpy(), train_y[:, 1].detach().numpy(), color='grey', marker='*')
    # Predicted and true means
    y2_ax.plot(test_x.numpy(), true[:, 1].numpy(), 'r')
    y2_ax.plot(test_x.numpy(), mean[:, 1].numpy(), 'cyan')
    # Shade in confidence
    y2_ax.fill_between(test_x.numpy(), lower[:, 1].numpy(), upper[:, 1].numpy(), alpha=0.3)
    #y2_ax.set_ylim([-3, 3])
    y2_ax.legend(['Observed Data', 'Noiseless', 'Mean', 'Confidence'])
    y2_ax.set_title('Observed Values (Likelihood)')

    plt.show(block=True)
    plt.savefig('test.pdf')