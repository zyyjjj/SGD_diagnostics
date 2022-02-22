import torch
import time
import matplotlib.pyplot as plt
from test_functions import *
from botorch.models import SingleTaskGP, KroneckerMultiTaskGP, FixedNoiseMultiTaskGP, MultiTaskGP
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.optim.fit import fit_gpytorch_torch
from botorch.sampling import IIDNormalSampler
from botorch.acquisition.objective import GenericMCObjective, LinearMCObjective

from gpytorch.mlls import ExactMarginalLogLikelihood




torch.random.manual_seed(2010)

if torch.cuda.is_available():
    torch.cuda.set_device("cuda:0")
tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}



def optimize_acqf_and_get_candidate(acq_func, bounds, batch_size):
    """Optimizes the acquisition function, and returns a new candidate and a noisy observation."""
    # optimize
    candidates, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=batch_size,
        num_restarts=10,
        raw_samples=512,  # used for intialization heuristic
        options={"batch_limit": 5, "maxiter": 200, "init_batch_limit": 5},
    )
    # observe new values 
    new_x = candidates.detach()
    return new_x

# TODO: change to cost-aware KG here
def construct_acqf(model, objective, num_samples, best_f):
    sampler = IIDNormalSampler(num_samples=num_samples)
    qEI = qExpectedImprovement(
        model=model, 
        best_f=best_f,
        sampler=sampler, 
        objective=objective,
    )
    return qEI


n_init = 20
n_steps = 20
batch_size = 3
num_samples = 128
n_trials = 3
verbose = True



if __name__ == '__main__':

    # TODO: modify objective (but what do you map non-main output to? posterior mean of main output?)
    # is this why EI is not suitable here? Do KG and PES require objective()?
    # Instead of explicitly using objective(), KG relies on the "project" operation



    problem = TestProblem(noise_std = 0.001, negate=True).to(**tkwargs)
    weights = torch.tensor([0.8, 0.2]).to(**tkwargs)
    objective = LinearMCObjective(weights)
    bounds = problem.bounds

    # TODO: record cost here
    mtgp_trial_objectives = []
    batch_trial_objectives = []
    rand_trial_objectives = []

    mtgp_trial_times, batch_trial_times, rand_trial_times = [], [], []

    for trial in range(n_trials):
        print('trial', trial)
        mtgp_trial_times.append([])
        batch_trial_times.append([])
        rand_trial_times.append([])

        init_x = (bounds[1] - bounds[0]) * torch.rand(n_init, bounds.shape[1], **tkwargs) + bounds[0]
        init_x = init_x.to(**tkwargs)
        init_y = problem(init_x)

        print('init_x device: {}, init_y device: {}'.format(init_x.device, init_y.device))
        print('init_x shape: {}, init_y shape: {}'.format(init_x.shape, init_y.shape))

        mtgp_train_x, mtgp_train_y = init_x, init_y
        batch_train_x, batch_train_y = init_x, init_y
        rand_x, rand_y = init_x, init_y

        objective_init = objective(init_y)
        objective_init.to(**tkwargs)
        print('objective() device: ', objective_init.device)

        best_value_mtgp = objective(init_y).max()
        best_value_batch = best_value_mtgp
        best_random = best_value_mtgp
        
        for iteration in range(n_steps):
            # we empty the cache to clear memory out
            torch.cuda.empty_cache()

            mtgp_t0 = time.time()
            mtgp = KroneckerMultiTaskGP(
                mtgp_train_x, 
                mtgp_train_y, 
            )
            mtgp_mll = ExactMarginalLogLikelihood(mtgp.likelihood, mtgp)
            fit_gpytorch_torch(mtgp_mll, options={"maxiter": 3000, "lr": 0.01, "disp": False})
            mtgp_acqf = construct_acqf(mtgp, objective, num_samples, best_value_mtgp)
            new_mtgp_x = optimize_acqf_and_get_candidate(mtgp_acqf, bounds, batch_size)
            mtgp_t1 = time.time()
            mtgp_trial_times[-1].append(mtgp_t1-mtgp_t0)

            # SingleTaskGP treats train_y as uncorrelated (sharing the same input data)
            # fits a separate GP to each task
            batch_t0 = time.time()
            batchgp = SingleTaskGP(
                batch_train_x, 
                batch_train_y, 
            )
            batch_mll = ExactMarginalLogLikelihood(batchgp.likelihood, batchgp)
            fit_gpytorch_torch(batch_mll, options={"maxiter": 3000, "lr": 0.01, "disp": False})
            batch_acqf = construct_acqf(batchgp, objective, num_samples, best_value_batch)
            new_batch_x = optimize_acqf_and_get_candidate(batch_acqf, bounds, batch_size)
            batch_t1 = time.time()
            batch_trial_times[-1].append(batch_t1-batch_t0)

            mtgp_train_x = torch.cat((mtgp_train_x, new_mtgp_x), dim=0)
            batch_train_x = torch.cat((batch_train_x, new_batch_x), dim=0)

            mtgp_train_y = torch.cat((mtgp_train_y, problem(new_mtgp_x)), dim=0)
            batch_train_y = torch.cat((batch_train_y, problem(new_batch_x)), dim=0)

            best_value_mtgp = objective(mtgp_train_y).max()
            best_value_batch = objective(batch_train_y).max()
            print('best values of mtgp and batch gp', best_value_mtgp, best_value_batch)
            
            rand_t0 = time.time()
            new_rand_x = (bounds[1] - bounds[0]) * torch.rand(batch_size, bounds.shape[1], **tkwargs) + bounds[0]
            rand_x = torch.cat((rand_x, new_rand_x))
            rand_y = torch.cat((rand_y, problem(new_rand_x)))
            best_random = objective(rand_y).max()
            rand_t1 = time.time()
            rand_trial_times[-1].append(rand_t1-rand_t0)
            print('best random', best_random)

            if verbose:
                print(
                    f"\nBatch {iteration:>2}: best_value (random, mtgp, batch) = "
                    f"({best_random:>4.2f}, {best_value_mtgp:>4.2f}, {best_value_batch:>4.2f}), "
                    f"batch time = {batch_t1-batch_t0:>4.2f}, mtgp time = {mtgp_t1-mtgp_t0:>4.2f}", end=""
                )
            else:
                print(".", end="")
                
        mtgp_trial_objectives.append(objective(mtgp_train_y).detach().cpu())
        batch_trial_objectives.append(objective(batch_train_y).detach().cpu())
        rand_trial_objectives.append(objective(rand_y).detach().cpu())
    
    # cummax() returns the max-so-far in the array
    mtgp_results = torch.stack(mtgp_trial_objectives)[:, n_init:].cummax(1).values
    batch_results = torch.stack(batch_trial_objectives)[:, n_init:].cummax(1).values
    random_results = torch.stack(rand_trial_objectives)[:, n_init:].cummax(1).values

    plt.figure(figsize=(10,8))

    plt.plot(mtgp_results.mean(0))
    plt.fill_between(
        torch.arange(n_steps * batch_size), 
        mtgp_results.mean(0) - 2. * mtgp_results.std(0) / (n_trials ** 0.5),
        mtgp_results.mean(0) + 2. * mtgp_results.std(0) / (n_trials ** 0.5),
        alpha = 0.3, label = "MTGP",
    )

    plt.plot(batch_results.mean(0))
    plt.fill_between(
        torch.arange(n_steps * batch_size), 
        batch_results.mean(0) - 2. * batch_results.std(0) / (n_trials ** 0.5),
        batch_results.mean(0) + 2. * batch_results.std(0) / (n_trials ** 0.5),
        alpha = 0.3, label = "Batch"
    )

    plt.plot(random_results.mean(0))
    plt.fill_between(
        torch.arange(n_steps * batch_size), 
        random_results.mean(0) - 2. * random_results.std(0) / (n_trials ** 0.5),
        random_results.mean(0) + 2. * random_results.std(0) / (n_trials ** 0.5),
        alpha = 0.3, label = "Random"
    )

    plt.legend(loc = "lower right", fontsize = 15)
    plt.xlabel("Number of Function Queries")
    plt.ylabel("Best Objective Achieved")
    plt.title('Multi-task test problem 1/2*(Branin(x) + Branin(x*1.1))')
    plt.savefig('Feb19_Branin.pdf')