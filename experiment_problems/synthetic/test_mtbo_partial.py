import torch
import os
import time
from torch import Tensor
from botorch import fit_gpytorch_model
from botorch.test_functions import SyntheticTestFunction
from botorch.models import SingleTaskGP, KroneckerMultiTaskGP, MultiTaskGP
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch.optim import optimize_acqf
from botorch.optim.fit import fit_gpytorch_torch
from botorch.acquisition.objective import GenericMCObjective, LinearMCObjective
from botorch.models.cost import AffineFidelityCostModel
from botorch.acquisition.cost_aware import InverseCostWeightedUtility
from botorch.acquisition import PosteriorMean
from botorch.acquisition.utils import project_to_target_fidelity
from botorch.acquisition.knowledge_gradient import qMultiFidelityKnowledgeGradient
from botorch.acquisition.fixed_feature import FixedFeatureAcquisitionFunction
from botorch.optim.initializers import gen_one_shot_kg_initial_conditions
from botorch.optim.optimize import optimize_acqf_mixed
import pdb

SMOKE_TEST = os.environ.get("SMOKE_TEST")

torch.random.manual_seed(2010)

if torch.cuda.is_available():
    torch.cuda.set_device("cuda:0")
tkwargs = {
    "dtype": torch.double,
    "device": torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
}

class TestProblem(SyntheticTestFunction):
    def __init__(self, dim: int = 3, noise_std = None, negate = False):
        
        self.dim = dim
        #self.task_range = torch.linspace(0, 1, num_tasks).unsqueeze(-1) # shape = (num_tasks, 1)
        self._bounds = [(0.0, 1.0) for _ in range(self.dim)]
        self._optimizers = None
        super().__init__(noise_std = noise_std, negate = negate)        
        self.bounds = torch.tensor(self._bounds).t()

        
    def evaluate_true(self, X: Tensor) -> Tensor:

        x = X[:,:-2]
        s = X[:,-2].unsqueeze(-1)
        task = X[:, -1].unsqueeze(-1).long()

        output = torch.cat([
            (x * torch.exp(-x) + (10-x)*torch.exp(x-10)) * 10 * 1/(2*torch.sqrt(s+0.000001)),
            (x * torch.exp(-x) + (10-x)*torch.exp(x-10)) * torch.sqrt(s) * 10
        ], -1)

        return torch.gather(output, 1, task.view(-1,1))

def project(X):
    return project_to_target_fidelity(X=X, target_fidelities=target_fidelities)

def get_mfkg(model):
    
    curr_val_acqf = FixedFeatureAcquisitionFunction(
        acq_function=PosteriorMean(model),
        d=3,
        columns=[2],
        values=[1],
    )
    
    pdb.set_trace()
    _, current_value = optimize_acqf(
        acq_function=curr_val_acqf,
        bounds=bounds[:,:-1],
        q=1,
        num_restarts=10 if not SMOKE_TEST else 2,
        raw_samples=1024 if not SMOKE_TEST else 4,
        options={"batch_limit": 10, "maxiter": 200},
    )
    # effective dim is 2 in gen batch IC
    # n = 1024
    #  X_rnd.shape = torch.Size([1024, 1, 2])
    # d_prime = 2, d_f = 1

        
    return qMultiFidelityKnowledgeGradient(
        model=model,
        num_fantasies=128 if not SMOKE_TEST else 2,
        current_value=current_value,
        cost_aware_utility=cost_aware_utility,
        project=project,
    )

def optimize_mfkg_and_get_observation(mfkg_acqf):
    """Optimizes MFKG and returns a new candidate, observation, and cost."""
    
    X_init = gen_one_shot_kg_initial_conditions(
        acq_function = mfkg_acqf,
        bounds=bounds,
        q=4,
        num_restarts=10,
        raw_samples=512,
    )
    candidates, _ = optimize_acqf_mixed(
        acq_function=mfkg_acqf,
        bounds=bounds,
        fixed_features_list=[{2: 0}, {2: 1}],
        q=4,
        num_restarts=NUM_RESTARTS,
        raw_samples=RAW_SAMPLES,
        batch_initial_conditions=X_init,
        options={"batch_limit": 5, "maxiter": 200},
    )
    # observe new values
    cost = cost_model(candidates).sum()
    new_x = candidates.detach()
    new_obj = problem(new_x).unsqueeze(-1)
    print(f"candidates:\n{new_x}\n")
    print(f"observations:\n{new_obj}\n\n")
    return new_x, new_obj, cost

def generate_initial_data(data_dim, task_dim=1, n=16):
    # generate training data
    train_x = torch.rand(n, data_dim, **tkwargs)
    train_f = fidelities[torch.randint(len(fidelities), (n,task_dim))]
    train_x_full = torch.cat((train_x, train_f), dim=1)
    train_obj = problem(train_x_full) # add output dimension
    return train_x_full, train_obj
    
def initialize_model(train_x, train_obj):
    # define a surrogate model suited for a "training data"-like fidelity parameter
    # in dimension 6, as in [2]
    # model = SingleTaskMultiFidelityGP(
    #     train_x, 
    #     train_obj, 
    #     outcome_transform=Standardize(m=1),
    #     data_fidelity=6
    # )   


    model = MultiTaskGP(train_x, train_obj, task_feature = 2, output_tasks = [1])
    
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    return mll, model

n_init = 20
n_steps = 20
batch_size = 3
num_samples = 128
n_trials = 3
verbose = True

fidelities = torch.tensor([0, 1], **tkwargs)
target_fidelities = {2: 1.0}

NUM_RESTARTS = 10 if not SMOKE_TEST else 2
RAW_SAMPLES = 512 if not SMOKE_TEST else 4


if __name__ == '__main__':

    problem = TestProblem(noise_std = 0.001, negate=True).to(**tkwargs)
    weights = torch.tensor([0.8, 0.2]).to(**tkwargs)
    objective = LinearMCObjective(weights)
    bounds = problem.bounds

    cost_model = AffineFidelityCostModel(fidelity_weights = {2: 10}, fixed_cost=1)
    cost_aware_utility = InverseCostWeightedUtility(cost_model=cost_model)

    train_x, train_obj = generate_initial_data(2, 1, n=16)

    cumulative_cost = 0.0
    N_ITER = 3 if not SMOKE_TEST else 1

    for iter in range(N_ITER):
        print('iteration {}'.format(iter))
        mll, model = initialize_model(train_x, train_obj)
        fit_gpytorch_model(mll)
        # pdb.set_trace()
        mfkg_acqf = get_mfkg(model)
        new_x, new_obj, cost = optimize_mfkg_and_get_observation(mfkg_acqf)
        train_x = torch.cat([train_x, new_x])
        train_obj = torch.cat([train_obj, new_obj])
        cumulative_cost += cost


    # mtgp_trial_objectives = []
    # batch_trial_objectives = []
    # rand_trial_objectives = []

    # mtgp_trial_times, batch_trial_times, rand_trial_times = [], [], []

    # for trial in range(n_trials):
    #     print('trial', trial)
    #     mtgp_trial_times.append([])
    #     batch_trial_times.append([])
    #     rand_trial_times.append([])

    #     init_x = (bounds[1] - bounds[0]) * torch.rand(n_init, bounds.shape[1], **tkwargs) + bounds[0]
    #     init_x = init_x.to(**tkwargs)
    #     init_y = problem(init_x)

    #     print('init_x device: {}, init_y device: {}'.format(init_x.device, init_y.device))
    #     print('init_x shape: {}, init_y shape: {}'.format(init_x.shape, init_y.shape))

    #     mtgp_train_x, mtgp_train_y = init_x, init_y
    #     batch_train_x, batch_train_y = init_x, init_y
    #     rand_x, rand_y = init_x, init_y

    #     objective_init = objective(init_y)
    #     objective_init.to(**tkwargs)
    #     print('objective() device: ', objective_init.device)

    #     best_value_mtgp = objective(init_y).max()
    #     best_value_batch = best_value_mtgp
    #     best_random = best_value_mtgp
        
    #     for iteration in range(n_steps):
    #         # we empty the cache to clear memory out
    #         torch.cuda.empty_cache()

    #         mtgp_t0 = time.time()
    #         mtgp = KroneckerMultiTaskGP(
    #             mtgp_train_x, 
    #             mtgp_train_y, 
    #         )
    #         mtgp_mll = ExactMarginalLogLikelihood(mtgp.likelihood, mtgp)
    #         fit_gpytorch_torch(mtgp_mll, options={"maxiter": 3000, "lr": 0.01, "disp": False})
    #         mtgp_acqf = construct_acqf(mtgp, objective, num_samples, best_value_mtgp)
    #         new_mtgp_x = optimize_acqf_and_get_candidate(mtgp_acqf, bounds, batch_size)
    #         mtgp_t1 = time.time()
    #         mtgp_trial_times[-1].append(mtgp_t1-mtgp_t0)

    #         # SingleTaskGP treats train_y as uncorrelated (sharing the same input data)
    #         # fits a separate GP to each task
    #         batch_t0 = time.time()
    #         batchgp = SingleTaskGP(
    #             batch_train_x, 
    #             batch_train_y, 
    #         )
    #         batch_mll = ExactMarginalLogLikelihood(batchgp.likelihood, batchgp)
    #         fit_gpytorch_torch(batch_mll, options={"maxiter": 3000, "lr": 0.01, "disp": False})
    #         batch_acqf = construct_acqf(batchgp, objective, num_samples, best_value_batch)
    #         new_batch_x = optimize_acqf_and_get_candidate(batch_acqf, bounds, batch_size)
    #         batch_t1 = time.time()
    #         batch_trial_times[-1].append(batch_t1-batch_t0)

    #         mtgp_train_x = torch.cat((mtgp_train_x, new_mtgp_x), dim=0)
    #         batch_train_x = torch.cat((batch_train_x, new_batch_x), dim=0)

    #         mtgp_train_y = torch.cat((mtgp_train_y, problem(new_mtgp_x)), dim=0)
    #         batch_train_y = torch.cat((batch_train_y, problem(new_batch_x)), dim=0)

    #         best_value_mtgp = objective(mtgp_train_y).max()
    #         best_value_batch = objective(batch_train_y).max()
    #         print('best values of mtgp and batch gp', best_value_mtgp, best_value_batch)
            
    #         rand_t0 = time.time()
    #         new_rand_x = (bounds[1] - bounds[0]) * torch.rand(batch_size, bounds.shape[1], **tkwargs) + bounds[0]
    #         rand_x = torch.cat((rand_x, new_rand_x))
    #         rand_y = torch.cat((rand_y, problem(new_rand_x)))
    #         best_random = objective(rand_y).max()
    #         rand_t1 = time.time()
    #         rand_trial_times[-1].append(rand_t1-rand_t0)
    #         print('best random', best_random)

    #         if verbose:
    #             print(
    #                 f"\nBatch {iteration:>2}: best_value (random, mtgp, batch) = "
    #                 f"({best_random:>4.2f}, {best_value_mtgp:>4.2f}, {best_value_batch:>4.2f}), "
    #                 f"batch time = {batch_t1-batch_t0:>4.2f}, mtgp time = {mtgp_t1-mtgp_t0:>4.2f}", end=""
    #             )
    #         else:
    #             print(".", end="")
                
    #     mtgp_trial_objectives.append(objective(mtgp_train_y).detach().cpu())
    #     batch_trial_objectives.append(objective(batch_train_y).detach().cpu())
    #     rand_trial_objectives.append(objective(rand_y).detach().cpu())
    
    # # cummax() returns the max-so-far in the array
    # mtgp_results = torch.stack(mtgp_trial_objectives)[:, n_init:].cummax(1).values
    # batch_results = torch.stack(batch_trial_objectives)[:, n_init:].cummax(1).values
    # random_results = torch.stack(rand_trial_objectives)[:, n_init:].cummax(1).values

    # plt.figure(figsize=(10,8))

    # plt.plot(mtgp_results.mean(0))
    # plt.fill_between(
    #     torch.arange(n_steps * batch_size), 
    #     mtgp_results.mean(0) - 2. * mtgp_results.std(0) / (n_trials ** 0.5),
    #     mtgp_results.mean(0) + 2. * mtgp_results.std(0) / (n_trials ** 0.5),
    #     alpha = 0.3, label = "MTGP",
    # )

    # plt.plot(batch_results.mean(0))
    # plt.fill_between(
    #     torch.arange(n_steps * batch_size), 
    #     batch_results.mean(0) - 2. * batch_results.std(0) / (n_trials ** 0.5),
    #     batch_results.mean(0) + 2. * batch_results.std(0) / (n_trials ** 0.5),
    #     alpha = 0.3, label = "Batch"
    # )

    # plt.plot(random_results.mean(0))
    # plt.fill_between(
    #     torch.arange(n_steps * batch_size), 
    #     random_results.mean(0) - 2. * random_results.std(0) / (n_trials ** 0.5),
    #     random_results.mean(0) + 2. * random_results.std(0) / (n_trials ** 0.5),
    #     alpha = 0.3, label = "Random"
    # )

    # plt.legend(loc = "lower right", fontsize = 15)
    # plt.xlabel("Number of Function Queries")
    # plt.ylabel("Best Objective Achieved")
    # plt.title('Multi-task test problem 1/2*(Branin(x) + Branin(x*1.1))')
    # plt.savefig('Feb19_Branin.pdf')