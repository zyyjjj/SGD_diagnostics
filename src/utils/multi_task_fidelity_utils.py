import numpy as np
import torch


def covariance_to_correlation(matrix):
    matrix = torch.tensor(matrix) 
    diag_inv_sqrt = torch.diag(1/torch.sqrt(torch.diag(matrix)))   
    return diag_inv_sqrt @ matrix @ diag_inv_sqrt


def process_multitask_data(X, y, num_checkpoints, add_last_col_X = False):
    """
    the goal is to transform (multi-dim input, multi-dim output) data into (multi-dim input, single-dim output) data
    input shapes: X: (num_trials * num_checkpoints) x (design_dim + 1) 
                      or (num_trials * num_checkpoints) x (design_dim + 2) ; 
                  y: (num_trials * num_checkpoints) x output_dim
    target shapes: X: (num_trials * num_checkpoints * output_dim) x (design_dim + 2) ;
                  y: (num_trials * num_checkpoints * output_dim) x 1
    """

    num_trials = y.shape[0] // num_checkpoints
    num_outputs = y.shape[-1]
    # print('number of trials, ', num_trials, 'number of outputs, ', num_outputs)

    X_repeat = X.repeat_interleave(num_outputs, 0)
    # X_repeat = X.repeat(num_outputs, 1)
    # task_idx_repeat = torch.arange(0,num_outputs).unsqueeze(1).repeat(num_trials,1).repeat_interleave(num_checkpoints, 0)
    task_idx_repeat = torch.arange(0, num_outputs).unsqueeze(1).repeat(num_checkpoints, 1).repeat(num_trials, 1)

    if add_last_col_X:
        new_X = torch.cat((X_repeat, task_idx_repeat), 1)
    else:
        new_X = torch.cat((X_repeat[:, :-1], task_idx_repeat), 1)
    
    new_y = y.flatten().unsqueeze(1) 

    # print('expanded X shape {}, value{}'.format(new_X.shape, new_X))
    # print('expanded y shape {}, value{}'.format(new_y.shape, new_y))

    return new_X, new_y

def expand_intermediate_fidelities(X, checkpoint_fidelities, last_dim_is_task = True):
    """ 
    Expands the input to include intermediate fidelity values.
    INPUT shape: X: trials x (design_dim + 1); 
    OUTPUT shape: X: (trials x len(checkpoint_fidelities)) x (design_dim + 1)
    
    Example: 1 trial, 2-dim design, 1-dim fidelity, checkpoint_fidelities = [0.25, 0.5, 1] 
    Input: [[0.5, 0.5, 0.8]]
    Output: [[0.5, 0.5, 0.2], [0.5, 0.5, 0.4], [0.5, 0.5, 0.8]]
    """

    num_fidelities = len(checkpoint_fidelities)

    if not last_dim_is_task:
        # last dim is fidelity; this case only holds when intial samples are taken and no task column has been created
        fidelity_dim = -1
    else:
        # last dim is task; this holds for processing candidates returned by the acquisition function
        fidelity_dim = -2
    
    designs = X[:, :fidelity_dim].repeat_interleave(num_fidelities, 0)

    fids = torch.kron(X[:,fidelity_dim], torch.tensor(checkpoint_fidelities)).unsqueeze(1)

    print('fidelity checkpoints', fids)

    # does not include the task column; task column to be added in process_multitask_data()
    return torch.cat((designs, fids), 1)

# TODO: test correctness of the task-fidelity covariance!

def get_task_fidelity_covariance(model, X, num_outputs, num_fidelities):
    # from existing data, fetch an array of datapoints with identical designs and spread over different fidelities and tasks
    # because the kernel components over (design, fidelity, task) are separable, we use this to extract the task_fidelity covariance

    tasks = torch.arange(0, num_outputs)
    fidelities = torch.linspace(0.25, 1, num_fidelities) # generate num_fidelities evenly spaced points between 0 and 1

    tasks = tasks.unsqueeze(1).repeat_interleave(num_fidelities, 0)
    fidelities = fidelities.unsqueeze(1).repeat(num_outputs, 1)
    
    test_X = torch.cat((X[:(num_outputs*num_fidelities), :-2], fidelities, tasks), 1)
    # print(test_X, test_X.shape)

    covar_matrix = model.covar_module(test_X, test_X).evaluate().detach().numpy()
    diag_inv_sqrt = np.zeros((num_outputs*num_fidelities, num_outputs*num_fidelities))
    for i in range(num_outputs*num_fidelities):
        diag_inv_sqrt[i,i] = 1 / np.sqrt(covar_matrix[i,i])
    
    result = diag_inv_sqrt @ covar_matrix @ diag_inv_sqrt
    print('condition number of task-fidelity covariance: {}'.format(np.linalg.cond(result)))
    
    return covariance_to_correlation(result)
    

def get_task_covariance(model, X, num_outputs):
    #covar_matrix = model.covar_module(X[:num_outputs], X[:num_outputs]).evaluate().detach().numpy()

    B = model.covar_module.kernels[2].covar_factor.detach().numpy()
    v =  model.covar_module.kernels[2].raw_var_constraint.transform(
        model.covar_module.kernels[2].raw_var.detach()).numpy()
    covar_matrix = np.matmul(B, np.transpose(B) ) + np.diag(np.exp(v))

    diag_inv_sqrt = np.zeros((num_outputs, num_outputs))
    for i in range(num_outputs):
        diag_inv_sqrt[i,i] = 1 / np.sqrt(covar_matrix[i,i])
    
    result = diag_inv_sqrt @ covar_matrix @ diag_inv_sqrt

    print('task covariance hyperparameters: B {}, v {}'.format(B, v))

    print('condition number of task covariance: {}'.format(np.linalg.cond(result)))

    return covariance_to_correlation(result)


# TODO: tackle the issue that fidelity kernel is very badly conditioned
"""
K(x_1, x_2) = w + beta^alpha / (x_1 + x_2 + beta)^alpha.

    where `w` is an offset parameter, `beta` is a lenthscale parameter, and
    `alpha` is a power parameter.
"""


def get_fidelity_covariance(model, checkpoints = [0.25, 0.5, 1]):

    w = model.covar_module.kernels[1].raw_offset_constraint.transform(
        model.covar_module.kernels[1].raw_offset.detach()).item()
    beta = model.covar_module.kernels[1].raw_lengthscale_constraint.transform(
        model.covar_module.kernels[1].raw_lengthscale.detach()).item()
    alpha = model.covar_module.kernels[1].raw_power_constraint.transform(        
        model.covar_module.kernels[1].raw_power.detach()).item()

    print('fidelity kernel hyperparameters - raw: offset {}, lengthscale {}, power {}'.format(
        model.covar_module.kernels[1].raw_offset.item(),
        model.covar_module.kernels[1].raw_lengthscale.item(),
        model.covar_module.kernels[1].raw_power.item()
    ))
    print('fidelity kernel hyperparamters: offset {}, lengthscale {}, power {}'.format(w, beta, alpha))

    num_checkpoints = len(checkpoints)

    covar = np.zeros((num_checkpoints, num_checkpoints))
    for i in range(num_checkpoints):
        for j in range(i, num_checkpoints):
            covar[i,j] = w + beta**alpha / (checkpoints[i] + checkpoints[j] + beta)**alpha
            if j != i:
                covar[j,i] = w + beta**alpha / (checkpoints[i] + checkpoints[j] + beta)**alpha
    
    print('condition number of fidelity covariance: {}'.format(np.linalg.cond(covar)))

    return covariance_to_correlation(covar)

# TODO: make this generalizable
def print_kernel_hyperparams(model):
    state_dict = model.covar_module.state_dict()
    print('raw output scale', state_dict['raw_outputscale'])
    print('design raw lengthscale: ', state_dict['base_kernel.kernels.0.raw_lengthscale'])
    print('fidelity raw lengthscale: ', state_dict['base_kernel.kernels.1.raw_lengthscale'])
    print('fidelity raw power: ', state_dict['base_kernel.kernels.1.raw_power'])
    print('fidelity raw offset: ', state_dict['base_kernel.kernels.1.raw_offset'])
    print('task covar factor: ', state_dict['base_kernel.kernels.2.covar_factor'])
    print('task log raw variance: ', state_dict['base_kernel.kernels.2.raw_var'])
