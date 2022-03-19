import numpy as np
import torch


def process_multitask_data(X, y, num_checkpoints, add_last_col_X = False):
    """
    the goal is to transform (multi-dim input, multi-dim output) data into (multi-dim input, single-dim output) data
    input shapes: X: (num_trials * num_checkpoints) x (design_dim + 1) 
                      or (num_trials * num_checkpoints) x (design_dim + 2) ; 
                  y: (num_trials * num_checkpoints) x output_dim
    target shapes: X: (num_trials * num_checkpoints * output_dim) x (design_dim + 2) ;
                  y: (num_trials * num_checkpoints) x 1
    """

    num_trials = y.shape[0] // num_checkpoints
    num_outputs = y.shape[-1]
    
    X_repeat = X.repeat_interleave(num_outputs, 0)
    task_idx_repeat = torch.arange(0,num_outputs).unsqueeze(1).repeat(num_trials,1).repeat_interleave(num_checkpoints, 0)

    if add_last_col_X:
        new_X = torch.cat((X_repeat, task_idx_repeat), 1)
    else:
        new_X = torch.cat((X_repeat[:, :-1], task_idx_repeat), 1)
    
    new_y = y.flatten().unsqueeze(1) 

    print('expanded X shape {}, value{}'.format(new_X.shape, new_X))
    print('expanded y shape {}, value{}'.format(new_y.shape, new_y))

    return new_X, new_y

def expand_intermediate_fidelities(X, checkpoint_fidelities):
    """ 
    Expands the input to include intermediate fidelity values.
    INPUT shape: X: trials x (design_dim + 1); 
    OUTPUT shape: X: (trials x len(checkpoint_fidelities)) x (design_dim + 1)
    
    Example: 1 trial, 2-dim design, 1-dim fidelity, checkpoint_fidelities = [0.25, 0.5, 1] 
    Input: [[0.5, 0.5, 0.8]]
    Output: [[0.5, 0.5, 0.2], [0.5, 0.5, 0.4], [0.5, 0.5, 0.8]]
    """

    num_fidelities = len(checkpoint_fidelities)
    designs = X[:, :-1].repeat_interleave(num_fidelities, 0)

    fids = torch.kron(X[:,-1], torch.tensor(checkpoint_fidelities)).unsqueeze(1)

    return torch.cat((designs, fids), 1)


def get_task_covariance(model, X, num_outputs):
    covar_matrix = model.covar_module(X[:num_outputs], X[:num_outputs]).evaluate().detach().numpy()
    diag_inv_sqrt = np.zeros((num_outputs, num_outputs))
    for i in range(num_outputs):
        diag_inv_sqrt[i,i] = 1 / np.sqrt(covar_matrix[i,i])
    
    return diag_inv_sqrt @ covar_matrix @ diag_inv_sqrt
    
