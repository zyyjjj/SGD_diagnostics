import torch
import os
import time
from torch import Tensor

from botorch.test_functions import Hartmann, SyntheticTestFunction, Branin

class ContextualHartmann6(Hartmann):
    # test function from BoTorch tutorial
    def __init__(self, num_tasks: int = 20, noise_std = None, negate = False):
        super().__init__(dim=6, noise_std = noise_std, negate = negate)
        self.task_range = torch.linspace(0, 1, num_tasks).unsqueeze(-1)
        self._bounds = [(0.0, 1.0) for _ in range(self.dim - 1)]
        self.bounds = torch.tensor(self._bounds).t()
        print('task_range shape: ', self.task_range.shape)
        
    def evaluate_true(self, X: Tensor) -> Tensor:
        # common shape of X: (number of data points x, input dimension) 
        # shape of batch_X: (number of data points x, 1, input dimension)
        # unsqueeze at -2 to make a dimension for tasks
        batch_X = X.unsqueeze(-2)
        batch_dims = X.ndim - 1
        
        expanded_task_range = self.task_range
        for _ in range(batch_dims):
            expanded_task_range = expanded_task_range.unsqueeze(0) # shape = (1,...,1,num_tasks,1)
        
        # construct task_range with shape = (n_datapts, num_tasks, 1)
        # repeat batch_X to get to shape (n_datapts, num_tasks, input dimension)
        # concatenate the two to get shape (n_datapts, num_tasks, input_dimension + 1)
        task_range = expanded_task_range.repeat(*X.shape[:-1], 1, 1).to(X)
        print(task_range.shape)
        concatenated_X = torch.cat(
            (batch_X.repeat(*[1]*batch_dims, self.task_range.shape[0], 1), task_range), dim=-1
        )

        # evaluate the concatenated input; output has shape (n_datapts, num_tasks)
        return super().evaluate_true(concatenated_X)
    

class TestProblem(SyntheticTestFunction):
    def __init__(self, dim: int = 2, num_tasks: int = 2, noise_std = None, negate = False):
        
        self.dim = dim
        self.task_range = torch.linspace(0, 1, num_tasks).unsqueeze(-1) # shape = (num_tasks, 1)
        self._bounds = [(0.0, 1.0) for _ in range(self.dim)]
        self._optimizers = None
        super().__init__(noise_std = noise_std, negate = negate)        
        self.bounds = torch.tensor(self._bounds).t()

        
    def evaluate_true(self, X: Tensor) -> Tensor:
        # common shape of X: (number of data points x, input dimension) 
        # shape of output: (number of data points, number of tasks)

        x = X[:,:-1]
        s = X[:,-1].unsqueeze(-1)

        output = torch.cat([
            (x * torch.exp(-x) + (10-x)*torch.exp(x-10)) * torch.sqrt(s) * 10,
            (x * torch.exp(-x) + (10-x)*torch.exp(x-10)) * 10 * 1/(2*torch.sqrt(s+0.000001))
        ], -1)  

        return output


class TestProblem_Branin(Branin):
    def __init__(self, dim: int = 2, num_tasks: int = 2, noise_std = None, negate = False):
        
        self.dim = dim
        self.task_range = torch.linspace(0, 1, num_tasks).unsqueeze(-1) # shape = (num_tasks, 1)
        super().__init__(noise_std = noise_std, negate = negate)        
        self.bounds = torch.tensor(self._bounds).t()
        
    def evaluate_true(self, X: Tensor) -> Tensor:
        # shape of X: (number of data points, input dimension) 
        # return shape: (number of data points, number of tasks)

        output = super().evaluate_true(X).unsqueeze(1)
        print(output.shape)
        output_shifted = super().evaluate_true(X*1.1).unsqueeze(1)
        output_multitask = torch.cat([output, output_shifted], -1)

        return output_multitask