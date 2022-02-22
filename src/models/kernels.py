import torch
from torch import Tensor

from gpytorch.kernels import Kernel
from gpytorch.kernels.matern_kernel import MaternKernel
from botorch.models import MultiTaskGP
from torch.nn import ModuleList # is this needed?
from gpytorch.kernels.index_kernel import IndexKernel
from gpytorch.kernels.scale_kernel import ScaleKernel
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.priors.torch_priors import GammaPrior
class ModifiedMaternKernel(MaternKernel):
    """
    Computes a covariance matrix based on the matern Kernel,
    rounding the value for integer-valued variables
    according to the approach in https://arxiv.org/pdf/1706.03673.pdf 
    """

    def __init__(self, is_int, **kwargs):
        super().__init__(**kwargs)
        # is_int is a boolean list indicating whether the ith variable is an integer
        self.is_int = is_int


    def process_before_forward(self, x1, x2):

        assert len(self.is_int) == x1.shape[-1],\
            'length of is_int {} and dimension -1 of input shape {} should align'.format(len(self.is_int), x1.shape)

        for i in range(x1.shape[-1]):
            if self.is_int[i]:
                x1[..., i] = torch.round(x1[..., i])
                x2[..., i] = torch.round(x2[..., i])

        return x1, x2

    def forward(self, x1, x2, **params):
        
        x1, x2 = self.process_before_forward(x1, x2)

        # Lesson learned: don't forget to add "return" when calling a method inherited from the parent class!!!
        return super().forward(x1, x2, **params)





class MisoKernel(Kernel):
    def __init__(self, base_kernel, num_tasks, **kwargs):
        super().__init__(**kwargs)
        self.kernels_list = ModuleList(base_kernel for i in range(num_tasks))
        self.num_tasks = num_tasks
    
    def compute_covar(self, x1, x2):
        # compute covariance between a pair of single data points
        x1_data = x1[..., :-1].unsqueeze(0)
        x2_data = x2[..., :-1].unsqueeze(0)
        x1_IS = x1[..., -1]
        x2_IS = x2[..., -1]
        covar = self.kernels_list[0](x1_data, x2_data)
        if x1_IS == x2_IS and x1_IS.item() != 0:
            covar += self.kernels_list[x1_IS](x1_data, x2_data)

        return covar

    def forward(self, X1, X2):
        # suppose last column is task index

        covar_matrix = Tensor(len(X1), len(X2))

        for i, pt1 in enumerate(X1):
            for j, pt2 in enumerate(X2):
                covar_matrix[i,j] = self.compute_covar(pt1, pt2).evaluate()
        
        return covar_matrix

class ProductKernel(Kernel):
    def __init__(self, data_dim, num_tasks, **kwargs):
        super().__init__(**kwargs)
        self.data_kernel = ScaleKernel(
            base_kernel=MaternKernel(
                nu=2.5, ard_num_dims=data_dim, lengthscale_prior=GammaPrior(3.0, 6.0)
            ),
            outputscale_prior=GammaPrior(2.0, 0.15),
        )
        self.task_kernel = IndexKernel(
            num_tasks=num_tasks
        )
        self.num_tasks = num_tasks
    
    def compute_covar(self, x1, x2):
        # compute covariance between a pair of single data points
        x1_data = x1[..., :-1].unsqueeze(0)
        x2_data = x2[..., :-1].unsqueeze(0)
        x1_task = x1[..., -1]
        x2_task = x2[..., -1]
        covar = self.data_kernel(x1_data, x2_data) * self.task_kernel(x1_task, x2_task)

        return covar

    def forward(self, X1, X2):
        # suppose last column is task index
        covar_matrix = Tensor(len(X1), len(X2))

        for i, pt1 in enumerate(X1):
            for j, pt2 in enumerate(X2):
                covar_matrix[i,j] = self.compute_covar(pt1, pt2).evaluate()
        
        return covar_matrix