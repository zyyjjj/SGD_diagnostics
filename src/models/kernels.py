import torch
from torch import Tensor

from gpytorch.kernels import Kernel
from gpytorch.kernels.matern_kernel import MaternKernel
from botorch.models import MultiTaskGP
from botorch.models.kernels.exponential_decay import ExponentialDecayKernel

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

    def forward(self, X1, X2, **params):
        # suppose last column is task index

        covar_matrix = Tensor(len(X1), len(X2))

        for i, pt1 in enumerate(X1):
            for j, pt2 in enumerate(X2):
                covar_matrix[i,j] = self.compute_covar(pt1, pt2).evaluate()
        
        return covar_matrix

class InputTaskProductKernel(Kernel):
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

    def forward(self, X1, X2, **params):
        # suppose last column is task index
        covar_matrix = Tensor(len(X1), len(X2))

        for i, pt1 in enumerate(X1):
            for j, pt2 in enumerate(X2):
                covar_matrix[i,j] = self.compute_covar(pt1, pt2).evaluate()
        
        return covar_matrix


class InvertedExponentialDecayKernel(ExponentialDecayKernel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, x1, x2, **params):
        offset = self.offset
        power = self.power
        if not params.get("diag", False):
            offset = offset.unsqueeze(-1)  # unsqueeze enables batch evaluation
            power = power.unsqueeze(-1)  # unsqueeze enables batch evaluation
        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)
        diff = self.covar_dist(x1_, -x2_, **params)
        res = offset + 1 + (diff + 1).pow(-power) - (x1_ + 1).pow(-power) - (x2_ + 1).pow(-power)
        
        return res



# class InputFidelityTaskProductKernel(Kernel):
    
#     def __init__(self, data_dims, fidelity_col, task_col, num_tasks, **kwargs):
#         super().__init__(**kwargs)
#         # self.data_kernel = ScaleKernel(
#         #     base_kernel=MaternKernel(
#         #         nu=2.5, ard_num_dims=data_dims, lengthscale_prior=GammaPrior(3.0, 6.0)
#         #     ),
#         #     outputscale_prior=GammaPrior(2.0, 0.15),
#         # )
#         self.data_kernel = MaternKernel(
#                 nu=2.5, ard_num_dims=data_dims, lengthscale_prior=GammaPrior(3.0, 6.0)
#             )
#         self.task_kernel = IndexKernel(
#             num_tasks=num_tasks
#         )
#         self.fidelity_kernel = ExponentialDecayKernel()
#         self.num_tasks = num_tasks
#         self.data_dims = data_dims
#         self.fidelity_col = fidelity_col
#         self.task_col = task_col
    
#     def compute_covar(self, x1, x2):
#         # compute covariance between a pair of single data points

#         x1_data = x1[..., :self.data_dims]
#         x2_data = x2[..., :self.data_dims]
#         if len(x1_data.shape) < 2:
#             x1_data = x1_data.unsqueeze(-2)
#         if len(x2_data.shape) < 2:
#             x2_data = x2_data.unsqueeze(-2)

#         x1_task = x1[..., self.data_dims].unsqueeze(-1)
#         x2_task = x2[..., self.data_dims].unsqueeze(-1)
#         x1_fidelity = x1[..., self.fidelity_col].unsqueeze(-1)
#         x2_fidelity = x2[..., self.fidelity_col].unsqueeze(-1)

#         print('data, task, fidelity components shape: ', x1_data.shape, x2_data.shape, x1_task.shape, x2_task.shape, \
#             x1_fidelity.shape, x2_fidelity.shape)

#         # covar = self.data_kernel(x1_data, x2_data).evaluate() *\
#         #         self.task_kernel(x1_task, x2_task).evaluate() *\
#         #         self.fidelity_kernel(x1_fidelity, x2_fidelity).evaluate()
        
#         covar1 = self.data_kernel(x1_data, x2_data)#.evaluate()
#         covar2 = self.task_kernel(x1_task, x2_task)#.evaluate()
#         covar3 = self.fidelity_kernel(x1_fidelity, x2_fidelity)#.evaluate()

#         print(covar1, covar2, covar3)

#         covar = covar1 * covar2 * covar3

#         return covar.evaluate()

#     def forward(self, X1, X2, **params):

#         covar_matrix = Tensor(len(X1), len(X2))

#         for i, pt1 in enumerate(X1):
#             for j, pt2 in enumerate(X2):
#                 covar_matrix[i,j] = self.compute_covar(pt1, pt2)#.evaluate()
        
#         return covar_matrix

# I'm messing up the dimensions here very likely
# or: just use the ProductKernel in gpytorch, set three kernels with their respective active_dims
class InputFidelityTaskProductKernel(Kernel):
    
    def __init__(self, data_dims, fidelity_col, task_col, num_tasks, **kwargs):
        super().__init__(**kwargs)
        # self.data_kernel = ScaleKernel(
        #     base_kernel=MaternKernel(
        #         nu=2.5, ard_num_dims=data_dims, lengthscale_prior=GammaPrior(3.0, 6.0)
        #     ),
        #     outputscale_prior=GammaPrior(2.0, 0.15),
        # )
        self.data_kernel = MaternKernel(
                nu=2.5, ard_num_dims=data_dims, lengthscale_prior=GammaPrior(3.0, 6.0)
            )
        self.task_kernel = IndexKernel(
            num_tasks=num_tasks
        )
        self.fidelity_kernel = ExponentialDecayKernel()
        self.num_tasks = num_tasks
        self.data_dims = data_dims
        self.fidelity_col = fidelity_col
        self.task_col = task_col
    
    def compute_covar(self, x1, x2):
        # compute covariance between a pair of single data points

        x1_data = x1[..., :self.data_dims]
        x2_data = x2[..., :self.data_dims]
        if len(x1_data.shape) < 2:
            x1_data = x1_data.unsqueeze(-2)
        if len(x2_data.shape) < 2:
            x2_data = x2_data.unsqueeze(-2)

        x1_task = x1[..., self.data_dims].unsqueeze(-1)
        x2_task = x2[..., self.data_dims].unsqueeze(-1)
        x1_fidelity = x1[..., self.fidelity_col].unsqueeze(-1)
        x2_fidelity = x2[..., self.fidelity_col].unsqueeze(-1)

        print('data, task, fidelity components shape: ', x1_data.shape, x2_data.shape, x1_task.shape, x2_task.shape, \
            x1_fidelity.shape, x2_fidelity.shape)

        # covar = self.data_kernel(x1_data, x2_data).evaluate() *\
        #         self.task_kernel(x1_task, x2_task).evaluate() *\
        #         self.fidelity_kernel(x1_fidelity, x2_fidelity).evaluate()
        
        covar1 = self.data_kernel(x1_data, x2_data)#.evaluate()
        covar2 = self.task_kernel(x1_task, x2_task)#.evaluate()
        covar3 = self.fidelity_kernel(x1_fidelity, x2_fidelity)#.evaluate()

        print(covar1, covar2, covar3)

        covar = covar1 * covar2 * covar3

        return covar.evaluate()

    def forward(self, X1, X2, **params):

        covar_matrix = Tensor(len(X1), len(X2))

        for i, pt1 in enumerate(X1):
            for j, pt2 in enumerate(X2):
                covar_matrix[i,j] = self.compute_covar(pt1, pt2)#.evaluate()
        
        return covar_matrix
