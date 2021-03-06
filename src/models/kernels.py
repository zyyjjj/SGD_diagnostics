from pyparsing import java_style_comment
import torch
from torch import Tensor

from gpytorch.kernels import Kernel
from gpytorch.kernels.matern_kernel import MaternKernel
from botorch.models import MultiTaskGP
from botorch.models.kernels.exponential_decay import ExponentialDecayKernel

from gpytorch.kernels import ScaleKernel, MaternKernel, IndexKernel, ProductKernel
from torch.nn import ModuleList # is this needed?
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


class ModifiedIndexKernel(IndexKernel):
    def __init__(self, num_tasks, rank=1, prior=None, var_constraint=None, **kwargs):
        super().__init__(num_tasks, **kwargs)
        if prior is not None:
            self.register_prior(
                "IndexKernelPrior", 
                prior, 
                lambda m: m.covar_fac,
                lambda m, v: m._set_covar_fac(v))

    @property
    def covar_fac(self):
        return self.covar_factor

    @covar_fac.setter
    def covar_fac(self, value):
        self._set_covar_fac(value)

    def _set_covar_fac(self, value):
        # value is a num_tasks x 1 tensor
        self.initialize(covar_factor = value)



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
    


class IndicatorKernel(Kernel):
    
    '''
    Returns 1 if inputs align on the specified dimension, 0 otherwise.
    '''
    
    # to modify
    def __init__(self, dimension_to_align, **kwargs):
        super().__init__(**kwargs)
        self.dimension_to_align = dimension_to_align

    def forward(self, x1, x2, **kwargs):

        # aligned = list(map(int, x1[..., self.dimension_to_align] == x2[..., self.dimension_to_align]))

        output = torch.zeros((x1.shape[0], x2.shape[0]))
        for i in range(x1.shape[0]):
            for j in range(x2.shape[0]):
                output[i,j] = int(x1[i, self.dimension_to_align] == x2[j, self.dimension_to_align])
        
        return output



        
if __name__ == '__main__':

    x1 = torch.tensor(
        [[1,2,3],
        [1,2,4]]
    )

    x2 = torch.tensor(
        [[1,2,3],
        [1,2,5],
        [1,2,4]]
    )

    kernel = IndicatorKernel(2)

    print(kernel(x1, x2))
    print(kernel(x1, x2).evaluate())