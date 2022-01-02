import torch
from gpytorch.kernels.matern_kernel import MaternKernel

class ModifiedMaternKernel(MaternKernel):
    """
    Computes a covariance matrix based on the matern Kernel,
    rounding the value for integer-valued variables
    according to the approach in https://arxiv.org/pdf/1706.03673.pdf 
    """

    def __init__(self, is_int):
        super().__init__()
        # is_int is a boolean list indicating whether the ith variable is an integer
        self.is_int = is_int


    def process_before_forward(self, x1, x2):

        print(len(self.is_int), x1.shape)

        assert len(self.is_int) == x1.shape[-1],\
            'length of is_int and input dimension -1 should align'

        for i in range(x1.shape[-1]):
            if self.is_int:
                x1[:,i] = torch.round(x1[:i])
                x2[:,i] = torch.round(x2[:i])

        return x1, x2

    def forward(self, x1, x2, diag=False, **params):
        
        x1, x2 = self.process_before_forward(x1, x2)

        super().forward(self, x1, x2, diag=False, **params)