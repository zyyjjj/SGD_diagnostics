import torch
from gpytorch.kernels.matern_kernel import MaternKernel

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