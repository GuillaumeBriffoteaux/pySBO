import gpytorch

#------------------------------------------------#
#-------------class MultitaskGPModel-------------#
#------------------------------------------------#
class MultitaskGPModel(gpytorch.models.ExactGP):
    """Class used to define GP_MO (cf GP_MO.py)"""
    
    def __init__(self, x_train, y_train, likelihood, kernel):
        super(MultitaskGPModel, self).__init__(x_train, y_train, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(gpytorch.means.ZeroMean(), num_tasks=y_train.shape[1])

        dim_x = list(x_train.size())[1]
        if kernel=="rbf":
            gpy_kernel = gpytorch.kernels.RBFKernel()
        elif kernel=="matern2.5":
            gpy_kernel = gpytorch.kernels.MaternKernel(nu=2.5)
        elif kernel=="matern1.5":
            gpy_kernel = gpytorch.kernels.MaternKernel(nu=1.5)
        elif kernel=="sm2":
            gpy_kernel = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=2, ard_num_dims=dim_x)
        elif kernel=="sm4":
            gpy_kernel = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=dim_x)
        else:
            print("[GP_MO.py] error, unknown argument value for kernel")
            assert False
        self.covar_module = gpytorch.kernels.MultitaskKernel(gpy_kernel, num_tasks=y_train.shape[1], rank=1)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
