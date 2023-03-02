import gpytorch


#--------------------------------------------#
#-------------class ExactGPModel-------------#
#--------------------------------------------#
class ExactGPModel(gpytorch.models.ExactGP):
    """Class used to define GP (cf GP.py)."""

    def __init__(self, x_train, y_train, likelihood, kernel):
        super(ExactGPModel, self).__init__(x_train, y_train, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

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
            print("[GP.py] error, unknown argument value for kernel")
            assert False
        self.covar_module = gpytorch.kernels.ScaleKernel(gpy_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
