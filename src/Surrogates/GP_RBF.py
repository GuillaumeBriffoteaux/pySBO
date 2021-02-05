import numpy as np
import time
import pickle
import gpytorch
import torch
from sklearn.metrics import mean_squared_error

from Surrogates.Surrogate import Surrogate


#---------------------------Defining Gaussian Process RBF kernel---------------------------#
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, x_train, y_train, likelihood):
        super(ExactGPModel, self).__init__(x_train, y_train, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


#--------------------------------------#
#-------------class GP_RBF-------------#
#--------------------------------------#
class GP_RBF(Surrogate):
    """Class for Gaussian Process model with Radial Basis kernel function (mono dimensional targets only).

    :param f_sim_archive: filename where are stored the past simulated individuals
    :type f_sim_archive: str
    :param pb: problem the surrogate is associated with
    :type pb: Problem
    :param n_train_samples: number of training samples to extract from the end of `f_sim_archive`, if float('inf') all the samples from `f_sim_archive` are considered
    :type n_train_samples: positive int or inf, not zero
    :param f_train_log: filename where will be recorded training log
    :type f_train_log: str
    :param f_trained_model: filename where will be recorded the trained surrogate model
    :type f_trained_model: str
    :param likelihood: likelihood function
    :type likelihood: gpytorch.likelihoods.GaussianLikelihood
    :param model: Gaussian Process model
    :type model: gpytorch.models.ExactGP
    """
    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#
 
    #-------------__init__-------------#
    def __init__(self, f_sim_archive, pb, n_train_samples, f_train_log, f_trained_model):
        Surrogate.__init__(self, f_sim_archive, pb, n_train_samples, f_train_log, f_trained_model)
        self.__likelihood = gpytorch.likelihoods.GaussianLikelihood()
        (x_train, y_train) = self.load_sim_archive()
        x_train = torch.from_numpy(x_train).float()
        y_train = torch.from_numpy(y_train.flatten()).float()
        self.__model = ExactGPModel(x_train, y_train, self.__likelihood)

    #-------------__del__-------------#
    def __del__(self):
        Surrogate.__del__(self)
        del self.__likelihood
        del self.__model

    #-------------__str__-------------#
    def __str__(self):
        return "Gaussian Process RBF Kernel\n  training set filename: "+self.f_sim_archive+"\n  associated problem: {"+self.pb.__str__()+"}\n  training set size = "+str(self.n_train_samples)+"\n  training log filename: "+self.f_train_log+"\n  trained model saved in "+self.f_trained_model


    #---------------------------------------------#
    #-------------getters and setters-------------#
    #---------------------------------------------#

    #-------------_get_likelihood-------------#
    def _get_likelihood(self):
        print("[GP_RBF.py] Impossible to modify the likelihood")
        return None

    #-------------_set_likelihood-------------#
    def _set_likelihood(self,new_likelihood):
        print("[GP_RBF.py] Impossible to modify the likelihood")

    #-------------_del_likelihood-------------#
    def _del_likelihood(self):
        print("[GP_RBF.py] Impossible to delete the likelihood")

    #-------------_get_model-------------#
    def _get_model(self):
        print("[GP_RBF.py] Impossible to modify the model")
        return None

    #-------------_set_model-------------#
    def _set_model(self,new_model):
        print("[GP_RBF.py] Impossible to modify the model")

    #-------------_del_model-------------#
    def _del_model(self):
        print("[GP_RBF.py] Impossible to delete the model")

    #-------------property-------------#
    likelihood=property(_get_likelihood, _set_likelihood, _del_likelihood)
    model=property(_get_model, _set_model, _del_model)


    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#
    
    #-------------perform_prediction-------------#
    def perform_prediction(self, candidates):
        assert self.pb.is_feasible(candidates)
        if candidates.ndim==1:
            candidates = np.array([candidates])

        self.__model.eval()
        self.__likelihood.eval()

        candidates = torch.tensor(candidates).float()
        predictions = self.__likelihood(self.__model(candidates))
        candidates = candidates.detach().numpy()
        mean_preds = predictions.mean
        mean_preds = mean_preds.detach().numpy()
        var_preds = predictions.stddev**2
        var_preds = var_preds.detach().numpy()

        return (mean_preds, var_preds)

    #-------------perform_training-------------#
    # from scratch (i.e. non incremental)
    def perform_training(self):
        Surrogate.perform_training(self)

        (x_train_np, y_train_np) = self.load_sim_archive()
        x_train_np = x_train_np[max(x_train_np.shape[0]-self.n_train_samples,0):x_train_np.shape[0]]
        y_train_np = y_train_np[max(y_train_np.shape[0]-self.n_train_samples,0):y_train_np.shape[0]]        
        x_train = torch.from_numpy(x_train_np).float()
        y_train = torch.from_numpy(y_train_np.flatten()).float()        
        self.__model = ExactGPModel(x_train, y_train, self.__likelihood)

        self.__model.train()
        self.__likelihood.train()

        optimizer = torch.optim.Adam([{'params': self.__model.parameters()},], lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.__likelihood, self.__model)
        # iterations_num = 10000
        iterations_num = 10 # for debugging

        # Early Stopping parameters
        es_tolerance = 1e-4
        es_patience = 56
        es_best_mse = None
        es_counter = 0

        t_start = time.time()
        for i in range(iterations_num):
            optimizer.zero_grad()
            output = self.__model(x_train)
            loss = -mll(output, y_train)
            loss.backward()
            # print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (i+1, iterations_num, loss.item(), model.covar_module.base_kernel.lengthscale.item(), model.likelihood.noise.item()))
            optimizer.step()
            mse = mean_squared_error(y_train.detach().numpy(), self.__likelihood(self.__model(x_train)).mean.detach().numpy() )
            # Early Stopping
            if es_best_mse is None: # first training iteration
                es_best_mse = mse
                torch.save(self.__model.state_dict(), self.f_trained_model)
            elif mse>es_best_mse: # no improvement
                es_counter+=1                
            else: # improvement
                torch.save(self.__model.state_dict(), self.f_trained_model)
                if (es_best_mse-mse) > es_tolerance: # improvement of at least es_tolerance
                    es_counter=0
                    es_best_mse = mse
                else:
                    es_counter+=1
            if es_counter>=es_patience:
                break
        t_end = time.time()
        self.__model.load_state_dict(torch.load(self.f_trained_model))
        mse = mean_squared_error(y_train.detach().numpy(), self.__likelihood(self.__model(x_train)).mean.detach().numpy() )

        # Log about training
        with open(self.f_train_log, 'a') as my_file:
            my_file.write(str(x_train_np.shape[0])+" "+str(mse)+" "+str((t_end-t_start))+"\n")
        
    #-------------load_trained_model-------------#
    def load_trained_model(self):
        self.__model.load_state_dict(torch.load(self.f_trained_model))
