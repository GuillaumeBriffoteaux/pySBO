import warnings
import numpy as np
import time
import pickle
import gpytorch
import torch
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler

from Surrogates.Surrogate import Surrogate
from Surrogates.MultitaskGPModel import MultitaskGPModel

#-------------------------------------#
#-------------class GP_MO-------------#
#-------------------------------------#
class GP_MO(Surrogate):
    """Class for Gaussian Process multitask model (multi dimensional targets only).

    :param f_sim_archive: filename where are stored the simulated candidates
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
    :param outputs_scaler: objective values normalizer
    :type outputs_scaler: sklearn.preprocessing
    :param kernel: covariance function
    :type kernel: str
    :param model: Gaussian Process model
    :type model: gpytorch.models.ExactGP
    """
    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#
 
    #-------------__init__-------------#
    def __init__(self, f_sim_archive, pb, n_train_samples, f_train_log, f_trained_model, kernel):
        Surrogate.__init__(self, f_sim_archive, pb, n_train_samples, f_train_log, f_trained_model)
        
        self.__likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=self.pb.n_obj)
        
        (x_train, y_train) = self.load_sim_archive()
        x_train = torch.from_numpy(x_train).float()
        y_train = torch.from_numpy(y_train).float()
        
        self.__outputs_scaler = None
        self.__kernel = kernel
        self.__model = MultitaskGPModel(x_train, y_train, self.__likelihood, self.__kernel)

        
    #-------------__del__-------------#
    def __del__(self):
        Surrogate.__del__(self)
        del self.__likelihood
        del self.__model

        
    #-------------__str__-------------#
    def __str__(self):
        return "Gaussian Process Multitask "+self.__kernel+" Kernel\n  training set filename: "+self.f_sim_archive+"\n  associated problem: {"+self.pb.__str__()+"}\n  training set size = "+str(self.n_train_samples)+"\n  training log filename: "+self.f_train_log+"\n  trained model saved in "+self.f_trained_model


    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#


    #-------------perform_training-------------#
    # from scratch (i.e. non incremental)
    def perform_training(self):
        Surrogate.perform_training(self)

        # Read training data
        (x_train_np, y_train_np) = self.load_sim_archive()
        x_train_np = x_train_np[max(x_train_np.shape[0]-self.n_train_samples,0):x_train_np.shape[0]]
        y_train_np = y_train_np[max(y_train_np.shape[0]-self.n_train_samples,0):y_train_np.shape[0]]
        # Normalize training data in [0,1]
        x_train_np = (x_train_np - self.pb.get_bounds()[0]) / (self.pb.get_bounds()[1] - self.pb.get_bounds()[0])
        self.__outputs_scaler = MinMaxScaler(copy=False)
        self.__outputs_scaler.fit(y_train_np)
        self.__outputs_scaler.transform(y_train_np)
        
        x_train = torch.from_numpy(x_train_np).float()
        y_train = torch.from_numpy(y_train_np).float()
        self.__model = MultitaskGPModel(x_train, y_train, self.__likelihood, self.__kernel)

        self.__model.train()
        self.__likelihood.train()

        optimizer = torch.optim.Adam([{'params': self.__model.parameters()},], lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.__likelihood, self.__model)
        iterations_num = 10000

        # Early Stopping parameters
        es_tolerance = 1e-8
        es_patience = 32
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
        training_time = t_end-t_start
        
        # Compute training MSE and R-square in real-world units
        self.__model.load_state_dict(torch.load(self.f_trained_model))
        self.__model.eval()
        self.__likelihood.eval()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preds = self.__likelihood(self.__model(x_train)).mean       
        preds = preds.detach().numpy()
        preds = self.denormalize_predictions(preds)
        y_train_np = self.denormalize_predictions(y_train_np)
        training_mse = mean_squared_error(y_train_np, preds)
        training_r2 = r2_score(y_train_np, preds)

        # Log about training
        with open(self.f_train_log, 'a') as my_file:
            my_file.write(str(y_train.shape[0])+" "+str(training_mse)+" "+str(training_r2)+" "+str(training_time)+"\n")


    #-------------perform_partial_training-------------#
    def perform_partial_training(self):

        (x_train_np, y_train_np) = self.load_sim_archive()
        x_train_np = x_train_np[max(x_train_np.shape[0]-self.n_train_samples,0):x_train_np.shape[0]]
        y_train_np = y_train_np[max(y_train_np.shape[0]-self.n_train_samples,0):y_train_np.shape[0]]
        
        # Normalize training data in [0,1]
        x_train_np = (x_train_np - self.pb.get_bounds()[0]) / (self.pb.get_bounds()[1] - self.pb.get_bounds()[0])
        self.__outputs_scaler = MinMaxScaler(copy=False)
        self.__outputs_scaler.fit(y_train_np)
        self.__outputs_scaler.transform(y_train_np)
        
        # Format training data into tensors
        x_train = torch.from_numpy(x_train_np).float()
        y_train = torch.from_numpy(y_train_np).float()
        self.__model = MultitaskGPModel(x_train, y_train, self.__likelihood, self.__kernel)

        self.__model.train()
        self.__likelihood.train()
        self.__model.load_state_dict(torch.load(self.f_trained_model))

        
    #-------------perform_prediction-------------#
    def perform_prediction(self, candidates):
        assert self.pb.is_feasible(candidates)
        if candidates.ndim==1:
            candidates = np.array([candidates])

        self.__model.eval()
        self.__likelihood.eval()

        # Normalizing candidates in [0,1]
        candidates = (candidates - self.pb.get_bounds()[0]) / (self.pb.get_bounds()[1] - self.pb.get_bounds()[0])

        candidates = torch.tensor(candidates).float()
        predictions = self.__likelihood(self.__model(candidates))
        candidates = candidates.detach().numpy()
        mean_preds = predictions.mean
        mean_preds = mean_preds.detach().numpy()
        var_preds = predictions.stddev
        var_preds = var_preds.detach().numpy()

        return (mean_preds, var_preds)

    
    #-------------denormalize_predictions-------------#
    def denormalize_predictions(self, preds):
        if self.pb.n_obj==1:
            preds = preds.reshape(-1, 1)
            
        denorm_preds = self.__outputs_scaler.inverse_transform(preds)
        
        if self.pb.n_obj==1:
            denorm_preds = denorm_preds.flatten()
            
        return denorm_preds

    
    #-------------get_stdevs_denorm-------------#
    def get_stdevs_denorm(self, candidates):
        assert self.pb.is_feasible(candidates)
        if candidates.ndim==1:
            candidates = np.array([candidates])

        self.__model.eval()
        self.__likelihood.eval()

        # # Normalizing candidates in [0,1]
        # candidates = (candidates - self.pb.get_bounds()[0]) / (self.pb.get_bounds()[1] - self.pb.get_bounds()[0])

        candidates = torch.tensor(candidates).float()
        predictions = self.__likelihood(self.__model(candidates))
        candidates = candidates.detach().numpy()
        mean_preds = predictions.mean
        mean_preds = mean_preds.detach().numpy()
        stdev_preds = predictions.stddev
        stdev_preds = stdev_preds.detach().numpy()

        # denormalized results
        return stdev_preds


    #-------------normalize_obj_vals-------------#
    def normalize_obj_vals(self, obj_vals):
        if self.pb.n_obj==1:
            obj_vals = obj_vals.reshape(-1, 1)

        norm_obj_vals = self.__outputs_scaler.transform(obj_vals)

        if self.pb.n_obj==1:
            norm_obj_vals = norm_obj_vals.flatten()
        
        return norm_obj_vals


    #-------------load_trained_model-------------#
    def load_trained_model(self):
        self.__model.load_state_dict(torch.load(self.f_trained_model))
