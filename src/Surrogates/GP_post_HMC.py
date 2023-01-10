import csv
import pickle
import time
import pyro
import torch
import numpy as np
import pyro.contrib.gp as gp
from pyro.infer.mcmc import NUTS, MCMC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score

from Surrogates.Surrogate import Surrogate


#-------------------------------------------#
#-------------class GP_post_HMC-------------#
#-------------------------------------------#
class GP_post_HMC(Surrogate):

    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#
 
    #-------------__init__-------------#
    def __init__(self, f_sim_archive, pb, n_train_samples, f_train_log, f_trained_model, rank, q):
        Surrogate.__init__(self, f_sim_archive, pb, n_train_samples, f_train_log, f_trained_model)
        
        (x_train, y_train) = self.load_sim_archive()
        x_train = x_train[max(x_train.shape[0]-self.n_train_samples,0):x_train.shape[0]]
        y_train = y_train[max(y_train.shape[0]-self.n_train_samples,0):y_train.shape[0]]        
        in_dim = x_train.shape[1]
        x_train = torch.from_numpy(x_train).float()
        y_train = torch.from_numpy(y_train.flatten()).float()
        
        self.__outputs_scaler = None
        self.__rank = rank
        self.__q = q
        
        self.__kernel = gp.kernels.RBF(input_dim=in_dim, variance=torch.tensor(5.), lengthscale=torch.tensor(10.))
        self.__gpr_model = gp.models.GPRegression(x_train, y_train, self.__kernel, noise=torch.tensor(0.1))
        self.__gpr_model.kernel.lengthscale = pyro.nn.PyroSample(pyro.distributions.Uniform(0.01, 10.))
        self.__gpr_model.kernel.variance = pyro.nn.PyroSample(pyro.distributions.Uniform(0.0, 5.0))
        self.__gpr_model.noise = pyro.nn.PyroSample(pyro.distributions.Uniform(0.01, 0.1))
        
        self.__lengthscales = None
        self.__variances = None
        self.__noises = None


    #-------------__del__-------------#
    def __del__(self):
        del self.__outputs_scaler
        del self.__rank
        del self.__q
        del self.__kernel
        del self.__gpr_model
        del self.__lengthscales
        del self.__variances
        del self.__noises
        
    #-------------__str__-------------#
    def __str__(self):
        return "GP_post_HMC"

    
    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#
    
    #-------------perform_training-------------#
    # from scratch (i.e. non incremental)
    def perform_training(self):
        Surrogate.perform_training(self)

        # Load training data
        (x_train_np, y_train_np) = self.load_sim_archive()
        x_train_np = x_train_np[max(x_train_np.shape[0]-self.n_train_samples,0):x_train_np.shape[0]]
        in_dim = x_train_np.shape[1]
        y_train_np = y_train_np[max(y_train_np.shape[0]-self.n_train_samples,0):y_train_np.shape[0]]
        y_train_np = y_train_np.reshape(-1, 1)
        
        # Normalize training data in [0,1]
        x_train_np = (x_train_np - self.pb.get_bounds()[0]) / (self.pb.get_bounds()[1] - self.pb.get_bounds()[0])
        self.__outputs_scaler = MinMaxScaler(copy=False)
        self.__outputs_scaler.fit(y_train_np)
        self.__outputs_scaler.transform(y_train_np)
        
        # Format training data into tensors       
        x_train = torch.from_numpy(x_train_np).float()
        y_train = torch.from_numpy(y_train_np.flatten()).float()

        # Create the model
        self.__kernel = gp.kernels.RBF(input_dim=in_dim, variance=torch.tensor(5.), lengthscale=torch.tensor(10.))
        self.__gpr_model = gp.models.GPRegression(x_train, y_train, self.__kernel, noise=torch.tensor(0.1))
        self.__gpr_model.kernel.lengthscale = pyro.nn.PyroSample(pyro.distributions.Uniform(0.01, 10.))
        self.__gpr_model.kernel.variance = pyro.nn.PyroSample(pyro.distributions.Uniform(0.0, 5.0))
        self.__gpr_model.noise = pyro.nn.PyroSample(pyro.distributions.Uniform(0.01, 0.1))

        # Train
        t_start = time.time()
        hmc_kernel = NUTS(self.__gpr_model.model)
        mcmc = MCMC(hmc_kernel, num_samples=self.__q, warmup_steps=50, disable_progbar=True)
        mcmc.run()
        t_end = time.time()
        training_time = t_end-t_start

        # Save the sampled lengthscales, variances and noises
        samples = mcmc.get_samples()
        self.__lengthscales = samples['kernel.lengthscale'].detach().numpy()
        self.__variances = samples['kernel.variance'].detach().numpy()
        self.__noises = samples['noise'].detach().numpy()

        # Saving the trained model
        with open(self.f_trained_model, 'wb') as my_file:
            pickle.dump(self.__dict__, my_file)

        # Computing training MSE and training R
        # All computations are based on real-world units
        preds = np.empty(shape=(0,x_train.shape[0]))  # (0:q, 0:256)
        for i in range(self.__q):
            self.__gpr_model.kernel.lengthscale = torch.tensor(self.__lengthscales[i])
            self.__gpr_model.kernel.variance = torch.tensor(self.__variances[i])
            self.__gpr_model.noise = torch.tensor(self.__noises[i])
            # training data
            (new_preds, _) = self.__gpr_model(x_train)
            new_preds = new_preds.detach().numpy()
            new_preds = new_preds.reshape(1, x_train.shape[0])
            new_preds = self.denormalize_predictions(new_preds)
            preds = np.vstack((preds, new_preds))
        # training MSE and R2
        preds = np.average(preds, axis=0)
        preds = preds.reshape(-1,1)
        y_train_np = self.denormalize_predictions(y_train_np)
        training_mse = mean_squared_error(y_train_np, preds)
        training_r2 = r2_score(y_train_np, preds)

        # Reset this surrogate variant for future predictions
        self.__gpr_model.kernel.lengthscale = torch.tensor(self.__lengthscales[self.__rank])
        self.__gpr_model.kernel.variance = torch.tensor(self.__variances[self.__rank])
        self.__gpr_model.noise = torch.tensor(self.__noises[self.__rank])

        # Log about training
        with open(self.f_train_log, 'a') as my_file:
            my_file.write(str(y_train.shape[0])+" "+str(training_mse)+" "+str(training_r2)+" "+str(training_time)+"\n")

        
    #-------------perform_prediction-------------#
    def perform_prediction(self, candidates):
        assert self.pb.is_feasible(candidates)
        if candidates.ndim==1:
            candidates = np.array([candidates])

        # Normalizing the candidate in [0,1]
        candidates = (candidates - self.pb.get_bounds()[0]) / (self.pb.get_bounds()[1] - self.pb.get_bounds()[0])
        candidates = torch.tensor(candidates).float()
        
        (preds, stdevs) = self.__gpr_model(candidates)
        preds = preds.detach().numpy()
        stdevs = stdevs.detach().numpy()

        # result normalized
        return (preds, stdevs)


    #-------------load_trained_model-------------#
    def load_trained_model(self):
        saved_rank = self.__rank
        with open(self.f_trained_model, 'rb') as my_file:
            self.__dict__.update(pickle.load(my_file))
        self.__rank = saved_rank
            
        self.__gpr_model.kernel.lengthscale = torch.tensor(self.__lengthscales[self.__rank])
        self.__gpr_model.kernel.variance = torch.tensor(self.__variances[self.__rank])
        self.__gpr_model.noise = torch.tensor(self.__noises[self.__rank])

            
    #-------------denormalize_predictions-------------#
    def denormalize_predictions(self, preds):
        if self.pb.n_obj==1:
            preds = preds.reshape(-1, 1)
            
        denorm_preds = self.__outputs_scaler.inverse_transform(preds)
        
        if self.pb.n_obj==1:
            denorm_preds = denorm_preds.flatten()
            
        return denorm_preds

    
    #-------------normalize_obj_vals-------------#
    def normalize_obj_vals(self, costs):
        if self.pb.n_obj==1:
            costs = costs.reshape(-1, 1)

        norm_costs = self.__outputs_scaler.transform(costs)

        if self.pb.n_obj==1:
            norm_costs = norm_costs.flatten()
            
        return norm_costs
