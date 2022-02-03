import time
import pickle
import sklearn
import scipy as sp
import numpy as np
from pybnn import DNGO
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from pybnn.util.normalization import zero_mean_unit_var_normalization

from Surrogates.Surrogate import Surrogate


#---------------------------------------#
#-------------class BLR_ANN-------------#
#---------------------------------------#
class BLR_ANN(Surrogate):
    """Class for Bayesian Linear Regressor with Artificial Neural Network basis functions.

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
    :param model: DNGO model for the network
    :type model: pybnn.DNGO
    """

    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#
 
    #-------------__init__-------------#
    def __init__(self, f_sim_archive, pb, n_train_samples, f_train_log, f_trained_model):
        Surrogate.__init__(self, f_sim_archive, pb, n_train_samples, f_train_log, f_trained_model)
        self.__model = DNGO(num_epochs=500, do_mcmc=False, normalize_input=True, normalize_output=True)

    #-------------__del__-------------#
    def __del__(self):
        Surrogate.__del__(self)
        del self.__model

    #-------------__str__-------------#
    def __str__(self):
        return "Bayesian Linear Regression\n  training set filename: "+self.f_sim_archive+"\n  associated problem: {"+self.pb.__str__()+"}\n  training set size = "+str(self.n_train_samples)+"\n  training log filename: "+self.f_train_log+"\n  trained model saved in "+self.f_trained_model

    
    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#

    
    #-------------perform_training-------------#
    def perform_training(self):
        Surrogate.perform_training(self)

        # Reading training data
        (x_train, y_train) = self.load_sim_archive()
        x_train = x_train[max(x_train.shape[0]-self.n_train_samples,0):x_train.shape[0]]
        y_train = y_train[max(y_train.shape[0]-self.n_train_samples,0):y_train.shape[0]]

        # Training
        t_start = time.time()
        self.__model.train(x_train, y_train, do_optimize=False) # No Early Stopping is implemented in DNGO (from pybnn)
        t_end = time.time()
        training_time = (t_end-t_start)

        # Compute training MSE and R-square in real-world units
        preds, _ = self.__model.predict(x_train)
        training_mse = mean_squared_error(y_train, preds)
        training_r2 = r2_score(y_train, preds)

        # # Saving the trained model
        # with open(self.f_trained_model, 'wb') as my_file:
        #     pickle.dump(self.__model.__dict__, my_file)
        
        # Log about training
        with open(self.f_train_log, 'a') as my_file:
            my_file.write(str(y_train.shape[0])+" "+str(training_mse)+" "+str(training_r2)+" "+str(training_time)+"\n")

    
    #-------------perform_prediction-------------#
    def perform_prediction(self, candidates):
        assert self.pb.is_feasible(candidates)
        if candidates.ndim==1:
            candidates = np.array([candidates])

        self.__model.normalize_output = False
        mean_preds, var_preds = self.__model.predict(candidates)
        self.__model.normalize_output = True
        std_preds = np.sqrt(var_preds)

        # normalized results N(0,1)
        return (mean_preds, std_preds)


    #-------------perform_distance-------------#
    def perform_distance(self, candidates):
        """Returns the distance from each normalized candidate to the set of normalized known individuals.

        :param candidates: candidates
        :type candidates: np.ndarray
        :returns: distances from the normalized candidates to the set of normalized known individuals
        :rtype: np.ndarray
        """
        
        assert self.pb.is_feasible(candidates)
        if candidates.ndim==1:
            candidates = np.array([candidates])
        
        (x_train, y_train) = self.load_sim_archive()
        x_train, mean, std = zero_mean_unit_var_normalization(x_train)
        candidates = (candidates - mean) / std
        dists = np.amin(sp.spatial.distance.cdist(candidates, x_train), axis=1)

        return dists


    #-------------denormalize_predictions-------------#
    def denormalize_predictions(self, preds):
        if self.pb.n_obj==1:
            preds = preds.flatten()

        return (preds * self.__model.y_std + self.__model.y_mean)

    
    #-------------normalize_obj_vals-------------#
    def normalize_obj_vals(self, obj_val):
        if self.pb.n_obj==1:
            obj_val = obj_val.flatten()
            
        return (obj_val-self.__model.y_mean)/self.__model.y_std


    #-------------load_trained_model-------------#
    def load_trained_model(self):
        with open(self.f_trained_model, 'rb') as my_file:
            self.__model.__dict__.update(pickle.load(my_file))
