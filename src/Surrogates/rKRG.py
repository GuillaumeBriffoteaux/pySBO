import time
import pickle
import numpy as np
from pyKriging.regressionkrige import regression_kriging
import sklearn as sk
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler

from Surrogates.Surrogate import Surrogate


#------------------------------------#
#-------------class rKRG-------------#
#------------------------------------#
class rKRG(Surrogate):
    """Class for regression Kriging surrogate model.

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
    :param y_bounds: lower and upper bounds of the objective values found in the training set
    :type y_bounds: np.ndarray
    :param outputs_scaler: objective values normalizer
    :type outputs_scaler: sklearn.preprocessing
    :param model: regression_kriging model
    :type model: pyKriging.krige.regression_kriging

    If you encounter some "np.float deprecated" error while using rKRG.py, please refer to the following page <https://github.com/capaulson/pyKriging/pull/53/files>
    """

    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#
 
    #-------------__init__-------------#
    def __init__(self, f_sim_archive, pb, n_train_samples, f_train_log, f_trained_model):
        Surrogate.__init__(self, f_sim_archive, pb, n_train_samples, f_train_log, f_trained_model)

        self.__y_bounds=np.empty((2,))
        self.__outputs_scaler = None
        
        (x_train, y_train) = self.load_sim_archive()
        self.__model = regression_kriging(x_train, y_train)

    #-------------__del__-------------#
    def __del__(self):
        Surrogate.__del__(self)
        del self.__model

    #-------------__str__-------------#
    def __str__(self):
        return "Regression Kriging\n  training set filename: "+self.f_sim_archive+"\n  associated problem: {"+self.pb.__str__()+"}\n  training set size = "+str(self.n_train_samples)+"\n  training log filename: "+self.f_train_log+"\n  trained model saved in "+self.f_trained_model


    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#

    
    #-------------perform_training-------------#
    # from scratch
    def perform_training(self):
        Surrogate.perform_training(self)

        # Reading training data
        (x_train, y_train) = self.load_sim_archive()
        x_train = x_train[max(x_train.shape[0]-self.n_train_samples,0):x_train.shape[0]]
        y_train = y_train[max(y_train.shape[0]-self.n_train_samples,0):y_train.shape[0]]
        y_train_shape = y_train.shape
        y_train = y_train.reshape(-1, 1)

        # Normalize training data in [0,1]
        x_train = (x_train - self.pb.get_bounds()[0]) / (self.pb.get_bounds()[1] - self.pb.get_bounds()[0])
        self.__outputs_scaler = MinMaxScaler(copy=False)
        self.__outputs_scaler.fit(y_train)
        self.__outputs_scaler.transform(y_train)
        y_train = y_train.reshape(y_train_shape)
        
        self.__model = regression_kriging(x_train, y_train)

        # Training
        t_start = time.time()
        self.__model.train('ga')
        t_end = time.time()
        training_time = t_end-t_start

        # Compute training MSE and R-square in real-world units
        preds = np.zeros((x_train.shape[0],))
        for i,x in enumerate(x_train):
            preds[i] = self.__model.predict_normalized(x)
        preds = self.denormalize_predictions(preds)
        y_train = y_train.reshape(-1, 1)
        y_train = self.denormalize_predictions(y_train)
        training_mse = mean_squared_error(y_train, preds, squared=True)
        training_r2 = sk.metrics.r2_score(y_train, preds)

        # Log about training
        with open(self.f_train_log, 'a') as my_file:
            my_file.write(str(y_train.shape[0])+" "+str(training_mse)+" "+str(training_r2)+" "+str(training_time)+"\n")


    #-------------add_point-------------#
    def add_point(self, candidate, obj_val):
        """Partial update (used in q-EGO)

        :param candidate: new candidate to add to the training set
        :type candidate: np.ndarray
        :param obj_val: objective value associated to the candidate
        :type obj_val: np.ndarray
        """
        
        # normalize candidate
        copy_candidate = (candidate - self.pb.get_bounds()[0]) / (self.pb.get_bounds()[1] - self.pb.get_bounds()[0])
        # normalize obj_val
        obj_val = self.normalize_obj_vals(obj_val)
        # add point to the model
        self.__model.addPoint(candidate, obj_val, False)


    #-------------perform_prediction-------------#
    def perform_prediction(self, candidates):
        # candidate in real-world units
        assert self.pb.is_feasible(candidates)
        if candidates.ndim==1:
            candidates = np.array([candidates])

        # Normalizing candidates in [0,1]
        copy_candidates = (candidates - self.pb.get_bounds()[0]) / (self.pb.get_bounds()[1] - self.pb.get_bounds()[0])

        mean_preds = np.zeros((copy_candidates.shape[0],))
        stdev_preds = np.zeros((copy_candidates.shape[0],))
        for i,x in enumerate(copy_candidates):
            mean_preds[i] = self.__model.predict_normalized(x)
            stdev_preds[i] = self.__model.predict_var(x)
        stdev_preds = np.sqrt(stdev_preds)
            
        # results normalized
        return (mean_preds, stdev_preds)

    
    #-------------denormalize_predictions-------------#
    def denormalize_predictions(self, preds):
        if self.pb.n_obj==1:
            preds = preds.reshape(-1, 1)
            
        denorm_preds = self.__outputs_scaler.inverse_transform(preds)
        
        if self.pb.n_obj==1:
            denorm_preds = denorm_preds.flatten()
            
        return denorm_preds

    
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
        with open(self.f_trained_model, 'rb') as my_file:
            self.__model.__dict__.update(pickle.load(my_file))
