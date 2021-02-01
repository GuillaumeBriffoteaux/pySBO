import time
import pickle
import numpy as np
from pyKriging.krige import kriging
from sklearn.metrics import mean_squared_error

from Surrogates.Surrogate import Surrogate


#-----------------------------------#
#-------------class KRG-------------#
#-----------------------------------#
class KRG(Surrogate):
    """Class for Kriging surrogate model.

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
    :param model: kriging model
    :type model: pyKriging.krige.kriging
    """

    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#
 
    #-------------__init__-------------#
    def __init__(self, f_sim_archive, pb, n_train_samples, f_train_log, f_trained_model):
        Surrogate.__init__(self, f_sim_archive, pb, n_train_samples, f_train_log, f_trained_model)
        
        (x_train, y_train) = self.load_sim_archive()
        self.__model = kriging(x_train, y_train)

    #-------------__del__-------------#
    def __del__(self):
        Surrogate.__del__(self)
        del self.__model

    #-------------__str__-------------#
    def __str__(self):
        return "Kriging\n  training set filename: "+self.f_sim_archive+"\n  associated problem: {"+self.pb.__str__()+"}\n  training set size = "+str(self.n_train_samples)+"\n  training log filename: "+self.f_train_log+"\n  trained model saved in "+self.f_trained_model


    #---------------------------------------------#
    #-------------getters and setters-------------#
    #---------------------------------------------#

    #-------------_get_model-------------#
    def _get_model(self):
        print("[KRG.py] Impossible to modify the model")
        return None

    #-------------_set_model-------------#
    def _set_model(self,new_model):
        print("[KRG.py] Impossible to modify the model")

    #-------------_del_model-------------#
    def _del_model(self):
        print("[KRG.py] Impossible to delete the model")

    #-------------property-------------#
    model=property(_get_model, _set_model, _del_model)


    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#
    
    #-------------perform_prediction-------------#
    def perform_prediction(self, candidates):
        assert self.pb.is_feasible(candidates)
        if candidates.ndim==1:
            candidates = np.array([candidates])

        mean_preds = np.zeros((candidates.shape[0],))
        var_preds = np.zeros((candidates.shape[0],))
        for i,x in enumerate(candidates):
            mean_preds[i] = self.__model.predict(x)
            var_preds[i] = self.__model.predict_var(x)

        return (mean_preds, var_preds)

    #-------------perform_training-------------#
    # from scratch
    def perform_training(self):
        Surrogate.perform_training(self)

        (x_train, y_train) = self.load_sim_archive()
        x_train = x_train[max(x_train.shape[0]-self.n_train_samples,0):x_train.shape[0]]
        y_train = y_train[max(y_train.shape[0]-self.n_train_samples,0):y_train.shape[0]]
        self.__model = kriging(x_train, y_train)

        t_start = time.time()        
        self.__model.train('ga')
        t_end = time.time()

        # Saving the trained model
        with open(self.f_trained_model, 'wb') as my_file:
            pickle.dump(self.__model.__dict__, my_file)

        # Log about training
        with open(self.f_train_log, 'a') as my_file:
            my_file.write(str(x_train.shape[0])+" - "+str((t_end-t_start))+"\n")
        
    #-------------load_trained_model-------------#
    def load_trained_model(self):
        with open(self.f_trained_model, 'rb') as my_file:
            self.__model.__dict__.update(pickle.load(my_file))
