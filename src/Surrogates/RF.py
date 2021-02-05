import time
import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.utils.validation import check_is_fitted

from Surrogates.Surrogate import Surrogate


#----------------------------------#
#-------------class RF-------------#
#----------------------------------#
class RF(Surrogate):
    """Class for Random Forest model (mono and multi dimensional targets).

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
    :param model: Random Forest model
    :type model: sklearn.ensemble.RandomForestRegressor
    """
    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#
 
    #-------------__init__-------------#
    def __init__(self, f_sim_archive, pb, n_train_samples, f_train_log, f_trained_model):
        Surrogate.__init__(self, f_sim_archive, pb, n_train_samples, f_train_log, f_trained_model)
        self.__model = RandomForestRegressor(n_estimators=1000, criterion='mse', max_depth=None, min_samples_split=10, min_samples_leaf=2, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False)

    #-------------__del__-------------#
    def __del__(self):
        Surrogate.__del__(self)
        del self.__model

    #-------------__str__-------------#
    def __str__(self):
        return "Random Forest\n  training set filename: "+self.f_sim_archive+"\n  associated problem: {"+self.pb.__str__()+"}\n  training set size = "+str(self.n_train_samples)+"\n  training log filename: "+self.f_train_log+"\n  trained model saved in "+self.f_trained_model

    
    #---------------------------------------------#
    #-------------getters and setters-------------#
    #---------------------------------------------#

    #-------------_get_model-------------#
    def _get_model(self):
        print("[RF.py] Impossible to modify the model")
        return None

    #-------------_set_model-------------#
    def _set_model(self,new_model):
        print("[RF.py] Impossible to modify the model")

    #-------------_del_model-------------#
    def _del_model(self):
        print("[RF.py] Impossible to delete the model")

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

        print(candidates.shape)
        print()
            
        check_is_fitted(self.__model, 'estimators_')
        candidates = self.__model._validate_X_predict(candidates)

        if self.__model.n_outputs_ > 1:
            mean_preds = np.zeros((candidates.shape[0], self.__model.n_outputs_), dtype=np.float64)
            var_preds = np.zeros((candidates.shape[0], self.__model.n_outputs_), dtype=np.float64)
        else:
            mean_preds = np.zeros((candidates.shape[0]), dtype=np.float64)
            var_preds = np.zeros((candidates.shape[0]), dtype=np.float64)

        for e in self.__model.estimators_:
            prediction = e.predict(candidates)
            mean_preds += prediction
            var_preds += pow(prediction, 2)

        mean_preds /= len(self.__model.estimators_)
        var_preds /= len(self.__model.estimators_)
        var_preds = var_preds - mean_preds**2
        
        return (mean_preds, var_preds)

    #-------------perform_training-------------#
    # from scratch
    def perform_training(self):
        Surrogate.perform_training(self)

        (x_train, y_train) = self.load_sim_archive()
        x_train = x_train[max(x_train.shape[0]-self.n_train_samples,0):x_train.shape[0]]
        y_train = y_train[max(y_train.shape[0]-self.n_train_samples,0):y_train.shape[0]]

        t_start = time.time()
        self.__model.fit(x_train, y_train)
        t_end = time.time()
        mse = mean_squared_error(y_train, self.__model.predict(x_train))

        # Saving the trained model
        with open(self.f_trained_model, 'wb') as my_file:
            pickle.dump(self.__model.__dict__, my_file)

        # Log about training
        with open(self.f_train_log, 'a') as my_file:
            my_file.write(str(x_train.shape[0])+" "+str(mse)+" "+str((t_end-t_start))+"\n")

    #-------------load_trained_model-------------#
    def load_trained_model(self):
        with open(self.f_trained_model, 'rb') as my_file:
            self.__model.__dict__.update(pickle.load(my_file))
