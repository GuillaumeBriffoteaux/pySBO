import csv
import numpy as np
import scipy as sp
from abc import ABC
from abc import abstractmethod

from Problems.Problem import Problem


#--------------------------------------------------#
#-------------abstract class Surrogate-------------#
#--------------------------------------------------#
class Surrogate(ABC):
    """Abstract class for surrogate models.

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
    """
    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#

    #-------------__init__-------------#
    @abstractmethod
    def __init__(self, f_sim_archive, pb, n_train_samples, f_train_log, f_trained_model):
        assert type(f_sim_archive)==str
        assert isinstance(pb, Problem)
        assert (type(n_train_samples)==int and n_train_samples>0) or n_train_samples==float('inf')
        assert type(f_train_log)==str
        assert type(f_trained_model)==str
        self.__f_sim_archive=f_sim_archive
        self.__pb=pb
        self.__n_train_samples=n_train_samples
        self.__f_train_log=f_train_log
        self.__f_trained_model=f_trained_model
        pass
    
    #-------------__del__-------------#
    @abstractmethod
    def __del__(self):
        del self.__f_sim_archive
        del self.__pb
        del self.__n_train_samples
        del self.__f_train_log
        del self.__f_trained_model
        pass

    #-------------__str__-------------#
    @abstractmethod
    def __str__(self):
        pass


    #---------------------------------------------#
    #-------------getters and setters-------------#
    #---------------------------------------------#

    #-------------_get_f_sim_archive-------------#
    def _get_f_sim_archive(self):
        return self.__f_sim_archive

    #-------------_get_pb-------------#
    def _get_pb(self):
        return self.__pb

    #-------------_get_n_train_samples-------------#
    def _get_n_train_samples(self):
        return self.__n_train_samples

    #-------------_get_f_train_log-------------#
    def _get_f_train_log(self):
        return self.__f_train_log

    #-------------_get_f_trained_model-------------#
    def _get_f_trained_model(self):
        return self.__f_trained_model
        
    #-------------property-------------#
    f_sim_archive=property(_get_f_sim_archive, None, None)
    pb=property(_get_pb, None, None)
    n_train_samples=property(_get_n_train_samples, None, None)
    f_train_log=property(_get_f_train_log, None, None)
    f_trained_model=property(_get_f_trained_model, None, None)

    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#

    #-------------perform_training-------------#
    @abstractmethod
    def perform_training(self):
        """Trains the surrogate model."""
        
        pass

    #-------------perform_prediction-------------#
    @abstractmethod
    def perform_prediction(self, candidates):
        """Returns the predicted objective value of the candidates and the uncertainty around it. Both quantities are normalized.

        :param candidates: candidates
        :type candidates: np.ndarray
        :returns: predicted objective values and uncertainties
        :rtype: (np.ndarray, np.ndarray)
        """        
        pass
    
    #-------------perform_distance-------------#
    def perform_distance(self, candidates):
        """Returns the distance from each normalized candidate to the set of normalized known candidates.

        :param candidates: candidates
        :type candidates: np.ndarray
        :returns: distances from the normalized candidates to the set of normalized known candidates
        :rtype: np.ndarray
        """
        
        assert self.pb.is_feasible(candidates)
        if candidates.ndim==1:
            candidates = np.array([candidates])
        
        (x_train, y_train) = self.load_sim_archive()
        x_train = (x_train - self.pb.get_bounds()[0]) / (self.pb.get_bounds()[1] - self.pb.get_bounds()[0]) # normalization [0,1]
        copy_candidates = (candidates - self.pb.get_bounds()[0]) / (self.pb.get_bounds()[1] - self.pb.get_bounds()[0]) # normalization [0,1]
        dists = np.amin(sp.spatial.distance.cdist(copy_candidates, x_train), axis=1)

        return dists
    
    #-------------load_sim_archive-------------#
    def load_sim_archive(self):
        """Returns the past simulated candidates.

        :returns: past simulated decision vectors along with their objective value
        :rtype: tuple(np.ndarray,np.ndarray)
        """

        with open(self.__f_sim_archive, 'r') as my_file:
            # Counting the number of lines.
            reader = csv.reader(my_file, delimiter=' ')
            n_samples = sum(1 for line in reader)
            my_file.seek(0)

            # Following lines contain (candidate, obj_val)
            candidates = np.zeros((n_samples, self.__pb.n_dvar))
            obj_vals = np.zeros((n_samples, self.__pb.n_obj))
            for i, line in enumerate(reader):
                candidates[i] = np.asarray(line[0:self.__pb.n_dvar])
                obj_vals[i,0:self.__pb.n_obj] = np.asarray(line[self.__pb.n_dvar:self.__pb.n_dvar+self.__pb.n_obj])
            if obj_vals.shape[1]<2:
                obj_vals = np.ndarray.flatten(obj_vals)

        return (candidates, obj_vals)


    #-------------denormalize_predictions-------------#
    @abstractmethod
    def denormalize_predictions(self, preds):
        """Denormalize predicted objective value."""
        pass
    
    #-------------normalize_obj_vals-------------#
    @abstractmethod
    def normalize_obj_vals(self, obj_vals):
        """Normalized objective values."""
        pass

    #-------------load_trained_model-------------#
    @abstractmethod
    def load_trained_model(self):
        """Loads a model trained in the past."""
        pass
