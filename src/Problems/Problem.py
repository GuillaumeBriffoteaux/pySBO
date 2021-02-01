import csv
import numpy as np
from abc import ABC
from abc import abstractmethod


#------------------------------------------------#
#-------------abstract class Problem-------------#
#------------------------------------------------#
class Problem(ABC):
    """Abstract class for real-valued optimization problems.

    :param n_dvar: number of decision variables
    :type n_dvar: positive int, not zero
    :param n_obj: number of objectives
    :type n_obj: positive int, not zero
    """
    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#

    #-------------__init__-------------#
    @abstractmethod
    def __init__(self, n_dvar, n_obj):
        assert type(n_dvar)==int and type(n_obj)==int
        assert n_dvar>0 and n_obj>0
        self.__n_dvar=n_dvar
        self.__n_obj=n_obj

    #-------------__del__-------------#
    @abstractmethod
    def __del__(self):
        del self.__n_dvar
        del self.__n_obj

    #-------------__str__-------------#
    @abstractmethod
    def __str__(self):
        pass


    #---------------------------------------------#
    #-------------getters and setters-------------#
    #---------------------------------------------#

    #-------------_get_n_dvar-------------#
    def _get_n_dvar(self):
        return self.__n_dvar

    #-------------_set_n_dvar-------------#
    def _set_n_dvar(self,new_n_dvar):
        print("[Problem.py] Impossible to modify the number of decision variables")

    #-------------_del_n_dvar-------------#
    def _del_n_dvar(self):
        print("[Problem.py] Impossible to delete the number of decision variables")

    #-------------_get_n_obj-------------#
    def _get_n_obj(self):
        return self.__n_obj

    #-------------_set_n_obj-------------#
    def _set_n_obj(self,new_n_obj):
        print("[Problem.py] Impossible to modify the number of objectives")

    #-------------_del_n_obj-------------#
    def _del_n_obj(self):
        print("[Problem.py] Impossible to delete the number of objectives")

    #-------------property-------------#
    n_dvar=property(_get_n_dvar, _set_n_dvar, _del_n_dvar)
    n_obj=property(_get_n_obj, _set_n_obj, _del_n_obj)

    
    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#
    
    #-------------perform_real_evaluation-------------#
    @abstractmethod
    def perform_real_evaluation(self, candidates):
        assert self.is_feasible(candidates)==True

    #-------------is_feasible-------------#
    @abstractmethod
    def is_feasible(self, candidates):
        res=False
        if type(candidates) is np.ndarray:
            if candidates.ndim==1:
                res=(candidates.size==self.__n_dvar)
            else:
                res=(candidates.shape[1]==self.__n_dvar)
        return res

    #-------------get_bounds-------------#
    @abstractmethod
    def get_bounds(self):
        pass
