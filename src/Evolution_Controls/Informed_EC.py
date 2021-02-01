import numpy as np
from abc import abstractmethod

from Evolution_Controls.Evolution_Control import Evolution_Control
from Surrogates.Surrogate import Surrogate


#----------------------------------------------------#
#-------------abstract class Informed_EC-------------#
#----------------------------------------------------#
class Informed_EC(Evolution_Control):
    """Abstract class for informed evolution controls.

    :param surr: surrogate model
    :type surr: Surrogate
    """

    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#
    
    @abstractmethod
    def __init__(self, surr):
        Evolution_Control.__init__(self)
        assert isinstance(surr, Surrogate)
        self.__surr=surr
        pass


    #---------------------------------------------#
    #-------------getters and setters-------------#
    #---------------------------------------------#

    #-------------_get_surr-------------#
    def _get_surr(self):
        return self.__surr

    #-------------_set_surr-------------#
    def _set_surr(self,new_surr):
        print("[Informed_EC.py] Impossible to modify the surrogate")

    #-------------_del_surr-------------#
    def _del_surr(self):
        print("[Informed_EC.py] Impossible to delete the surrogate")

    #-------------property-------------#
    surr=property(_get_surr, _set_surr, _del_surr)


    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#

    #-------------get_IC_value-------------#
    @abstractmethod
    def get_IC_value(self, dvec):
        """Returns the promise values of candidates.

        :param dvec: decision vectors
        :type dvec: np.ndarray
        :returns: the promise
        :rtype: np.ndarray
        """

        assert type(dvec)==np.ndarray
        pass
