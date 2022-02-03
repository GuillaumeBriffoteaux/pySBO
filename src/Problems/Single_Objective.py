import numpy as np
from abc import abstractmethod

from Problems.Box_Constrained import Box_Constrained


#---------------------------------------------------------#
#-------------abstract class Single_Objective-------------#
#---------------------------------------------------------#
class Single_Objective(Box_Constrained):
    """Abstract class for artificial single-objective real-valued box-constrained optimization problems.

    :param n_dvar: number of decision variables
    :type n_dvar: positive int, not zero
    """

    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#

    #-------------__init__-------------#
    @abstractmethod
    def __init__(self, n_dvar):
        Box_Constrained.__init__(self, n_dvar, 1)


    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#
    
    #-------------plot-------------#
    @abstractmethod
    def plot(self):
        pass
