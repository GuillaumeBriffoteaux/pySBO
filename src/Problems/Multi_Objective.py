import numpy as np
from abc import abstractmethod

from Problems.Box_Constrained import Box_Constrained


#--------------------------------------------------------#
#-------------abstract class Multi_Objective-------------#
#--------------------------------------------------------#
class Multi_Objective(Box_Constrained):
    """Abstract class for artificial multi-objective real-valued box-constrained optimization problems.

    :param n_dvar: number of decision variables
    :type n_dvar: positive int, not zero
    """


    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#

    #-------------__init__-------------#
    @abstractmethod
    def __init__(self, n_dvar, n_obj):
        Box_Constrained.__init__(self, n_dvar, n_obj)
        assert n_obj>1
