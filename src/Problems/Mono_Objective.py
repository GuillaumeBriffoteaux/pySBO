import numpy as np
from abc import abstractmethod

from Problems.Box_Constrained import Box_Constrained


#-------------------------------------------------------#
#-------------abstract class Mono_Objective-------------#
#-------------------------------------------------------#
class Mono_Objective(Box_Constrained):
    """Abstract class for artificial mono-objective real-valued box-constraints optimization problems."""

    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#

    #-------------__init__-------------#
    @abstractmethod
    def __init__(self, n_dvar, n_obj):
        Box_Constrained.__init__(self, n_dvar, n_obj)
        assert n_obj==1


    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#
    
    #-------------plot-------------#
    @abstractmethod
    def plot(self):
        pass
