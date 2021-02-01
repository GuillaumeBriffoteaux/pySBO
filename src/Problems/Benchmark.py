import numpy as np
from abc import abstractmethod

from Problems.Box_Constrained import Box_Constrained


#--------------------------------------------------#
#-------------abstract class Benchmark-------------#
#--------------------------------------------------#
class Benchmark(Box_Constrained):
    """Abstract class for artificial real-valued box-constraints optimization problems."""

    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#
    
    #-------------plot-------------#
    @abstractmethod
    def plot(self):
        pass
