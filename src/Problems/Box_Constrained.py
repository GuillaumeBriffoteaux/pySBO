import numpy as np
from abc import abstractmethod

from Problems.Problem import Problem


#--------------------------------------------------------#
#-------------abstract class Box_Constrained-------------#
#--------------------------------------------------------#
class Box_Constrained(Problem):
    """Abstract class for real-valued box-constrained optimization problems."""

    
    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#

    #-------------get_bounds-------------#
    @abstractmethod
    def get_bounds(self):
        pass
