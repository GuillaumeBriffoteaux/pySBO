import numpy as np
from abc import ABC
from abc import abstractmethod

from Evolution.Population import Population


#----------------------------------------------------#
#-------------abstract class Replacement-------------#
#----------------------------------------------------#
class Replacement(ABC):
    """Abstract class for replacement operators."""
    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#

    #-------------__init__-------------#
    @abstractmethod
    def __init__(self):
        pass
    
    #-------------__del__-------------#
    @abstractmethod
    def __del__(self):
        pass

    #-------------__str__-------------#
    @abstractmethod
    def __str__(self):
        pass

    
    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#
    
    #-------------perform_replacement-------------#
    @abstractmethod
    def perform_replacement(self, pop, children):
        assert isinstance(pop, Population)
        assert isinstance(children, Population)
        pass
