import numpy as np
from abc import ABC
from abc import abstractmethod

from Evolution.Population import Population


#--------------------------------------------------#
#-------------abstract class Selection-------------#
#--------------------------------------------------#
class Selection(ABC):
    """Abstract class for selection operators."""
    
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
    
    #-------------perform_selection-------------#
    @abstractmethod
    def perform_selection(self, pop, n_par):
        """Selects individuals from a population.

        :param pop: population to select from
        :type pop: Population
        :param n_par: number of individuals to select
        :type n_par: positive int, not zero
        :returns: the selected individuals
        :rtype: Population
        """

        assert isinstance(pop, Population)
        assert type(n_par)==int and n_par>0
        pass
