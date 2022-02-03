import numpy as np
from abc import ABC
from abc import abstractmethod

from Evolution.Population import Population


#--------------------------------------------------#
#-------------abstract class Crossover-------------#
#--------------------------------------------------#
class Crossover(ABC):
    """Abstract class for crossover operators.

    :param prob: probability of crossover
    :type prob: float in [0,1]
    """
    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#

    #-------------__init__-------------#
    @abstractmethod
    def __init__(self, prob):
        assert type(prob)==float
        assert (prob>=0.0 and prob<=1.0)
        self.__prob=prob

    #-------------__del__-------------#
    @abstractmethod
    def __del__(self):
        del self.__prob

    #-------------__str__-------------#
    @abstractmethod
    def __str__(self):
        pass


    #---------------------------------------------#
    #-------------getters and setters-------------#
    #---------------------------------------------#
    
    #-------------_get_prob-------------#
    def _get_prob(self):
        return self.__prob

    #-------------property-------------#
    prob=property(_get_prob, None, None)

    
    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#
    
    #-------------perform_crossover-------------#
    @abstractmethod
    def perform_crossover(self, pop):
        assert isinstance(pop, Population)
        pass
