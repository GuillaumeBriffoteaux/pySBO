from abc import ABC
from abc import abstractmethod

from Evolution.Population import Population


#----------------------------------------------------------#
#-------------abstract class Evolution_Control-------------#
#----------------------------------------------------------#
class Evolution_Control(ABC):
    """Abstract class for evolution controls."""
    
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
    
    #-------------get_sorted_indexes-------------#
    @abstractmethod
    def get_sorted_indexes(self, pop):
        """Returns the candidates' indexes sorted in descending promise order.

        :param pop: population to split
        :type pop: Population
        :returns: list of indexes
        :rtype: np.ndarray
        """

        assert isinstance(pop, Population)
        assert pop.obj_vals.size==0 and pop.fitness_modes.size==0
        pass
