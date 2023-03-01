import numpy as np

from Evolution.Population import Population
from Evolution.Selection import Selection


#---------------------------------------------------#
#-------------class Tournament_Position-------------#
#---------------------------------------------------#
class Tournament_Position(Selection):
    """Class for Tournament_Position selection.

    Candidates are ordered within the population according to their promise (decreasing order).

    :param size: tournament size
    :type size: positive int, not zero
    """
    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#

    #-------------__init__-------------#
    def __init__(self, size):
        assert type(size)==int
        assert size>0
        self.__size=size

    #-------------__del__-------------#
    def __del__(self):
        del self.__size

    #-------------__str__-------------#
    def __str__(self):
        return "Tournament_Position size "+str(self.__size)


    #---------------------------------------------#
    #-------------getters and setters-------------#
    #---------------------------------------------#

    #-------------_get_size-------------#
    def _get_size(self):
        return self.__size

    #-------------property-------------#
    size=property(_get_size, None, None)


    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#    

    #-------------perform_selection-------------#    
    def perform_selection(self, pop, n_par):
        Selection.perform_selection(self, pop, n_par)
        assert pop.dvec.shape[0]>0

        replace_mode = False        
        if pop.dvec.shape[0]<self.__size:
            # can occur in RVEA
            replace_mode = True

        parents = Population(pop.pb)

        # n_par tournaments
        for i in range(0, n_par):
            idx = np.random.choice(np.arange(0, pop.dvec.shape[0], 1, dtype=int), self.__size, replace_mode)
            parents.dvec = np.vstack( (parents.dvec, pop.dvec[np.amin(idx)]) )
                
        return parents
