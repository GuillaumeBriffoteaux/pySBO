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

    #-------------_set_size-------------#
    def _set_size(self,new_size):
        print("[Tournament_Position.py] Impossible to modify the tournament size")

    #-------------_del_size-------------#
    def _del_size(self):
        print("[Tournament_Position.py] Impossible to delete the tournament size")

    #-------------property-------------#
    size=property(_get_size, _set_size, _del_size)


    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#    

    #-------------perform_selection-------------#    
    def perform_selection(self, pop, n_par):
        Selection.perform_selection(self, pop, n_par)
        assert pop.dvec.shape[0]>0

        parents = Population(pop.dvec.shape[1])

        # n_par tournaments
        for i in range(0, n_par):
            idx = np.random.choice(np.arange(0, pop.dvec.shape[0], 1, dtype=np.int), self.__size, False)
            parents.dvec = np.vstack( (parents.dvec, pop.dvec[np.amin(idx)]) )
            # parents.costs = np.append(parents.costs, pop.costs[np.amin(idx)])
            # parents.fitness_modes = np.append(parents.fitness_modes, pop.fitness_modes[np.amin(idx)])
                
        return parents
