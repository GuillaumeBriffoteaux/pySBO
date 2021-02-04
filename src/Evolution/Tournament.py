import numpy as np

from Evolution.Population import Population
from Evolution.Selection import Selection


#------------------------------------------#
#-------------class Tournament-------------#
#------------------------------------------#
class Tournament(Selection):
    """Class for Tournament selection.

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
        return "Tournament size "+str(self.__size)


    #---------------------------------------------#
    #-------------getters and setters-------------#
    #---------------------------------------------#

    #-------------_get_size-------------#
    def _get_size(self):
        return self.__size

    #-------------_set_size-------------#
    def _set_size(self,new_size):
        print("[Tournament.py] Impossible to modify the tournament size")

    #-------------_del_size-------------#
    def _del_size(self):
        print("[Tournament.py] Impossible to delete the tournament size")

    #-------------property-------------#
    size=property(_get_size, _set_size, _del_size)


    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#    

    #-------------perform_selection-------------#    
    def perform_selection(self, pop, n_par):
        Selection.perform_selection(self, pop, n_par)
        assert pop.dvec.shape[0]>0

        parents = Population(pop.pb)

        # first tournament
        idx = np.random.choice(np.arange(0, pop.dvec.shape[0], 1, dtype=np.int), self.__size, False)
        group_dvec = pop.dvec[idx,:]
        group_costs = pop.costs[idx]
        group_fitness_modes = pop.fitness_modes[idx]

        parents.dvec = group_dvec[np.argmin(group_costs)]
        parents.costs = np.min(group_costs)
        parents.fitness_modes = group_fitness_modes[np.argmin(group_costs)]

        # next tournaments
        for i in range(1, n_par):
            idx = np.random.choice(np.arange(0, pop.dvec.shape[0], 1, dtype=np.int), self.__size, False)
            group_dvec = pop.dvec[idx,:]
            group_costs = pop.costs[idx]
            group_fitness_modes = pop.fitness_modes[idx]
            
            parents.dvec = np.vstack( (parents.dvec, group_dvec[np.argmin(group_costs)]) )
            parents.costs = np.append(parents.costs, np.min(group_costs))
            parents.fitness_modes = np.append(parents.fitness_modes, group_fitness_modes[np.argmin(group_costs)])
                
        return parents
