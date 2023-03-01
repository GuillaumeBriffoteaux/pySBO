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

        # first tournament
        idx = np.random.choice(np.arange(0, pop.dvec.shape[0], 1, dtype=int), self.__size, replace_mode)
        group_dvec = pop.dvec[idx,:]
        group_obj_vals = pop.obj_vals[idx]
        group_fitness_modes = pop.fitness_modes[idx]

        parents.dvec = group_dvec[np.argmin(group_obj_vals)]
        parents.obj_vals = np.min(group_obj_vals)
        parents.fitness_modes = group_fitness_modes[np.argmin(group_obj_vals)]

        # next tournaments
        for i in range(1, n_par):
            idx = np.random.choice(np.arange(0, pop.dvec.shape[0], 1, dtype=int), self.__size, replace_mode)
            group_dvec = pop.dvec[idx,:]
            group_obj_vals = pop.obj_vals[idx]
            group_fitness_modes = pop.fitness_modes[idx]
            
            parents.dvec = np.vstack( (parents.dvec, group_dvec[np.argmin(group_obj_vals)]) )
            parents.obj_vals = np.append(parents.obj_vals, np.min(group_obj_vals))
            parents.fitness_modes = np.append(parents.fitness_modes, group_fitness_modes[np.argmin(group_obj_vals)])
                
        return parents
