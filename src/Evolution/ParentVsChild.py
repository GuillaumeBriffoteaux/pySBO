import numpy as np

from Evolution.Population import Population
from Evolution.Replacement import Replacement


#---------------------------------------#
#-------------class ParentVsChild-------------#
#---------------------------------------#
class ParentVsChild(Replacement):
    """Class for ParentVsChild replacement.

    Minimization is assumed.
    """
    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#

    #-------------__init__-------------#
    def __init__(self):
        Replacement.__init__(self)

    #-------------__del__-------------#
    def __del__(self):
        Replacement.__del__(self)

    #-------------__str__-------------#
    def __str__(self):
        return "ParentVsChild replacement"


    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#

    #-------------perform_replacement-------------#    
    def perform_replacement(self, pop, children):
        """Keeps the best individuals comparing only the parent and the child
        Must not be used with operators that change the order of individuals in the population.

        :param pop: first population, will store the best individuals
        :type pop: Population
        :param children: second population
        :type children: Population
        """

        Replacement.perform_replacement(self, pop, children)

        if pop.obj_vals.ndim==1:
            idx = pop.obj_vals > children.obj_vals
        # multi-objective
        else:
            # maybe use pygmo crowding distance ?
            raise NotImplementedError

        pop.dvec[idx] = children.dvec[idx]
        pop.obj_vals[idx] = children.obj_vals[idx]
        pop.fitness_modes[idx] = children.fitness_modes[idx]
