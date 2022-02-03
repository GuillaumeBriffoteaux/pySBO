import numpy as np

from Evolution.Population import Population
from Evolution.Replacement import Replacement


#---------------------------------------#
#-------------class Elitist-------------#
#---------------------------------------#
class Elitist(Replacement):
    """Class for Elitist replacement.

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
        return "Elistist replacement"


    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#

    #-------------perform_replacement-------------#    
    def perform_replacement(self, pop, children):
        """Keeps the best individuals of from two populations.

        :param pop: first population, will store the best individuals
        :type pop: Population
        :param children: second population
        :type children: Population
        """

        Replacement.perform_replacement(self, pop, children)

        # merging
        merged_pop = Population(pop.pb)        
        merged_pop.append(pop)
        merged_pop.append(children)

        # sorting
        merged_pop.sort()

        # retaining
        pop.dvec = merged_pop.dvec[0:pop.dvec.shape[0]]
        pop.obj_vals = merged_pop.obj_vals[0:pop.dvec.shape[0]]
        pop.fitness_modes = merged_pop.fitness_modes[0:pop.dvec.shape[0]]

        # adding best predicted 
        if False not in pop.fitness_modes and False in merged_pop.fitness_modes:
            pop.dvec[pop.dvec.shape[0]-1] = merged_pop.dvec[np.where(merged_pop.fitness_modes==False)[0][0]]
            pop.obj_vals[pop.obj_vals.shape[0]-1] = merged_pop.obj_vals[np.where(merged_pop.fitness_modes==False)[0][0]]
            pop.fitness_modes[pop.fitness_modes.size-1] = False
