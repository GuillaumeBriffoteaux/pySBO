import numpy as np

from Evolution.Crossover import Crossover
from Evolution.Population import Population


#------------------------------------------#
#-------------class Two_Points-------------#
#------------------------------------------#
class Binomial(Crossover):
    """Class for Binomial crossover.

    :param prob: probability of crossover
    :type prob: float in [0,1]
    :param at_least_one: change at least one element
    :type at_least_one: bool
    """

    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#

    #-------------__init__-------------#
    def __init__(self, prob=1.0, at_least_one=False):
        Crossover.__init__(self, prob)
        assert type(at_least_one)==bool
        self.__at_least_one=at_least_one

        self.prng = np.random.default_rng()

    #-------------__del__-------------#
    def __del__(self):
        Crossover.__del__(self)
        del self.__at_least_one

    #-------------__str__-------------#
    def __str__(self):
        return f"Binomial crossover probability {self.prob}{' and change at least one' if self.__at_least_one else ''}"
    
    #---------------------------------------------#
    #-------------getters and setters-------------#
    #---------------------------------------------#
    
    #-------------_get_prob-------------#
    def _get_at_least_one(self):
        return self.__at_least_one

    #-------------property-------------#
    at_least_one=property(_get_at_least_one, None, None)

    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#
    
    #-------------perform_crossover-------------#
    def perform_crossover(self, pop, mutant_pop):
        """Applies crossover to the individuals of a population.

        :param pop: population of parents
        :type pop: Population
        :param mutant_pop: population of mutants
        :type mutant_pop: Population
        :returns: the crossed population
        :rtype: Population
        """
        
        Crossover.perform_crossover(self, pop)
        assert isinstance(mutant_pop, Population)

        children_pop = Population(pop.pb)
        children_pop.dvec = np.copy(pop.dvec)

        idx = self.prng.uniform(size=pop.dvec.shape) < self.prob
        children_pop.dvec[idx] = mutant_pop.dvec[idx]

        if self.at_least_one:
            # choose one random element of each row and change at least this one
            R = self.prng.integers(pop.dvec.shape[1], size=pop.dvec.shape[0])
            idx = np.s_[np.arange(pop.dvec.shape[0]),R]
            children_pop.dvec[idx] = mutant_pop.dvec[idx]
           
        return children_pop
