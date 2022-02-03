import numpy as np

from Evolution.Crossover import Crossover
from Evolution.Population import Population


#--------------------------------------------#
#-------------class Intermediate-------------#
#--------------------------------------------#
class Intermediate(Crossover):
    """Class for intermediate crossover (weighted average of decision variables).

    :param prob: probability of crossover
    :type prob: float in [0,1]
    """

    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#

    #-------------__init__-------------#
    def __init__(self, prob=1.0):
        Crossover.__init__(self, prob)
        

    #-------------__del__-------------#
    def __del__(self):
        Crossover.__del__(self)


    #-------------__str__-------------#
    def __str__(self):
        return "Intermediate crossover for Covid-vaccines problem\n  probability "+str(self.prob)

    
    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#
    
    #-------------perform_crossover-------------#
    def perform_crossover(self, pop):
        """Applies crossover to the individuals of a population.

        :param pop: population to mutate
        :type pop: Population
        :returns: the crossed population
        :rtype: Population
        """

        
        Crossover.perform_crossover(self, pop)
        assert isinstance(pop, Population)
        assert pop.dvec.shape[0]%2==0

        children_pop = Population(pop.pb)
        children_pop.dvec = np.copy(pop.dvec)

        alpha = np.random.uniform(size=2)

        i=0
        while i<children_pop.dvec.shape[0]:

            p = np.random.uniform()
            if p<self.prob:

                # first child
                children_pop.dvec[i] = (1.-alpha[0])*pop.dvec[i] + alpha[0]*pop.dvec[i+1]            
                # second child
                children_pop.dvec[i+1] = (1.-alpha[1])*pop.dvec[i] + alpha[1]*pop.dvec[i+1]
            
            i+=2
            
        return children_pop
