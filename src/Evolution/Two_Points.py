import numpy as np

from Evolution.Crossover import Crossover
from Evolution.Population import Population


#------------------------------------------#
#-------------class Two_Points-------------#
#------------------------------------------#
class Two_Points(Crossover):
    """Class for 2-point crossover.

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
        return "2-points crossover  probability "+str(self.prob)
    

    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#
    
    #-------------perform_crossover-------------#
    def perform_crossover(self, pop):
        """Applies crossover to the individuals of a population.

        :param pop: population of parents
        :type pop: Population
        :returns: the crossed population
        :rtype: Population
        """
        
        Crossover.perform_crossover(self, pop)
        assert isinstance(pop, Population)
        assert pop.dvec.shape[0]%2==0

        children_pop = Population(pop.pb)
        children_pop.dvec = np.copy(pop.dvec)

        i=0
        while i<children_pop.dvec.shape[0]:

            p1 = np.random.randint(0,children_pop.dvec.shape[1])
            p2 = np.random.randint(p1+1,children_pop.dvec.shape[1]+1)

            p = np.random.uniform()
            if p<self.prob:

                children_pop.dvec[i,p1:p2] = pop.dvec[i+1,p1:p2]
                children_pop.dvec[i+1,p1:p2] = pop.dvec[i,p1:p2]
            
            i+=2
            
        return children_pop
