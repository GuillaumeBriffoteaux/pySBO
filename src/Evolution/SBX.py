import numpy as np

from Evolution.Crossover import Crossover
from Evolution.Population import Population


#-----------------------------------#
#-------------class SBX-------------#
#-----------------------------------#
class SBX(Crossover):
    """Class for SBX crossover.

    :param prob: probability of crossover
    :type prob: float in [0,1]
    :param eta: distribution index
    :type eta: positive int, not zero
    """
    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#

    #-------------__init__-------------#
    def __init__(self, prob, eta):
        Crossover.__init__(self, prob)
        assert type(eta)==int
        assert eta>0
        self.__eta=eta

    #-------------__del__-------------#
    def __del__(self):
        Crossover.__del__(self)
        del self.__eta

    #-------------__str__-------------#
    def __str__(self):
        return "Simulated Binary Crossover probability "+str(self.prob)+" distribution index "+str(self.__eta)


    #---------------------------------------------#
    #-------------getters and setters-------------#
    #---------------------------------------------#

    #-------------_get_eta-------------#
    def _get_eta(self):
        return self.__eta

    #-------------property-------------#
    eta=property(_get_eta, None, None)


    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#
    
    #-------------perform_crossover-------------#
    def perform_crossover(self, pop):
        """Applies crossover to the individuals of a population.

        :param pop: population to mutate
        :type pop: Population
        :returns: the mutated population
        :rtype: Population
        """
        
        Crossover.perform_crossover(self, pop)
        assert isinstance(pop, Population)
        assert pop.dvec.shape[0]%2==0
        bounds = pop.pb.get_bounds()
        assert bounds.shape[1]==pop.dvec.shape[1]

        children_pop = Population(pop.pb)
        children_pop.dvec = np.copy(pop.dvec)

        it = iter(children_pop.dvec)
        # Loop over the population
        for childA in it:
            childB=next(it)

            if np.random.uniform()<=self.prob:

                # Loop over the decision variables
                for i in range(childA.size):

                    if np.random.uniform()<=0.5 and abs(childA[i]-childB[i])>1e-14 and bounds[0,i]!=bounds[1,i]:
                        
                        if childA[i]<childB[i]:
                            y1 = childA[i]
                            y2 = childB[i]
                        else:
                            y2 = childA[i]
                            y1 = childB[i]
                        
                        yl = bounds[0,i]
                        yu = bounds[1,i]

                        mu = np.random.uniform()
                        
                        beta = 1.0 + (2.0 * (y1 - yl) / (y2 - y1))
                        alpha = 2.0 - pow(beta, -(self.__eta + 1.0))
                        if mu<=(1.0/alpha):
                            betaq = pow( (mu * alpha), (1.0 / (self.__eta + 1.0)) )
                        else:
                            betaq = pow( (1.0 / (2.0-mu*alpha)),  (1.0 / (self.__eta + 1.0)) )
                        c1 = 0.5 * ((y1 + y2) - betaq * (y2 - y1))

                        beta = 1.0 + (2.0 * (yu - y2) / (y2 - y1))
                        alpha = 2.0 - pow(beta, -(self.__eta + 1.0))
                        if mu<=(1.0/alpha):
                            betaq = pow( (mu * alpha), (1.0 / (self.__eta + 1.0)) )
                        else:
                            betaq = pow( (1.0 / (2.0-mu*alpha)),  (1.0 / (self.__eta + 1.0)) )
                        c2 = 0.5 * ((y1 + y2) + betaq * (y2 - y1))

                        if c1<yl:
                            c1 = yl
                        if c2<yl:
                            c2 = yl
                        if c1>yu:
                            c1 = yu
                        if c2>yu:
                            c2 = yu

                        if np.random.uniform()>=0.5:
                            childA[i] = c1
                            childB[i] = c2
                        else:
                            childA[i] = c2
                            childB[i] = c1
        
        return children_pop
