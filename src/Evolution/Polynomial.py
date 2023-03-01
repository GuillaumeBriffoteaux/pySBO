import numpy as np

from Evolution.Mutation import Mutation
from Evolution.Population import Population


#------------------------------------------#
#-------------class Polynomial-------------#
#------------------------------------------#
class Polynomial(Mutation):
    """Class for polynomial mutation.

    :param prob: probability of mutation
    :type prob: float in [0,1]
    :param eta: distribution index
    :type eta: positive int, not zero
    """
    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#

    #-------------__init__-------------#
    def __init__(self, prob, eta):
        Mutation.__init__(self, prob)
        assert type(eta)==int
        assert eta>0
        self.__eta=eta

    #-------------__del__-------------#
    def __del__(self):
        Mutation.__del__(self)
        del self.__eta

    #-------------__str__-------------#
    def __str__(self):
        return "Polynomial mutation probability "+str(self.prob)+" distribution index "+str(self.__eta)


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

    #-------------perform_mutation-------------#    
    def perform_mutation(self, pop):
        """Mutates the individuals of a population.

        :param pop: population to mutate
        :type pop: Population
        :returns: the mutated population
        :rtype: Population
        """
        Mutation.perform_mutation(self, pop)
        bounds = pop.pb.get_bounds()
        assert bounds.shape[1]==pop.dvec.shape[1]

        children = Population(pop.pb)
        children.dvec = np.copy(pop.dvec)

        # Loop over the children population
        for child in children.dvec:

            # choosing the decision variables to mutate
            nb_dvar_to_mutate = np.random.binomial(child.size, self.prob)
            if self.prob>0.0 and nb_dvar_to_mutate==0:
                nb_dvar_to_mutate=1
            dvar_to_mutate = np.random.choice(np.arange(0, child.size, 1, dtype=int), nb_dvar_to_mutate, replace=False)
            # Loop over the decision variables to mutate
            for i in dvar_to_mutate:
                mu = np.random.uniform()
                if mu<=0.5:
                    delta_l = pow(2.0*mu, 1.0/(1.0+self.__eta)) - 1.0
                    child[i] += delta_l*(child[i]-bounds[0,i])
                else:
                    delta_r = 1.0 - pow(2.0*(1.0-mu), 1.0/(1.0+self.__eta))
                    child[i] += delta_r*(bounds[1,i]-child[i])

        return children
