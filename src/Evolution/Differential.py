import numpy as np

from Evolution.Mutation import Mutation
from Evolution.Population import Population


#------------------------------------------#
#-------------class Differential Evolution-------------#
#------------------------------------------#
class Differential(Mutation):
    """Class for differential mutation.

    :param weight: Differential weight
    :type weight: float in [0,2]
    :param base: Choice of agent to apply differential weight to
    :type base: string in [rand, best, target-to-best]
    :param diff: Number of difference to add
    :type diff: int > 0
    :param bchm: Box constraint handling method
    :type bchm: string in [reinitialization, projection, reflection, wraping]
    """
    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#

    #-------------__init__-------------#
    def __init__(self, weight, base='rand', diff=1, bchm='projection'):
        assert type(weight)==float
        assert (weight>=0.0 and weight<=2.0)
        self.__weight=weight

        assert type(base)==str
        assert (base=='rand' or base=='best' or base=='target-to-best')
        self.__base = base

        assert type(diff)==int
        assert (diff>0)
        self.__diff = diff

        self.bchm = getattr(self, f'bchm_{bchm}')
        self.prng = np.random.default_rng()

    #-------------__del__-------------#
    def __del__(self):
        del self.__weight
        del self.__base

    #-------------__str__-------------#
    def __str__(self):
        return f"Differential mutation scheme DE/{self.base}/{self.diff} with weight {self.weight} and bchm is {self.bchm.__name__}"


    #---------------------------------------------#
    #-------------getters and setters-------------#
    #---------------------------------------------#

    def _get_weight(self):
        return self.__weight
    def _get_base(self):
        return self.__base
    def _get_diff(self):
        return self.__diff

    #-------------property-------------#
    weight=property(_get_weight, None, None)
    base=property(_get_base, None, None)
    diff=property(_get_diff, None, None)


    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#
    def bounds(self, pop):
        bounds = pop.pb.get_bounds()
        assert bounds.shape[1]==pop.dvec.shape[1]

        lower_bounds = bounds[0,:]
        upper_bounds = bounds[1,:]
        return lower_bounds, upper_bounds


    #-------------Bound Constraint Handling Methods-------------#
    def bchm_reinitialization(self, pop):
        lower_bounds, upper_bounds = self.bounds(pop)
        out_of_bound = np.logical_or(pop.dvec < lower_bounds, pop.dvec > upper_bounds)
        for idx in np.argwhere(out_of_bound):
            idx = tuple(idx)
            pop.dvec[idx] = self.prng.uniform(lower_bounds[idx[1]], upper_bounds[idx[1]])

    def bchm_projection(self, pop):
        lower_bounds, upper_bounds = self.bounds(pop)
        under_bound = pop.dvec < lower_bounds
        for idx in np.argwhere(under_bound):
            idx = tuple(idx)
            pop.dvec[idx] = lower_bounds[idx[1]]

        over_bound = pop.dvec > upper_bounds
        for idx in np.argwhere(over_bound):
            idx = tuple(idx)
            pop.dvec[idx] = upper_bounds[idx[1]]

    def bchm_reflection(self, pop):
        lower_bounds, upper_bounds = self.bounds(pop)
        under_bound = pop.dvec < lower_bounds
        over_bound = pop.dvec > upper_bounds
        while(np.any(under_bound) or np.any(over_bound)):
            for idx in np.argwhere(under_bound):
                idx = tuple(idx)
                pop.dvec[idx] = 2 * lower_bounds[idx[1]] - pop.dvec[idx]

            for idx in np.argwhere(over_bound):
                idx = tuple(idx)
                pop.dvec[idx] = 2 * upper_bounds[idx[1]] - pop.dvec[idx]

            under_bound = pop.dvec < lower_bounds
            over_bound = pop.dvec > upper_bounds

    def bchm_wraping(self, pop):
        lower_bounds, upper_bounds = self.bounds(pop)
        under_bound = pop.dvec < lower_bounds
        over_bound = pop.dvec > upper_bounds
        while(np.any(under_bound) or np.any(over_bound)):
            for idx in np.argwhere(under_bound):
                idx = tuple(idx)
                pop.dvec[idx] = upper_bounds[idx[1]] + pop.dvec[idx] - lower_bounds[idx[1]]

            for idx in np.argwhere(over_bound):
                idx = tuple(idx)
                pop.dvec[idx] = lower_bounds[idx[1]] + pop.dvec[idx] - upper_bounds[idx[1]]

            under_bound = pop.dvec < lower_bounds
            over_bound = pop.dvec > upper_bounds

    #-------------perform_mutation-------------#
    def perform_mutation(self, pop):
        """Mutates the individuals of a population.

        :param pop: population to mutate
        :type pop: Population
        :returns: the mutated population
        :rtype: Population
        """
        Mutation.perform_mutation(self, pop)
        nb_agent_needed = 1 + self.diff * 2
        nb_agent_needed += 1 if self.base == 'rand' else 0
        assert pop.dvec.shape[0] >= nb_agent_needed

        children_pop = Population(pop.pb)
        children_pop.dvec = np.copy(pop.dvec)

        best = pop.obj_vals.argmin()
        ndvec = pop.dvec.shape[1]
        for i in range(ndvec):
            # Choose mutually different agents and different from i
            distribution = np.ones(ndvec) / (ndvec - 1)
            distribution[i] = 0
            e = self.prng.choice(ndvec, size=nb_agent_needed - 1, replace=False, p=distribution)

            vecdiff = np.sum(pop.dvec[e[:self.diff],:], axis=0) - np.sum(pop.dvec[e[self.diff:-1],:], axis=0)

            if self.base == 'best':
                children_pop.dvec[i] = pop.dvec[best] + self.weight * vecdiff
            elif self.base == 'target-to-best':
                children_pop.dvec[i] = pop.dvec[i] + self.weight * (pop.dvec[best] - pop.dvec[i]) + self.weight * vecdiff
            else: # 'rand'
                children_pop.dvec[i] = pop.dvec[e[-1]] + self.weight * vecdiff

        self.bchm(children_pop)

        return children_pop
