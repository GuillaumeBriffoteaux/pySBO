import numpy as np

from Evolution.Population import Population
from Evolution.Replacement import Replacement
from Evolution_Controls.Evolution_Control import Evolution_Control


#----------------------------------------------#
#-------------class Custom_Elitism-------------#
#----------------------------------------------#
class Custom_Elitism(Replacement):
    """Class for custom elitist replacement.

    :param ec: the criterion defining elitism
    :type ec: Surrogate
    """
    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#

    #-------------__init__-------------#
    def __init__(self, ec):
        Replacement.__init__(self)
        assert isinstance(ec, Evolution_Control)
        self.__ec = ec

    #-------------__del__-------------#
    def __del__(self):
        Replacement.__init__(self)
        del self.__ec

    #-------------__str__-------------#
    def __str__(self):
        return "Custom Elistist replacement\n  Evolution Control: "+str(self.__ec)


    #---------------------------------------------#
    #-------------getters and setters-------------#
    #---------------------------------------------#

    
    #-------------_get_ec-------------#
    def _get_ec(self):
        return self.__ec

    #-------------property-------------#
    ec=property(_get_ec, None, None)


    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#    

    #-------------perform_replacement-------------#    
    def perform_replacement(self, pop, children):
        """Keeps the best individuals from two populations.

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

        # ordering according to the promise
        ordering = self.__ec.get_sorted_indexes(merged_pop)

        # retaining
        pop.dvec = merged_pop.dvec[ordering[0:pop.dvec.shape[0]]]
