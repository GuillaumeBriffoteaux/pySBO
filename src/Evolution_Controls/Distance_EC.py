import numpy as np

from Evolution_Controls.Informed_EC import Informed_EC


#-------------------------------------------#
#-------------class Distance_EC-------------#
#-------------------------------------------#
class Distance_EC(Informed_EC):
    """Class for distance EC.

    Candidates with greater distance from the set of already simulated candidates are more promising.
    """

    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#

    #-------------__init__-------------#
    def __init__(self, surr):
        """
        __init__ method's input
        
        :param surr: surrogate model
        :type surr: Surrogate
        """

        Informed_EC.__init__(self, surr)
    
    #-------------__del__-------------#
    def __del__(self):
        Informed_EC.__del__(self)

    #-------------__str__-------------#
    def __str__(self):
        return "Distance Evolution Control\n  surrogate: {"+self.surr.__str__()+"}"


    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#
    
    #-------------get_sorted_indexes-------------#
    def get_sorted_indexes(self, pop):
        Informed_EC.get_sorted_indexes(self, pop)

        dists = self.surr.perform_distance(pop.dvec)
        idx = np.argsort(-dists)

        return idx

    #-------------get_IC_value-------------#
    def get_IC_value(self, dvec):

        Informed_EC.get_IC_value(self, dvec)
        return -self.surr.perform_distance(dvec)
