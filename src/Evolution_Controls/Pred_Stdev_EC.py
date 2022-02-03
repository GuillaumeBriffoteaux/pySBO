import numpy as np

from Evolution_Controls.Informed_EC import Informed_EC


#---------------------------------------------#
#-------------class Pred_Stdev_EC-------------#
#---------------------------------------------#
class Pred_Stdev_EC(Informed_EC):
    """Class for predictive standard deviation EC.

    Candidates with greater predictive standard deviation are more promising.
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
        return "Predictive Standard Deviation Evolution Control\n  surrogate: {"+self.surr.__str__()+"}"


    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#
    
    #-------------get_sorted_indexes-------------#
    def get_sorted_indexes(self, pop):
        Informed_EC.get_sorted_indexes(self, pop)

        uncert = self.surr.perform_prediction(pop.dvec)[1]
        idx = np.argsort(-uncert)

        return idx

    #-------------get_IC_value-------------#
    def get_IC_value(self, dvec):
        Informed_EC.get_IC_value(self, dvec)
        return -self.surr.perform_prediction(dvec)[1]
