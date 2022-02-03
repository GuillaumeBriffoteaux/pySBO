import numpy as np

from Evolution_Controls.Informed_EC import Informed_EC


#--------------------------------------#
#-------------class POV_EC-------------#
#--------------------------------------#
class POV_EC(Informed_EC):
    """Class for best predicted EC.

    Candidates with lower predicted objective value (POV)
    by the surrogate are more promising.
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
        return "Predicted Objective Value Evolution Control\n  surrogate: {"+self.surr.__str__()+"}"


    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#
    
    #-------------get_sorted_indexes-------------#
    def get_sorted_indexes(self, pop):
        Informed_EC.get_sorted_indexes(self, pop)

        preds = self.surr.perform_prediction(pop.dvec)[0]
        idx = np.argsort(preds)
        
        return idx

    #-------------get_IC_value-------------#    
    def get_IC_value(self, dvec):
        Informed_EC.get_IC_value(self, dvec)
        return self.surr.perform_prediction(dvec)[0]
