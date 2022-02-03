import numpy as np

from Evolution_Controls.Informed_EC import Informed_EC


#--------------------------------------------------------#
#-------------class Lower_Confident_Bound_EC-------------#
#--------------------------------------------------------#
class Lower_Confident_Bound_EC(Informed_EC):
    """Class for lower confident bound EC.

    Candidates with lower lower confident bound are more promising.

    :param w: weight for LCB
    :type w: float in [0; 3]
    """

    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#

    #-------------__init__-------------#
    def __init__(self, surr, w=1.0):
        """
        __init__ method's input
        
        :param surr: surrogate model
        :type surr: Surrogate
        """

        Informed_EC.__init__(self, surr)
        self.__w=w
    
    #-------------__del__-------------#
    def __del__(self):
        Informed_EC.__del__(self)

    #-------------__str__-------------#
    def __str__(self):
        return "Lower Confident Bound Evolution Control\n  surrogate: {"+self.surr.__str__()+"}"


    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#
    
    #-------------get_sorted_indexes-------------#
    def get_sorted_indexes(self, pop):
        Informed_EC.get_sorted_indexes(self, pop)

        (preds, uncert) = self.surr.perform_prediction(pop.dvec)        
        lcbs = preds-self.__w*uncert
        idx = np.argsort(lcbs)

        return idx

    #-------------get_IC_value-------------#
    def get_IC_value(self, dvec):
        Informed_EC.get_IC_value(self, dvec)
        (preds, uncert) = self.surr.perform_prediction(dvec)
        lcbs = preds-self.__w*uncert
        return lcbs
