import numpy as np
import scipy as sp

from Evolution_Controls.Informed_EC import Informed_EC
from Global_Var import *


#-------------------------------------------------------#
#-------------class Expected_Improvement_EC-------------#
#-------------------------------------------------------#
class Expected_Improvement_EC(Informed_EC):
    """Class for expected improvement EC.

    Candidates with greater expected improvement are more promising.
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
        return "Expected Improvement Adaptive Evolution Control\n  surrogate: {"+self.surr.__str__()+"}"


    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#
    
    #-------------get_sorted_indexes-------------#
    def get_sorted_indexes(self, pop):
        Informed_EC.get_sorted_indexes(self, pop)

        (preds, uncert) = self.surr.perform_prediction(pop.dvec)

        # treat division by zero case
        idx_zero_var=np.where(uncert<1e-100)
        uncert[idx_zero_var[0]]=1
        eis = (Global_Var.cost_min-preds) * sp.stats.norm.cdf((Global_Var.cost_min-preds)/uncert) + uncert * sp.stats.norm.pdf((Global_Var.cost_min-preds)/uncert)
        eis[idx_zero_var[0]]=(Global_Var.cost_min-preds[idx_zero_var[0]])
        
        idx = np.argsort(-eis)

        return idx

    #-------------get_IC_value-------------#
    def get_IC_value(self, dvec):
        Informed_EC.get_IC_value(self, dvec)
        (preds, uncert) = self.surr.perform_prediction(dvec)

        return -((Global_Var.cost_min-preds) * sp.stats.norm.cdf((Global_Var.cost_min-preds)/uncert) + uncert * sp.stats.norm.pdf((Global_Var.cost_min-preds)/uncert))
