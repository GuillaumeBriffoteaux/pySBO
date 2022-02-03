import numpy as np
import scipy as sp

from Evolution_Controls.Informed_EC import Informed_EC
from Global_Var import *


#----------------------------------------------------------#
#-------------class Probability_Improvement_EC-------------#
#----------------------------------------------------------#
class Probability_Improvement_EC(Informed_EC):
    """Class for probability of improvement EC.

    Candidates with greater probability of improvement are more promising.
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
        return "Probability Improvement Evolution Control\n  surrogate: {"+self.surr.__str__()+"}"


    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#
    
    #-------------get_sorted_indexes-------------#
    def get_sorted_indexes(self, pop):
        Informed_EC.get_sorted_indexes(self, pop)

        (preds, uncert) = self.surr.perform_prediction(pop.dvec)
        norm_y_min = self.surr.normalize_obj_vals(Global_Var.obj_val_min)[0]

        idx_nonzero_stdev=np.where(uncert>1e-16)
        pis = np.zeros(preds.shape)
        
        pis[idx_nonzero_stdev] = (0.5 + 0.5 * sp.special.erf((norm_y_min - preds[idx_nonzero_stdev])/(uncert[idx_nonzero_stdev]*np.sqrt(2.0))))
        idx = np.argsort(-pis)

        return idx

    #-------------get_IC_value-------------#
    def get_IC_value(self, dvec):
        Informed_EC.get_IC_value(self, dvec)

        (preds, uncert) = self.surr.perform_prediction(dvec)
        norm_y_min = self.surr.normalize_obj_vals(Global_Var.obj_val_min)[0]

        idx_nonzero_stdev=np.where(uncert>1e-16)
        pis = np.zeros(preds.shape)
        
        pis[idx_nonzero_stdev] = (0.5 + 0.5 * sp.special.erf((norm_y_min - preds[idx_nonzero_stdev])/(uncert[idx_nonzero_stdev]*np.sqrt(2.0))))
        return -pis
