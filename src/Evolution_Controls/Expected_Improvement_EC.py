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
        return "Expected Improvement Evolution Control\n  surrogate: {"+self.surr.__str__()+"}"


    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#
    
    #-------------get_sorted_indexes-------------#
    def get_sorted_indexes(self, pop):
        Informed_EC.get_sorted_indexes(self, pop)

        (preds, uncert) = self.surr.perform_prediction(pop.dvec)
        norm_y_min = self.surr.normalize_obj_vals(Global_Var.obj_val_min)[0]
        
        idx_nonzero_stdev=np.where(uncert>1e-16)
        eis = np.zeros(preds.shape)
        
        eis[idx_nonzero_stdev] = ((norm_y_min - preds[idx_nonzero_stdev]) * (0.5 + 0.5*sp.special.erf((1./np.sqrt(2.))*((norm_y_min - preds[idx_nonzero_stdev]) / uncert[idx_nonzero_stdev])))) + ((uncert[idx_nonzero_stdev] * (1. / np.sqrt(2. * np.pi))) * (np.exp(-(1./2.) * ((norm_y_min - preds[idx_nonzero_stdev])**2. / uncert[idx_nonzero_stdev]**2.))))
        idx = np.argsort(-eis)

        return idx

    #-------------get_IC_value-------------#
    def get_IC_value(self, dvec):
        Informed_EC.get_IC_value(self, dvec)
        
        (preds, uncert) = self.surr.perform_prediction(dvec)
        norm_y_min = self.surr.normalize_obj_vals(Global_Var.obj_val_min)[0]
        
        idx_nonzero_stdev=np.where(uncert>1e-16)
        eis = np.zeros(preds.shape)
        
        eis[idx_nonzero_stdev] = ((norm_y_min - preds[idx_nonzero_stdev]) * (0.5 + 0.5*sp.special.erf((1./np.sqrt(2.))*((norm_y_min - preds[idx_nonzero_stdev]) / uncert[idx_nonzero_stdev])))) + ((uncert[idx_nonzero_stdev] * (1. / np.sqrt(2. * np.pi))) * (np.exp(-(1./2.) * ((norm_y_min - preds[idx_nonzero_stdev])**2. / uncert[idx_nonzero_stdev]**2.))))

        return -eis
