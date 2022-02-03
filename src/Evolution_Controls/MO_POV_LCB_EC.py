import numpy as np
import pygmo
import math

from Evolution_Controls.Informed_EC import Informed_EC


#---------------------------------------------#
#-------------class MO_POV_LCB_EC-------------#
#---------------------------------------------#
class MO_POV_LCB_EC(Informed_EC):
    """Class for the EC based on multi-objective POV and LCB from:

    `X. Ruan, K. Li, B. Derbel, and A. Liefooghe. Surrogate assisted evolutionary algorithm for medium scale multi-objective optimisation problems. In Proceedings of the 2020 Genetic and Evolutionary Computation Conference, GECCO 2020, page560–568, New York, NY, USA, 2020. Association for Computing Machinery <https://hal.archives-ouvertes.fr/hal-02932303v1>`_

    :param q: number of candidates to retain for simulation
    :type q: int
    """
    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#

    #-------------__init__-------------#
    def __init__(self, surr, q):
        """
        __init__ method's input
        
        :param surr: surrogate model
        :type surr: Surrogate
        """

        Informed_EC.__init__(self, surr)
        self.__q = q
    
    #-------------__del__-------------#
    def __del__(self):
        Informed_EC.__del__(self)

    #-------------__str__-------------#
    def __str__(self):
        return "EC based on multi-objective POV and LCB\n  surrogate: {"+self.surr.__str__()+"}"


    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#

    
    #-------------get_sorted_indexes-------------#
    def get_sorted_indexes(self, pop):
        Informed_EC.get_sorted_indexes(self, pop)

        (preds, uncert) = self.surr.perform_prediction(pop.dvec)
        lcb = preds-2*np.square(uncert)

        ref_point_bp = np.amax(preds, axis=0)+1
        ref_point_lcb = np.amax(lcb, axis=0)+1

        hv_bp = pygmo.hypervolume(preds)
        hv_lcb = pygmo.hypervolume(lcb)
        
        idx_bp = np.argsort(-hv_bp.contributions(ref_point_bp))
        idx_lcb = np.argsort(-hv_lcb.contributions(ref_point_lcb))

        q_half = math.floor(self.__q/2)

        idx = np.array([], dtype=int)
        idx = np.append(idx, idx_bp[0:q_half])
        idx_bp = idx_bp[q_half:]
        idx_lcb = idx_lcb[~np.isin(idx_lcb, idx)]

        idx = np.append(idx, idx_lcb)

        return idx

    
    #-------------get_IC_value-------------#
    def get_IC_value(self, dvec):
        pass
