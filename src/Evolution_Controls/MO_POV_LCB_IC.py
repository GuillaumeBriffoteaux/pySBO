import numpy as np
import pygmo

from Evolution_Controls.Informed_EC import Informed_EC


#---------------------------------------------#
#-------------class MO_POV_LCB_IC-------------#
#---------------------------------------------#
class MO_POV_LCB_IC(Informed_EC):
    """Class for the IC based on multi-objective POV and LCB from:

    `X. Ruan, K. Li, B. Derbel, and A. Liefooghe. Surrogate assisted evolutionary algorithm for medium scale multi-objective optimisation problems. In Proceedings of the 2020 Genetic and Evolutionary Computation Conference, GECCO 2020, page560–568, New York, NY, USA, 2020. Association for Computing Machinery <https://hal.archives-ouvertes.fr/hal-02932303v1>`_
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
        return "IC based on multi-objective POV and LCB\n  surrogate: {"+self.surr.__str__()+"}"


    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#

    
    #-------------get_sorted_indexes-------------#
    def get_sorted_indexes(self, pop):
        Informed_EC.get_sorted_indexes(self, pop)

        (preds, uncert) = self.surr.perform_prediction(pop.dvec)
        bp_lcb_POV = np.hstack((preds, preds-np.square(uncert)))
        idx = pygmo.sort_population_mo(bp_lcb_POV)

        return idx

    
    #-------------get_IC_value-------------#
    def get_IC_value(self, dvec):
        pass
