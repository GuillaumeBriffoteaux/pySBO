import numpy as np
import math

from Evolution_Controls.Informed_EC import Informed_EC


#----------------------------------------------------#
#-------------class Adaptive_Wang2020_EC-------------#
#----------------------------------------------------#
class Adaptive_Wang2020_EC(Informed_EC):
    """Class for Wang-2020's adaptive EC.

    The EC is described in:
    `X. Wang, Y. Jin, S. Schmitt and M. Olhofer. An adaptive Bayesian approach to surrogate-assisted evolutionary multi-objective optimization. In Information Sciences 519 (2020), pp. 317–331. ISSN: 0020-0255. <https://doi.org/10.1016/j.ins.2020.01.048>`_

    At the early stage of the search, minimization of the predicted
    objective value prevails (favoring fast convergence). As the search 
    progresses, more importance is given to uncertainty minimization
    (favoring exploitation) or uncertainty maximization (favoring
    exploration).

    :param search_budget: search budget (expressed either in number of generations, number of cycles, number of batches or time)
    :type search_budget: positive int, not zero
    :param weight: weight (alpha in Wang-2020)
    :type weight: float
    :param uncert_treatment: 1 for uncertainty minimization, -1 for uncertainty maximization
    :type uncert_treatment: int
    """
    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#

    #-------------__init__-------------#
    def __init__(self, surr, search_budget, uncert_treatment):
        """
        __init__ method's input
        
        :param surr: surrogate model
        :type surr: Surrogate
        :param uncert_treatment: "min" for uncertainty minimization, "max" for uncertainty maximization
        :type uncert_treatment: str
        """
        
        Informed_EC.__init__(self, surr)
        assert type(search_budget)==int
        self.__search_budget=search_budget
        self.__weight = 0.0
        assert uncert_treatment=="min" or uncert_treatment=="max"
        if uncert_treatment=="min":
            self.__uncert_treatment=1
        elif uncert_treatment=="max":
            self.__uncert_treatment=-1
        else:
            print("[Adaptive_Wang2020_EC.py] uncert_treament should be either 'min' or 'max'")
            assert False
    
    #-------------__del__-------------#
    def __del__(self):
        Informed_EC.__del__(self)
        del self.__search_budget
        del self.__weight
        del self.__uncert_treatment

    #-------------__str__-------------#
    def __str__(self):
        return "Wang-2020's adaptive EC\n  surrogate: {"+self.surr.__str__()+"}\n  search budget: "+str(self.__search_budget)+"\n  weight: "+str(self.__weight)+"\n  uncertainty treatment: "+str(self.__uncert_treatment)


    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#

    
    #-------------get_sorted_indexes-------------#
    def get_sorted_indexes(self, pop):
        Informed_EC.get_sorted_indexes(self, pop)

        (preds, uncert) = self.surr.perform_prediction(pop.dvec)
        af = (1-self.__weight)*preds/np.amax(preds) + self.__uncert_treatment*self.__weight*uncert/np.amax(uncert)        
        idx = np.argsort(af)

        return idx

    
    #-------------get_IC_value-------------#
    def get_IC_value(self, dvec):
        Informed_EC.get_IC_value(self, dvec)        
        (preds, uncert) = self.surr.perform_prediction(pop.dvec)
        af = (1-self.__weight)*preds/np.amax(preds) + self.__uncert_treatment*self.__weight*uncert/np.amax(uncert)
            
        return af

    
    #-------------update_EC-------------#
    def update_EC(self, search_progress):
        """Set the weight according to the search progress.

        :param search_progress: current search progress (expressed either in number of generations, number of cycles, number of batches or time)
        :type search_progress: positive int
        """

        # search_progress has to be expressed in the same unit as search_budget
        self.__weight = -0.5*math.cos(search_progress*math.pi/self.__search_budget) + 0.5

        
