import numpy as np

from Evolution_Controls.Informed_EC import Informed_EC
from Evolution_Controls.Ensemble_EC import Ensemble_EC
import matplotlib.pyplot as plt
import pygmo


#--------------------------------------------------#
#-------------class Pareto_Tian2018_EC-------------#
#--------------------------------------------------#
class Pareto_Tian2018_EC(Ensemble_EC):
    """Class for bi-objective Pareto-based EC from Tian-2018.

    Candidates with lowest non-dominated and highest rank according to minimization of the ECs (multiplied by their respective coefficient 1 or -1) are more promising. Then, Candidates with increasing non-dominated rank are increasingly promising.

    The EC is described in :
    `J. Tian, Y. Tan, J. Zeng, C. Sun and Y. Jin. Multi-objective Infill Criterion Driven Gaussian Process-Assisted Particle Swarm Optimization of High-Dimensional Expensive Problems. In IEEE Transactions on Evolutionary Computation 23.3 (June 2019), pp. 459–472. ISSN: 1941-0026. <https://doi.org/10.1109/TEVC.2018.2869247>`_

    :param coeffs: coefficients (1 or -1) to multiply the EC with (allow to convert a minimization problem into a maximization problem and reversely)
    :type coeffs: np.ndarray
    """
    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#

    #-------------__init__-------------#
    def __init__(self, coeffs, *ECs):
        """
        __init__ method's input

        :param ECs: evolution controls
        :type ECs: list(Evolution_Control)
        """
        
        Ensemble_EC.__init__(self, *ECs)
        for ec in self.ECs_list:
            assert isinstance(ec, Informed_EC)

        assert type(coeffs)==list
        assert len(coeffs)==len(ECs)

        self.__coeffs=np.zeros((len(coeffs),))
        for i,mode in enumerate(coeffs):
            self.__coeffs[i] = mode
    
    #-------------__del__-------------#
    def __del__(self):
        Ensemble_EC.__del__(self)
        del self.__coeffs

    #-------------__str__-------------#
    def __str__(self):
        res="Pareto-based Ensemble Evolution Control (Tian-2018)\n  coeffs:{"
        for mode in self.__coeffs:
            res+=" "+str(mode)
        res+=" }\n  ECs: {"
        for ec in self.ECs_list:
            res+=" "+ec.__class__.__name__
        res+=" }"
        return res


    #---------------------------------------------#
    #-------------getters and setters-------------#
    #---------------------------------------------#

    #-------------_get_coeffs-------------#
    def _get_coeffs(self):
        return self.__coeffs
        
    #-------------property-------------#
    coeffs=property(_get_coeffs, None, None)


    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#
    
    #-------------get_sorted_indexes-------------#
    def get_sorted_indexes(self, pop):
        Ensemble_EC.get_sorted_indexes(self, pop)

        criteria = np.empty((len(self.ECs_list), pop.dvec.shape[0]))
        for i,ec in enumerate(self.ECs_list):
            criteria[i] = self.__coeffs[i]*ec.get_IC_value(pop.dvec)
        criteria = criteria.T

        (ndf, dom_list, dom_count, ndr) = pygmo.fast_non_dominated_sorting(criteria)
        idx = np.argsort(ndr)

        idx_rank_zero = np.where(ndr==0)[0]
        idx_rank_max = np.where(ndr==ndr[idx[idx.size-1]])[0]
        idx = np.concatenate((idx_rank_zero, idx_rank_max, idx[idx_rank_zero.size:idx.size-idx_rank_max.size]))

        return idx
