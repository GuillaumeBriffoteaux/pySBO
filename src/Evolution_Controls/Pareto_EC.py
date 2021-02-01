import numpy as np

from Evolution_Controls.Ensemble_EC import Ensemble_EC
import matplotlib.pyplot as plt
import pygmo


#-----------------------------------------#
#-------------class Pareto_EC-------------#
#-----------------------------------------#
class Pareto_EC(Ensemble_EC):
    """Class for Pareto-based EC.

    Candidates with lower non-dominated rank according to minimization of the ECs (multiplied by their respective coefficient 1 or -1) are more promising. 

    :param coeffs: coefficients (1 or -1) to multiply the EC with (allow to convert a minimization problem into a maximization problem and reversely)
    :type coeffs: np.ndarray
    :param distinct_mode: criterion to distinguish solutions with same non domination rank. When equals to 'rand' solutions considered as more promising are choosen randomly. When equals 'cd' solutions with higher crowded distance are considered as more promising. When equals to 'hvc', solutions with higher hypervolume contribution are considered as more promising.
    """
    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#

    #-------------__init__-------------#
    def __init__(self, coeffs, distinct_mode, *ECs):
        """
        __init__ method's input

        :param ECs: evolution controls
        :type ECs: list(Evolution_Control)
        """
        
        Ensemble_EC.__init__(self, *ECs)
        assert type(coeffs)==list
        assert len(coeffs)==len(ECs)
        assert type(distinct_mode)==str
        assert distinct_mode=='rand' or distinct_mode=='cd' or distinct_mode=='hvc'

        self.__coeffs=np.zeros((len(coeffs),))
        for i,mode in enumerate(coeffs):
            self.__coeffs[i] = mode
        self.__distinct_mode = distinct_mode
    
    #-------------__del__-------------#
    def __del__(self):
        Ensemble_EC.__del__(self)
        del self.__coeffs
        del self.__distinct_mode

    #-------------__str__-------------#
    def __str__(self):
        res="Pareto-based Ensemble Evolution Control\n  coeffs:{"
        for mode in self.__coeffs:
            res+=" "+str(mode)
        res+=" }\n  distinction mode: "+self.__distinct_mode+"\n  ECs: {"
        for ec in self.ECs_list:
            res+=" "+ec.__class__.__name__
        res+=" }"
        return res


    #---------------------------------------------#
    #-------------getters and setters-------------#
    #---------------------------------------------#

    #-------------_get_coeffs-------------#
    def _get_coeffs(self):
        print("[Pareto_EC.py] Impossible to get the list of coeffs")
        return self.__coeffs

    #-------------_set_coeffs-------------#
    def _set_coeffs(self, new_coeffs):
        print("[Pareto_EC.py] Impossible to modify the list of coeffs")

    #-------------_del_coeffs-------------#
    def _del_coeffs(self):
        print("[Pareto_EC.py] Impossible to delete the list of coeffs")

    #-------------_get_distinct_mode-------------#
    def _get_distinct_mode(self):
        print("[Pareto_EC.py] Impossible to get the distinction mode")
        return self.__distinct_mode

    #-------------_set_distinct_mode-------------#
    def _set_distinct_mode(self, new_distinct_mode):
        print("[Pareto_EC.py] Impossible to modify the distinction mode")

    #-------------_del_distinct_mode-------------#
    def _del_distinct_mode(self):
        print("[Pareto_EC.py] Impossible to delete the distinction mode")
        
    #-------------property-------------#
    coeffs=property(_get_coeffs, _set_coeffs, _del_coeffs)
    distinct_mode=property(_get_distinct_mode, _set_distinct_mode, _del_distinct_mode)


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

        if self.__distinct_mode=='rand':
            (ndf, dom_list, dom_count, ndr) = pygmo.fast_non_dominated_sorting(criteria)
            idx = np.argsort(ndr)
        elif self.__distinct_mode=='cd':
            idx = pygmo.sort_population_mo(criteria)
        elif self.__distinct_mode=='hvc':
            (ndf, dom_list, dom_count, ndr) = pygmo.fast_non_dominated_sorting(criteria)
            idx = np.array([], dtype=int)
            for i in range(int(ndr.max()+1)):
                if np.where(ndr==i)[0].size>1:
                    hv = pygmo.hypervolume(criteria[np.where(ndr==i)[0]])
                    ref_point = pygmo.nadir(criteria[np.where(ndr==i)[0]])+1
                    idx = np.append(idx, np.where(ndr==i)[0][np.argsort(-hv.contributions(ref_point))])
                else:
                    idx = np.append(idx, np.where(ndr==i)[0])
        else:
            print("[Pareto_EC.py] error: dinstinction mode is wrong")
            assert False
            
        return idx
