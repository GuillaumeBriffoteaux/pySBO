import numpy as np

from Evolution_Controls.Ensemble_EC import Ensemble_EC
import matplotlib.pyplot as plt
import pygmo


#-----------------------------------------#
#-------------class Pareto_EC-------------#
#-----------------------------------------#
class Pareto_EC(Ensemble_EC):
    """Class for Pareto-based EC.

    Candidate minimazing the first EC (multiplied by its coefficient -1 or 1) is the most promising. Candidate minimizing the second EC (multiplied by its coefficient -1 or 1) is the second most promising. Remaining candidates are ordering according to their non-dominated rank (the lowest the rank the more promising is the candidate). Candidates with same non-dominated rank are ordered either according to their crowding distance `cd` or according to their hypervolume contribution `hvc`.

    :param coeffs: coefficients (1 or -1) to multiply the EC with (allow to convert a minimization problem into a maximization problem and reversely)
    :type coeffs: np.ndarray
    :param distinct_mode: criterion to distinguish solutions with same non domination rank. When equals `cd` solutions with higher crowded distance are considered as more promising. When equals to `hvc`, solutions with higher hypervolume contribution are considered as more promising.
    :type distinct_mode: either `cd` or `hvc`
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
        assert distinct_mode=='cd' or distinct_mode=='hvc'

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
        return self.__coeffs

    #-------------_get_distinct_mode-------------#
    def _get_distinct_mode(self):
        return self.__distinct_mode
        
    #-------------property-------------#
    coeffs=property(_get_coeffs, None, None)
    distinct_mode=property(_get_distinct_mode, None, None)


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

        if self.__distinct_mode=='cd':

            # ordering according to NDF and crowding distance
            idx = pygmo.sort_population_mo(criteria) 

            # first is the index of the best individual according to the first EC
            criteria = criteria.T
            tmp_idx = np.argsort(criteria[0])
            if idx[0]!=tmp_idx[0]:
                idx=np.insert(idx[idx!=tmp_idx[0]], 0, tmp_idx[0])

            # second is the index of the best individual according to the second EC (only if it is different than the best individual according to the first EC)
            tmp_idx = np.argsort(criteria[1])
            if not np.in1d(tmp_idx[0], idx[0:2]):
                idx=np.insert(idx[idx!=tmp_idx[0]], 1, tmp_idx[0])
            
        elif self.__distinct_mode=='hvc':

            # ordering according to NDF and hypervolume contribution
            (ndf, dom_list, dom_count, ndr) = pygmo.fast_non_dominated_sorting(criteria)
            idx = np.array([], dtype=int)
            for i in range(int(ndr.max()+1)):
                if np.where(ndr==i)[0].size>1:
                    hv = pygmo.hypervolume(criteria[np.where(ndr==i)[0]])
                    ref_point = pygmo.nadir(criteria[np.where(ndr==i)[0]])+1
                    idx = np.append(idx, np.where(ndr==i)[0][np.argsort(-hv.contributions(ref_point))])
                else:
                    idx = np.append(idx, np.where(ndr==i)[0])

            # first is the index of the best individual according to the first EC
            criteria = criteria.T
            tmp_idx = np.argsort(criteria[0])
            if idx[0]!=tmp_idx[0]:
                idx=np.insert(idx[idx!=tmp_idx[0]], 0, tmp_idx[0])

            # second is the index of the best individual according to the second EC (only if it is different than the best individual according to the first EC)
            tmp_idx = np.argsort(criteria[1])
            if not np.in1d(tmp_idx[0], idx[0:2]):
                idx=np.insert(idx[idx!=tmp_idx[0]], 1, tmp_idx[0])

        else:
            print("[Pareto_EC.py] error: dinstinction mode is wrong")
            assert False
            
        return idx
