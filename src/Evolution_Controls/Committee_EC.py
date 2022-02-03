import numpy as np

from Evolution_Controls.Ensemble_EC import Ensemble_EC


#--------------------------------------------#
#-------------class Committee_EC-------------#
#--------------------------------------------#
class Committee_EC(Ensemble_EC):
    """Class for committee of evolution controls.

    All the ECs vote to determine the promise of the candidates.

    :param n_sim: number of simulated candidates from the batch
    :type n_sim: positive int, not zero
    """
    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#

    #-------------__init__-------------#
    def __init__(self, n_sim, *ECs):
        """__init__ method's input

        :param ECs: evolution controls
        :type ECs: list(Evolution_Control)
        """

        Ensemble_EC.__init__(self, *ECs)
        assert type(n_sim)==int
        assert n_sim>=0
        self.__n_sim=n_sim
        
    #-------------__del__-------------#
    def __del__(self):
        Ensemble_EC.__del__(self)
        del self.__n_sim

    #-------------__str__-------------#
    def __str__(self):
        res = "Committee Ensemble Evolution Control\n  n_sim="+str(self.__n_sim)+"\n  ECs: {"
        for i,ec in enumerate(self.ECs_list):
            res+=" "+ec.__class__.__name__
        res+="}"
        return res

    
    #---------------------------------------------#
    #-------------getters and setters-------------#
    #---------------------------------------------#

    
    #-------------_get_n_sim-------------#
    def _get_n_sim(self):
        return self.__n_sim

    #-------------property-------------#
    n_sim=property(_get_n_sim, None, None)


    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#
    
    #-------------get_sorted_indexes-------------#
    def get_sorted_indexes(self, pop):
        Ensemble_EC.get_sorted_indexes(self, pop)

        all_idx = np.arange(pop.dvec.shape[0], dtype=int)
        votes = np.zeros(all_idx.shape, dtype=int)
        
        for ec in self.ECs_list:
            votes[np.where(np.in1d(all_idx, ec.get_sorted_indexes(pop)[0:self.__n_sim]))[0]] +=1
        idx_sort = np.argsort(-votes)

        return idx_sort
    
    #-------------update_active-------------#
    def update_active(self, search_progress):
        print("[Committee_EC.py] update_active() not implemented")
        assert False
