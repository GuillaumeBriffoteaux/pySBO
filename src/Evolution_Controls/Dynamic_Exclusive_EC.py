import numpy as np

from Evolution_Controls.Ensemble_EC import Ensemble_EC


#----------------------------------------------------#
#-------------class Dynamic_Exclusive_EC-------------#
#----------------------------------------------------#
class Dynamic_Exclusive_EC(Ensemble_EC):
    """Class for dynamic exclusive EC.

    Only one EC is active at a time. The active EC changes during the search according to the search budget, the activation sections and the current search progress.

    :param search_budget: search budget (expressed either in number of generations, number of acquisition processes or time)
    :type search_budget: positive int, not zero
    :param sections: activation sections (determines the moment to change the active EC)
    :type sections: list
    :param idx_active: index of the current active EC in ECs_list
    :type idx_active: positive int
    """
    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#

    #-------------__init__-------------#
    def __init__(self, search_budget, sections, *ECs):
        """
        __init__ method's input

        :param ECs: evolution controls
        :type ECs: list(Evolution_Control)
        """

        Ensemble_EC.__init__(self, *ECs)
        assert type(search_budget)==int
        assert type(sections)==list
        assert len(sections)==len(ECs)
        assert sum(sections)==1

        self.__sections=np.array(sections)
        self.__idx_active=0
        self.__search_budget=search_budget

        
    #-------------__del__-------------#
    def __del__(self):
        Ensemble_EC.__del__(self)
        del self.__sections
        del self.__idx_active
        del self.__search_budget

        
    #-------------__str__-------------#
    def __str__(self):
        res = "Dynamic Exclusive Ensemble Evolution Control\n  ECs: {"
        for i,ec in enumerate(self.ECs_list):
            res+=" "+ec.__class__.__name__
        res+="}\n  index active EC: "+str(self.__idx_active)+"\n  search budget: "+str(self.__search_budget)+"\n  sections: "+str(self.__sections)
        return res

    
    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#

    #-------------get_sorted_indexes-------------#
    def get_sorted_indexes(self, pop):        
        Ensemble_EC.get_sorted_indexes(self, pop)

        return self.ECs_list[self.__idx_active].get_sorted_indexes(pop)

    
    #-------------update_active-------------#
    def update_active(self, search_progress):
        """Update the active EC.

        :param search_progress: current search progress (expressed either in number of generations, number of acquisition processes or time)
        :type search_progress: positive int
        """

        # search_progress has to be expressed in the same unit as search_budget
        for i in range(1,len(self.__sections)):
            if search_progress>=round(np.sum(self.__sections[0:i])*self.__search_budget) and search_progress<round(np.sum(self.__sections[0:i+1])*self.__search_budget):
                self.__idx_active=i
        if search_progress>=round(np.sum(self.__sections[0:self.__sections.size-1])*self.__search_budget):
            self.__idx_active=self.__sections.size-1
