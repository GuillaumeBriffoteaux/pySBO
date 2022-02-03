import numpy as np

from Evolution_Controls.Ensemble_EC import Ensemble_EC


#----------------------------------------------------#
#-------------class Dynamic_Inclusive_EC-------------#
#----------------------------------------------------#
class Dynamic_Inclusive_EC(Ensemble_EC):
    """Class for dynamic inclusive EC.

    Designed as an ensemble of 2 ECs only.
    Two ECs are active at a time. The proportion of use of each EC changes during the search according to the search budget, and the current search progress.

    :param search_budget: search budget (expressed either in number of generations, number of acquisition processes or time)
    :type search_budget: positive int, not zero
    :param search_progress: search progress (expressed either in number of generations, number of acquisition processes or time)
    :type search_progress: positive int, not zero
    :param N_SIM: number of solution to select for simulation
    :type N_SIM: int, positive
    :param N_PRED: number of solution to select for prediction
    :type N_PRED: int, positive
    """
    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#

    #-------------__init__-------------#
    def __init__(self, search_budget, N_SIM, N_PRED, *ECs):
        """
        __init__ method's input

        :param ECs: evolution controls
        :type ECs: list(Evolution_Control)
        """

        Ensemble_EC.__init__(self, *ECs)
        assert len(ECs)==2
        assert type(search_budget)==int
        assert type(N_SIM)==int and type(N_PRED)==int
        assert N_SIM>=0 and N_PRED>=0

        self.__search_budget=search_budget
        self.__search_progress=0.0
        self.__N_SIM = N_SIM
        self.__N_PRED = N_PRED
    
    #-------------__del__-------------#
    def __del__(self):
        Ensemble_EC.__del__(self)
        del self.__search_budget
        del self.__search_progress
        del self.__N_SIM
        del self.__N_PRED

    #-------------__str__-------------#
    def __str__(self):
        res = "Dynamic Inclusive Ensemble Evolution Control\n  ECs: {"
        for i,ec in enumerate(self.ECs_list):
            res+=" "+ec.__class__.__name__
        res+="}\n  search budget: "+str(self.__search_budget)+"\n  N_SIM: "+str(self.__N_SIM)+"  N_PRED: "+str(self.__N_PRED)
        return res

    
    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#

    
    #-------------get_sorted_indexes-------------#
    def get_sorted_indexes(self, pop):        
        Ensemble_EC.get_sorted_indexes(self, pop)

        sorted_indexes_1 = self.ECs_list[0].get_sorted_indexes(pop)
        sorted_indexes_2 = self.ECs_list[1].get_sorted_indexes(pop)

        if self.__search_progress < 0.2:
            returned_indexes = sorted_indexes_1
        elif self.__search_progress < 0.4:
            p=0.75
            # 0.75*self.__N_SIM selected by ECs_list[0]
            returned_indexes = sorted_indexes_1[0:int(p*self.__N_SIM)]
            sorted_indexes_1 = sorted_indexes_1[int(p*self.__N_SIM):]
            sorted_indexes_2 = sorted_indexes_2[~np.isin(sorted_indexes_2, returned_indexes)]
            # 0.25*self.__N_SIM selected by ECs_list[1]
            returned_indexes = np.append(returned_indexes, sorted_indexes_2[0:int((1-p)*self.__N_SIM)])
            sorted_indexes_1 = sorted_indexes_1[~np.isin(sorted_indexes_1, sorted_indexes_2[0:int((1-p)*self.__N_SIM)])]
            sorted_indexes_2 = sorted_indexes_2[int((1-p)*self.__N_SIM):]
            # 0.75*self.__N_PRED selected by ECs_list[0]
            returned_indexes = np.append(returned_indexes, sorted_indexes_1[0:int(p*self.__N_PRED)])
            sorted_indexes_1 = sorted_indexes_1[int(p*self.__N_PRED):]
            sorted_indexes_2 = sorted_indexes_2[~np.isin(sorted_indexes_2, returned_indexes)]
            # 0.25*self.__N_PRED selected by ECs_list[1]
            returned_indexes = np.append(returned_indexes, sorted_indexes_2[0:int((1-p)*self.__N_PRED)])
            sorted_indexes_1 = sorted_indexes_1[~np.isin(sorted_indexes_1, sorted_indexes_2[0:int((1-p)*self.__N_PRED)])]
            sorted_indexes_2 = sorted_indexes_2[int((1-p)*self.__N_PRED):]
            # N_DISC remaining
            returned_indexes = np.append(returned_indexes, sorted_indexes_1)
        elif self.__search_progress < 0.6:
            p=0.5
            # 0.5*self.__N_SIM selected by ECs_list[0]
            returned_indexes = sorted_indexes_1[0:int(p*self.__N_SIM)]
            sorted_indexes_1 = sorted_indexes_1[int(p*self.__N_SIM):]
            sorted_indexes_2 = sorted_indexes_2[~np.isin(sorted_indexes_2, returned_indexes)]
            # 0.5*self.__N_SIM selected by ECs_list[1]
            returned_indexes = np.append(returned_indexes, sorted_indexes_2[0:int((1-p)*self.__N_SIM)])
            sorted_indexes_1 = sorted_indexes_1[~np.isin(sorted_indexes_1, sorted_indexes_2[0:int((1-p)*self.__N_SIM)])]
            sorted_indexes_2 = sorted_indexes_2[int((1-p)*self.__N_SIM):]
            # 0.5*self.__N_PRED selected by ECs_list[0]
            returned_indexes = np.append(returned_indexes, sorted_indexes_1[0:int(p*self.__N_PRED)])
            sorted_indexes_1 = sorted_indexes_1[int(p*self.__N_PRED):]
            sorted_indexes_2 = sorted_indexes_2[~np.isin(sorted_indexes_2, returned_indexes)]
            # 0.5*self.__N_PRED selected by ECs_list[1]
            returned_indexes = np.append(returned_indexes, sorted_indexes_2[0:int((1-p)*self.__N_PRED)])
            sorted_indexes_1 = sorted_indexes_1[~np.isin(sorted_indexes_1, sorted_indexes_2[0:int((1-p)*self.__N_PRED)])]
            sorted_indexes_2 = sorted_indexes_2[int((1-p)*self.__N_PRED):]
            # N_DISC remaining
            returned_indexes = np.append(returned_indexes, sorted_indexes_1)
        elif self.__search_progress < 0.8:
            p=0.75
            # 0.75*self.__N_SIM selected by ECs_list[1]
            returned_indexes = sorted_indexes_2[0:int(p*self.__N_SIM)]
            sorted_indexes_2 = sorted_indexes_2[int(p*self.__N_SIM):]
            sorted_indexes_1 = sorted_indexes_1[~np.isin(sorted_indexes_1, returned_indexes)]
            # 0.25*self.__N_SIM selected by ECs_list[0]
            returned_indexes = np.append(returned_indexes, sorted_indexes_1[0:int((1-p)*self.__N_SIM)])
            sorted_indexes_2 = sorted_indexes_2[~np.isin(sorted_indexes_2, sorted_indexes_1[0:int((1-p)*self.__N_SIM)])]
            sorted_indexes_1 = sorted_indexes_1[int((1-p)*self.__N_SIM):]
            # 0.75*self.__N_PRED selected by ECs_list[1]
            returned_indexes = np.append(returned_indexes, sorted_indexes_2[0:int(p*self.__N_PRED)])
            sorted_indexes_2 = sorted_indexes_2[int(p*self.__N_PRED):]
            sorted_indexes_1 = sorted_indexes_1[~np.isin(sorted_indexes_1, returned_indexes)]
            # 0.25*self.__N_PRED selected by ECs_list[0]
            returned_indexes = np.append(returned_indexes, sorted_indexes_1[0:int((1-p)*self.__N_PRED)])
            sorted_indexes_2 = sorted_indexes_2[~np.isin(sorted_indexes_2, sorted_indexes_1[0:int((1-p)*self.__N_PRED)])]
            sorted_indexes_1 = sorted_indexes_1[int((1-p)*self.__N_PRED):]
            # N_DISC remaining
            returned_indexes = np.append(returned_indexes, sorted_indexes_2)
        elif self.__search_progress <= 1.0:
            returned_indexes = sorted_indexes_2
        else:
            print("[Dynamic_Inclusive_EC.py] error: unexpected value for self.__search_progress")
            assert False

        return returned_indexes

    
    #-------------update_active-------------#
    def update_active(self, search_progress):
        """Update the proportion of use of each EC.

        :param search_progress: current search progress (expressed either in number of generations, number of acquisition processes or time)
        :type search_progress: positive int
        """
        
        self.__search_progress = search_progress/self.__search_budget
