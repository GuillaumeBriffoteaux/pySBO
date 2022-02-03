import numpy as np

from Evolution_Controls.Evolution_Control import Evolution_Control


#-------------------------------------------#
#-------------class Ensemble_EC-------------#
#-------------------------------------------#
class Ensemble_EC(Evolution_Control):
    """Abstract class for ensembles of EC.

    :param ECs_list: evolution controls
    :type ECs_list: list(Evolution_Control)
    """
    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#

    #-------------__init__-------------#
    def __init__(self, *ECs):
        Evolution_Control.__init__(self)
        assert len(ECs)>1
        self.__ECs_list = [None for i in range(0,len(ECs))]
        for i,ec in enumerate(ECs):
            assert isinstance(ec, Evolution_Control)
            self.__ECs_list[i] = ec
        
    #-------------__del__-------------#
    def __del__(self):
        Evolution_Control.__del__(self)
        del self.__ECs_list


    #---------------------------------------------#
    #-------------getters and setters-------------#
    #---------------------------------------------#

    #-------------_get_ECs_list-------------#
    def _get_ECs_list(self):
        return self.__ECs_list

    #-------------property-------------#
    ECs_list=property(_get_ECs_list, None, None)
