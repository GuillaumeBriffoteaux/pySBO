import numpy as np

from Evolution_Controls.Evolution_Control import Evolution_Control


#-----------------------------------------#
#-------------class Random_EC-------------#
#-----------------------------------------#
class Random_EC(Evolution_Control):
    """Class for random evolution control."""
    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#

    #-------------__init__-------------#
    def __init__(self):
        Evolution_Control.__init__(self)
    
    #-------------__del__-------------#
    def __del__(self):
        Evolution_Control.__del__(self)

    #-------------__str__-------------#
    def __str__(self):
        return "Random Evolution Control"


    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#
    
    #-------------get_sorted_indexes-------------#
    def get_sorted_indexes(self, pop):
        Evolution_Control.get_sorted_indexes(self, pop)

        idx = np.arange(0, pop.dvec.shape[0], dtype=int)
        np.random.shuffle(idx)

        return idx
