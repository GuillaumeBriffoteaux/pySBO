import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pygmo

from Problems.Multi_Objective import Multi_Objective


#-----------------------------------#
#-------------class ZDT-------------#
#-----------------------------------#
class ZDT(Multi_Objective):
    """Class for bi-objective problems from the ZDT test suite.

    :param f_id: problem's identifier into the pygmo library
    :type f_id: int in {1,2,3,4,6}
    :param n_dvar: number of decision variables
    :type n_dvar: positive int, strictly greater than 1
    """
    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#

    #-------------__init__-------------#    
    def __init__(self, f_id, n_dvar):
        assert type(f_id)==int
        assert f_id>=1 and f_id<=6 and f_id!=5
        assert n_dvar>=2
        Multi_Objective.__init__(self, n_dvar, 2)
        self.__pb = pygmo.problem(pygmo.zdt(f_id, n_dvar))

    #-------------__del__-------------#
    def __del__(self):
        Multi_Objective.__del__(self)

    #-------------__str__-------------#
    def __str__(self):
        return self.__pb.get_name()+" "+str(self.n_dvar)+" decision variables "+str(self.n_obj)+" objectives"


    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#
    
    #-------------perform_real_evaluation-------------#
    def perform_real_evaluation(self, candidates):
        """Objective function.

        :param candidates: candidate decision vectors
        :type candidates: np.ndarray
        :return: objective values
        :rtype: np.ndarray
        """

        assert self.is_feasible(candidates)
        if candidates.ndim==1:
            candidates = np.array([candidates])

        obj_vals = np.zeros((candidates.shape[0], self.n_obj))
        for i,cand in enumerate(candidates):
            obj_vals[i] = self.__pb.fitness(cand)

        return obj_vals

    #-------------get_bounds-------------#
    def get_bounds(self):
        """Returns search space bounds.

        :returns: search space bounds
        :rtype: np.ndarray
        """

        res=np.ones((2,self.n_dvar))
        res[0,:]=self.__pb.get_bounds()[0]
        res[1,:]=self.__pb.get_bounds()[1]
        return res

    #-------------is_feasible-------------#
    def is_feasible(self, candidates):
        """Check feasibility of candidates.

        :param candidates: candidate decision vectors
        :type candidates: np.ndarray
        :returns: boolean indicating whether candidates are feasible
        :rtype: bool
        """

        res=False
        if Multi_Objective.is_feasible(self, candidates)==True:
            lower_bounds=self.get_bounds()[0,:]
            upper_bounds=self.get_bounds()[1,:]
            res=(lower_bounds<=candidates).all() and (candidates<=upper_bounds).all()
        return res
