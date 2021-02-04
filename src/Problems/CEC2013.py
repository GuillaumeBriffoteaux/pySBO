import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pygmo

from Problems.Mono_Objective import Mono_Objective


#---------------------------------------#
#-------------class CEC2013-------------#
#---------------------------------------#
class CEC2013(Mono_Objective):
    """Class for mono-objective problems from the CEC2013.

    :param f_id: problem's identifier into the pygmo library
    :type f_id: int in {1,...,28}
    :param n_dvar: number of decision variable
    :type n_dvar: positive int, not zero
    """
    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#

    #-------------__init__-------------#    
    def __init__(self, f_id, n_dvar):
        assert type(f_id)==int
        assert f_id>=1 and f_id<=28
        assert n_dvar==2 or n_dvar==5 or n_dvar==20 or n_dvar==30 or n_dvar==40 or n_dvar==50 or n_dvar==60 or n_dvar==70 or n_dvar==80 or n_dvar==90 or n_dvar==100
        Mono_Objective.__init__(self, n_dvar, 1)
        self.__pb = pygmo.problem(pygmo.cec2013(f_id, n_dvar))

    #-------------__del__-------------#
    def __del__(self):
        Mono_Objective.__del__(self)

    #-------------__str__-------------#
    def __str__(self):
        return self.__pb.get_name()+" "+str(self.n_dvar)+" decision variables "+str(self.n_obj)+" objective"

    
    #---------------------------------------------#
    #-------------getters and setters-------------#
    #---------------------------------------------#

    #-------------_get_pb-------------#
    def _get_pb(self):
        print("[CEC2013.py] Impossible to get the Pygmo problem")
        return None

    #-------------_set_pb-------------#
    def _set_pb(self,new_pb):
        print("[CEC2013.py] Impossible to modify the Pygmo problem")

    #-------------_del_pb-------------#
    def _del_pb(self):
        print("[CEC2013.py] Impossible to delete the the Pygmo problem")

    #-------------property-------------#
    pb=property(_get_pb, _set_pb, _del_pb)



    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#
    
    #-------------perform_real_evaluation-------------#
    def perform_real_evaluation(self, candidates):
        """Fitness function.

        :param candidates: candidate decision vectors
        :type candidates: np.ndarray
        :return: costs (i.e. objective values)
        :rtype: np.ndarray
        """

        assert self.is_feasible(candidates)
        if candidates.ndim==1:
            candidates = np.array([candidates])

        costs = np.zeros((candidates.shape[0],))
        for i,cand in enumerate(candidates):
            costs[i] = self.__pb.fitness(cand)

        return costs

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
        if Mono_Objective.is_feasible(self, candidates)==True:
            lower_bounds=self.get_bounds()[0,:]
            upper_bounds=self.get_bounds()[1,:]
            res=(lower_bounds<=candidates).all() and (candidates<=upper_bounds).all()
        return res

    #-------------plot-------------#
    def plot(self):
        """Plot the 2D CEC2013 considered problem's fitness function."""
        
        if self.n_dvar==2:
            fig = plt.figure()

            lower_bounds = self.get_bounds()[0,:]
            upper_bounds = self.get_bounds()[1,:]

            x = np.linspace(lower_bounds[0], upper_bounds[0], 100)
            y = np.linspace(lower_bounds[1], upper_bounds[1], 100)
            z = self.perform_real_evaluation( np.array(np.meshgrid(x, y)).T.reshape(-1,2) ).reshape(x.size, y.size)
            x, y = np.meshgrid(x, y)
            
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(x, y, z, rstride=1, cstride=1, cmap=cm.jet, antialiased=False)
            plt.title(self.__pb.get_name()+" 2D")
            plt.show()

        else:
            print("[CEC2013.py] Impossible to plot CEC2013 with n_dvar="+str(self.n_dvar))
