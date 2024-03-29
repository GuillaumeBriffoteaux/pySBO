import numpy as np
from pyDOE import lhs

from Problems.Box_Constrained import Box_Constrained


#-----------------------------------#
#-------------class DoE-------------#
#-----------------------------------#
class DoE:
    """Class for Design of Experiments.

    :param pb: related optimization problem
    :type pb: Box_Constrained
    """
    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#

    #-------------__init__-------------#
    def __init__(self, pb):
        assert isinstance(pb, Box_Constrained)
        self.__pb=pb

    #-------------__del__-------------#
    def __del__(self):
        del self.__pb

    #-------------__str__-------------#
    def __str__(self):
        return "DoE for problem {"+self.__pb.__str__()+"}"


    #---------------------------------------------#
    #-------------getters and setters-------------#
    #---------------------------------------------#

    #-------------_get_pb-------------#
    def _get_pb(self):
        return self.__pb

    #-------------property-------------#
    pb=property(_get_pb)


    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#
    
    #-------------random_uniform_sampling-------------#
    def random_uniform_sampling(self, nb_samples):
        """Returns samples generated by random uniform sampling over the search space.

        :param nb_samples: number of samples to generate
        :type nb_samples: positive int, not zero
        :returns: samples
        :rtype: np.ndarray
        """
        
        assert type(nb_samples) is int
        assert nb_samples>0

        return np.random.uniform(self.__pb.get_bounds()[0], self.__pb.get_bounds()[1], (nb_samples, self.__pb.n_dvar))

    #-------------latin_hypercube_sampling-------------#
    def latin_hypercube_sampling(self, nb_samples):
        """Returns samples generated by latin hypercube sampling over the search space.

        :param nb_samples: number of samples to generate
        :type nb_samples: positive int, not zero
        :returns: samples
        :rtype: np.ndarray
        """

        assert type(nb_samples) is int
        assert nb_samples>0

        return (self.__pb.get_bounds()[1]-self.__pb.get_bounds()[0])*lhs(self.__pb.n_dvar, nb_samples, "maximin") + self.__pb.get_bounds()[0]
