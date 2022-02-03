import csv
import pickle
import numpy as np
import itertools
import pygmo
import matplotlib.pyplot as plt
import scipy

from Evolution.Population import Population
from Problems.Problem import Problem
from Global_Var import *


#----------------------------------------------------#
#-------------class Reference_Vector_Set-------------#
#----------------------------------------------------#
class Reference_Vector_Set:
    """Class for the set of reference vectors of RVEA.

    :param pb: problem
    :type pb: Problem
    :param rv: set of reference vectors
    :type rv: np.ndarray
    """

    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#

    #-------------__init__-------------#
    def __init__(self, H, pb):
        """
        __init__ method's input

        :param H: simplex lattice parameter
        :type H: int
        :param pb: problem
        :type pb: Problem
        """

        assert isinstance(pb, Problem)

        self.__pb = pb
        self.rv = np.array([])
        # simplex lattice
        values = np.arange(H + 1)/(H)
        for items in itertools.product(values, repeat=pb.n_obj):
            if sum(items) == 1.0:
                self.rv = np.append(self.rv, list(items))
        self.rv = np.reshape(self.rv, (-1, pb.n_obj))

        # normalizing
        for i in range(self.rv.shape[0]):
            self.rv[i] = self.rv[i]/scipy.linalg.norm(self.rv[i], ord=2)

    #-------------__del__-------------#
    def __del__(self):
        del self.__pb
        del self.rv

    #-------------__str__-------------#
    def __str__(self):
        return "Set of reference vectors\n  Problem:\n  "+str(self.__pb)+"\n  Reference vectors:\n  "+str(self.rv)

    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#
    
    #-------------reference_vector_guided_replacement-------------#
    def reference_vector_guided_replacement(self, pop, search_progress, budget):
        """Performs replacement based on the set of reference vectors.

        :param pop: population to perform replacement on
        :type pop: Population
        :param search_progress: current generation index (or elapsed time)
        :type search_progress: int or float
        :param budget: number of generations (or time) allocated for the search
        :type budget: int or float
        :returns: the new population
        :rtype: Population
        """
        
        assert isinstance(pop, Population)
        if search_progress>budget:
            print("[Reference_Vector_Set.py] search_progress>budget")
            search_progress=budget

        # Objective value translation
        z_min = np.amin(pop.obj_vals, axis=0)
        trans_obj_vec = np.copy(pop.obj_vals)
        trans_obj_vec = trans_obj_vec - z_min

        # Angle from each reference vector to the remaining reference vectors
        theta_rv = np.array([])
        for (rv1, rv2) in itertools.permutations(self.rv, 2):
            theta_rv = np.append(theta_rv, np.arccos(np.dot(rv1,rv2)))
        theta_rv = np.reshape(theta_rv, (self.rv.shape[0], self.rv.shape[0]-1))

        # Minimum angle from each reference vector to the remaining reference vectors
        min_theta_rv = np.amin(theta_rv, axis=1)

        # Compute the cosine angles
        cos_theta = np.array([]) # first index: translated objective vector | second index: reference vector
        for (tc, rv) in itertools.product(trans_obj_vec, self.rv):
            cos_theta = np.append(cos_theta, np.dot(tc, rv)/scipy.linalg.norm(tc))
        cos_theta = np.reshape(cos_theta, (trans_obj_vec.shape[0], self.rv.shape[0]))

        # Form the sub-populations and compute the APD
        trans_obj_vec_subpop = -1*np.ones((trans_obj_vec.shape[0],), dtype=int) # contains the index of the sub-population for each objective vector
        APD = np.array([])
        for i in range(trans_obj_vec.shape[0]):
            trans_obj_vec_subpop[i] = np.argmax(cos_theta[i,:])
            theta = np.arccos(cos_theta[i, trans_obj_vec_subpop[i]])
            APD = np.append( APD, ( 1.0 + self.__pb.n_obj * pow(float(search_progress)/float(budget),2) * (theta/min_theta_rv[trans_obj_vec_subpop[i]]) ) * scipy.linalg.norm(trans_obj_vec[i]) )

        # Elitism replacement
        new_pop = Population(self.__pb)
        for i in range(self.rv.shape[0]): # i is the sub-population index
            indexes_obj_vec = np.where(trans_obj_vec_subpop==i)[0] # indexes of objective vectors pertaining to sub-population i
            if indexes_obj_vec.size>0:
                k = indexes_obj_vec[np.argmin( APD[indexes_obj_vec] )]
                new_pop.dvec = np.vstack( (new_pop.dvec, pop.dvec[k]) )
                new_pop.obj_vals =  np.vstack( (new_pop.obj_vals, pop.obj_vals[k]) )
                new_pop.fitness_modes =  np.vstack( (new_pop.fitness_modes, pop.fitness_modes[k]) )
                
        return new_pop


    #-------------reference_vector_update-------------#
    # self should be the initial set of reference vectors
    def reference_vector_update(self, pop):
        """Updates the set of reference vectors.

        :param pop: current population
        :type pop: Population
        :returns: the updated set of reference vectors
        :rtype: Reference_Vector_Set
        """

        assert isinstance(pop, Population)
                                
        z_min = np.amin(pop.obj_vals, axis=0)
        z_max = np.amax(pop.obj_vals, axis=0)
        new_rv = self.rv * ( z_max - z_min )
        # normalizing
        for i in range(new_rv.shape[0]):
            new_rv[i] = new_rv[i]/scipy.linalg.norm(new_rv[i], ord=2)

        return new_rv


    #-------------reference_vector_regeneration-------------#
    def reference_vector_regeneration(self, pop):
        """Regenerates the set of reference vectors (for RVEA*).

        :param pop: current population
        :type pop: Population
        """
        
        (ndf, dom_list, dom_count, ndr) = pygmo.fast_non_dominated_sorting(pop.obj_vals)
        P = Population(self.__pb)
        P.dvec = np.copy(pop.dvec[ndf[0]])
        P.obj_vals = np.copy(pop.obj_vals[ndf[0]])
        P.fitness_modes = np.copy(pop.fitness_modes[ndf[0]])

        # Objective value translation
        z_min = np.amin(P.obj_vals, axis=0)
        trans_obj_vec = np.copy(P.obj_vals)
        trans_obj_vec = trans_obj_vec - z_min
            
        # Compute the cosine angles and the angle penalized distance
        cos_theta = np.array([]) # first index: translated objective vector | second index: reference vector
        for (tc, rv) in itertools.product(trans_obj_vec, self.rv):
            cos_theta = np.append(cos_theta, np.dot(tc, rv)/scipy.linalg.norm(tc))
        cos_theta = np.reshape(cos_theta, (trans_obj_vec.shape[0], self.rv.shape[0]))

        # Form the sub-populations
        trans_obj_vec_subpop = -1*np.ones((trans_obj_vec.shape[0],), dtype=int) # contains the index of the sub-population for each objective vector
        for i in range(trans_obj_vec.shape[0]):
            trans_obj_vec_subpop[i] = np.argmax(cos_theta[i,:])

        # Regeneration
        z_max = np.amax(P.obj_vals, axis=0)
        for i in range(self.rv.shape[0]): # i is the sub-population index
            indexes_obj_vec = np.where(trans_obj_vec_subpop==i)[0] # indexes of objective vectors pertaining to sub-population i
            if indexes_obj_vec.size==0:
                self.rv[i] = np.random.uniform(size=z_max.shape)
                self.rv[i] = np.multiply(self.rv[i], z_max)
                self.rv[i] = self.rv[i]/scipy.linalg.norm(self.rv[i])
                
    
    #-------------save_to_csv_file-------------#
    def save_to_csv_file(self, f_rv_archive):
        """Prints the set of reference vectors to a CSV file.

        The CSV file is organized as follows:
        First row: number of reference vectors and number of objectives
        Remaining rows: the reference vectors

        :param f_rv_archive: filename of the CSV file.
        :type f_rv_archive: str
        """

        assert type(f_rv_archive)==str

        with open(f_rv_archive, 'w') as my_file:
            my_file.write(str(self.rv.shape[0])+" "+str(self.rv.shape[1])+"\n")
            for rv in self.rv:
                my_file.write(" ".join(map(str, rv))+"\n")

                
    #-------------plot-------------#
    def plot(self):
        """Plot the 3D reference vectors."""

        if self.rv.shape[1]==3:
            fig = plt.figure()
            ax = plt.axes(projection ='3d')
            ax.scatter(self.rv[:,0], self.rv[:,1], self.rv[:,2])
            plt.show()
