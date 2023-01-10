import csv
import pickle
import numpy as np
import itertools
import pygmo

from Problems.Problem import Problem
from Global_Var import *


#------------------------------------------#
#-------------class Population-------------#
#------------------------------------------#
class Population:
    """Class for the population of an evolutionary algorithm.

    :param pb: problem
    :type pb: Problem
    :param dvec: decision vectors of the individuals
    :type dvec: np.ndarray
    :param obj_vals: objective value associated with each individual
    :type obj_vals: np.ndarray
    :param fitness_modes: evaluation mode associated with each individual: True for real evaluation and False for prediction (surrogate evaluation)
    :type fitness_modes: np.ndarray
    """
    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#

    #-------------__init__-------------#
    def __init__(self, pb):
        """
        __init__ method's input
        
        :param pb: problem
        :type pb: Problem
        """
        assert isinstance(pb, Problem)

        self.pb=pb
        self.dvec=np.empty((0,self.pb.n_dvar))
        # mono-objective
        if self.pb.n_obj==1:
            self.obj_vals=np.empty((0,))
            self.fitness_modes=np.empty((0,), dtype=bool)
        # multi-objective
        else:
            self.obj_vals=np.empty((0,self.pb.n_obj))
            self.fitness_modes=np.empty((0,self.pb.n_obj))

    #-------------__del__-------------#
    def __del__(self):
        del self.pb
        del self.dvec
        del self.obj_vals
        del self.fitness_modes

    #-------------__str__-------------#
    def __str__(self):
        return "Population\n  Problem:\n  "+str(self.pb)+"\n  Decision vectors:\n  "+str(self.dvec)+"\n  Objective values:\n  "+str(self.obj_vals)+"\n  Fitness modes:\n  "+str(self.fitness_modes)


    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#
    
    #-------------print_shapes-------------#
    def print_shapes(self):
        """Prints the shapes of the arrays `dvec`, `obj_vals` and `fitness_modes` forming the population."""
        
        print(self.dvec.shape)
        print(self.obj_vals.shape)
        print(self.fitness_modes.shape)

        
    #-------------check_integrity-------------#
    def check_integrity(self):
        """Checks arrays' shapes are consistent.

        :returns: True for arrays' consistency and False otherwise
        :rtype: bool
        """

        return (self.dvec.shape[1]==self.pb.n_dvar) and ((self.obj_vals.size==0 and self.fitness_modes.size==0) or (self.obj_vals.shape[0]==self.dvec.shape[0] and self.fitness_modes.shape[0]==self.dvec.shape[0] and (self.fitness_modes.ndim==1 or self.fitness_modes.shape[1]==self.pb.n_obj) and (self.obj_vals.ndim==1 or self.obj_vals.shape[1]==self.pb.n_obj)))

    
    #-------------append-------------#
    def append(self, pop):
        """Appends individuals to the current population.
        
        :param pop: individuals to be appended
        :type pop: Population
        """

        assert isinstance(pop, Population)

        self.dvec = np.vstack( (self.dvec, pop.dvec) )
        # mono-objective
        if self.obj_vals.ndim==1:
            self.obj_vals = np.concatenate( (self.obj_vals, pop.obj_vals) )
            self.fitness_modes = np.concatenate( (self.fitness_modes, pop.fitness_modes) )
        # multi-objective
        else:
            self.obj_vals = np.vstack( (self.obj_vals, pop.obj_vals) )
            self.fitness_modes = np.vstack( (self.fitness_modes, pop.fitness_modes) )

            
    #-------------sort-------------#
    def sort(self):
        """Sorts the population according to ascending individuals' objective value (single-objective) or non-dominated and crowded distance sorting (multi-objective)."""

        # mono-objective
        if self.obj_vals.ndim==1:
            idx = np.argsort(self.obj_vals)
        # multi-objective
        else:
            idx = pygmo.sort_population_mo(self.obj_vals)

        self.dvec = self.dvec[idx]
        self.fitness_modes = self.fitness_modes[idx]
        self.obj_vals = self.obj_vals[idx]

        
    #-------------split_in_batches-------------#
    def split_in_batches(self, n_batch):
        """Splits the population in batches.

        :param n_batch: number of batches
        :type n_batch: positive int, not zero
        :returns: list of batches
        :rtype: list(Population)
        """

        assert self.obj_vals.size==0 and self.fitness_modes.size==0

        batches = [Population(self.pb) for i in range(n_batch)]
        batches_dvec = np.split(self.dvec, n_batch)

        for (batch, batch_dvec) in zip(batches, batches_dvec):
            batch.dvec = batch_dvec

        return batches

    
    #-------------update_best_sim-------------#
    def update_best_sim(self, f_best_profile, f_hypervolume=None):
        """Updates the best individual (single-objective) or the best non-dominated front (multi-objective) and logs.

        For mono-objective:
        The best evaluated decision vector (minimisation assumed) is saved in `Global_Var.dvec_min` and its associated objective value is saved in `Global_Var.obj_val_min`.
        The best evaluated decision vector is printed to a file along with its associated objective value.

        For multi-objective:
        If the hypervolume has improved, the evaluated decision vectors composing the best non-dominated front are printed to a file along with their respective objective value.
        The best hyper-volume is always printed to a file.

        :param f_best_profile: filename for logging
        :type f_best_profile: str
        :param f_hypervolume: filename for logging hypervolume
        :type f_hypervolume: str
        """

        # mono-objective
        if self.obj_vals.ndim==1: 
            assert self.dvec.shape[0]==self.obj_vals.size
        
            # sorting
            tmp_dvec = self.dvec[np.argsort(self.obj_vals)]
            tmp_fitness_modes = self.fitness_modes[np.argsort(self.obj_vals)]
            tmp_obj_vals = self.obj_vals[np.argsort(self.obj_vals)]
        
            if np.where(tmp_fitness_modes==True)[0].size>0:
                
                best_idx = np.where(tmp_fitness_modes==True)[0][0]

                if tmp_obj_vals[best_idx]<Global_Var.obj_val_min:
                    Global_Var.obj_val_min = tmp_obj_vals[best_idx]
                    Global_Var.dvec_min = tmp_dvec[best_idx]

                with open(f_best_profile, 'a') as my_file:
                    my_file.write(" ".join(map(str, Global_Var.dvec_min))+" "+str(Global_Var.obj_val_min)+"\n")
        # multi-objective
        else:

            # considering only simulated decision vectors
            idx = np.where(self.fitness_modes==True)[0]
            idx = np.unique(idx)
            if idx.size>1:
                                
                # bi-objective
                if self.obj_vals.shape[1]==2:
                    idx_idx = pygmo.non_dominated_front_2d(self.obj_vals[idx,:])
                # three ojectives or more
                else:
                    (ndf, dom_list, dom_count, ndr) = pygmo.fast_non_dominated_sorting(self.obj_vals[idx,:])
                    idx_idx = ndf[0]

                # update best hypervolume
                hv = pygmo.hypervolume(self.obj_vals[idx[idx_idx],:])
                hv_value = hv.compute(Global_Var.ref_point)
                if hv_value>Global_Var.best_hv:
                    Global_Var.best_hv = hv_value

                    # write to logging file
                    with open(f_best_profile, 'a') as my_file:
                        for (dvec, obj_val, fitness_mode) in itertools.zip_longest(self.dvec[idx[idx_idx],:], self.obj_vals[idx[idx_idx],:], self.fitness_modes[idx[idx_idx]], fillvalue=''):
                            my_file.write(" ".join(map(str, dvec))+" "+" ".join(map(str, obj_val))+(" "+str(int(fitness_mode)) if type(fitness_mode)==np.bool_ else " "+" ".join(map(str, map(lambda x: 1 if x else 0, fitness_mode))))+"\n")
                        my_file.write(" ".join(map(str, Global_Var.ref_point))+"\n"+str(Global_Var.best_hv)+"\n\n")
                    
            with open(f_hypervolume, 'a') as my_file:
                my_file.write(str(Global_Var.best_hv)+"\n")



    #-------------save_to_csv_file-------------#
    def save_to_csv_file(self, f_pop_archive):
        """Prints the population to a CSV file.

        The CSV file is organized as follows:
        First row: number of decision variables, number of objectives, number of fitness modes
        Second row: lower bounds of the decision variables
        Thrid row: upper bounds of the decision variables
        Remaining rows (one per individual): decision variables, objective values, fitness mode

        :param f_pop_archive: filename of the CSV file.
        :type f_pop_archive: str
        """

        assert type(f_pop_archive)==str
        assert self.check_integrity()
        
        with open(f_pop_archive, 'w') as my_file:
            # writing number of decision variables, number of objectives and number of fitness modes
            my_file.write(str(self.dvec.shape[1])+" "+str(self.obj_vals.shape[1] if len(self.obj_vals.shape)>1 else 1 if self.obj_vals.shape[0]>0 else 0)+" "+str(self.fitness_modes.shape[1] if len(self.fitness_modes.shape)>1 else 1 if self.fitness_modes.shape[0]>0 else 0)+"\n")
            # writing bounds
            my_file.write(" ".join(map(str, self.pb.get_bounds()[0]))+"\n")
            my_file.write(" ".join(map(str, self.pb.get_bounds()[1]))+"\n")
            # writing each individuals
            if self.obj_vals.ndim==1: # mono-objective
                for (dvec, obj_val, fitness_mode) in itertools.zip_longest(self.dvec, self.obj_vals, self.fitness_modes, fillvalue=''):
                    my_file.write(" ".join(map(str, dvec))+" "+str(obj_val)+(" "+str(int(fitness_mode)) if type(fitness_mode)==np.bool_ else "")+"\n")
            else: # multi-objective
                for (dvec, obj_val, fitness_mode) in itertools.zip_longest(self.dvec, self.obj_vals, self.fitness_modes, fillvalue=''):
                    my_file.write(" ".join(map(str, dvec))+" "+" ".join(map(str, obj_val))+(" "+str(int(fitness_mode)) if type(fitness_mode)==np.bool_ else " "+" ".join(map(str, map(lambda x: 1 if x else 0, fitness_mode))))+"\n")

                
    #-------------load_from_csv_file-------------#
    # to load from the initial population archive
    def load_from_csv_file(self, f_pop_archive):
        """Loads the population from a CSV file.

        The CSV file has to be organized as follows:
        First row: number of decision variables, number of objectives, number of fitness modes
        Second row: lower bounds of the decision variables
        Third row: upper bounds of the decision variables
        Remaining rows (one per individual): decision variables, objective values, fitness mode
        
        :param f_pop_archive: filename of the CSV file
        :type f_pop_archive: str
        """

        assert type(f_pop_archive)==str

        with open(f_pop_archive, 'r') as my_file:
            # Counting the number of lines.
            reader = csv.reader(my_file, delimiter=' ')
            n_samples = sum(1 for line in reader) - 3
            my_file.seek(0)
        
            # First line: number of decision variables, number of objectives and number of fitness modes
            line = next(reader)
            n_dvar = int(line[0])
            n_obj = int(line[1])
            n_fm = int(line[2])

            # Second line: lower bounds
            lower_bounds = np.zeros((n_dvar,))
            lower_bounds[0:n_dvar] = np.asarray(next(reader))
            assert lower_bounds.all()==self.pb.get_bounds()[0].all()

            # Third line: upper bounds
            upper_bounds = np.zeros((n_dvar,))
            upper_bounds[0:n_dvar] = np.asarray(next(reader))
            assert upper_bounds.all()==self.pb.get_bounds()[1].all()

            # Following lines contain (dvec, obj_val)
            self.dvec = np.zeros((n_samples, n_dvar))
            self.obj_vals = np.zeros((n_samples,n_obj))
            self.fitness_modes = np.ones((n_samples, n_fm), dtype=bool)
            for i, line in enumerate(reader):
                self.dvec[i] = np.asarray(line[0:n_dvar])
                self.obj_vals[i,0:n_obj] = np.asarray(line[n_dvar:n_dvar+n_obj])
                self.fitness_modes[i,0:n_fm] = np.asarray(line[n_dvar+n_obj:n_dvar+n_obj+n_fm], dtype=int)
            if self.obj_vals.shape[1]<2:
                self.obj_vals = np.ndarray.flatten(self.obj_vals)
            if self.fitness_modes.shape[1]<2:
                self.fitness_modes = np.ndarray.flatten(self.fitness_modes)

        assert self.check_integrity()


    #-------------save_sim_archive-------------#
    def save_sim_archive(self, f_sim_archive):
        """Prints the real-evaluated individuals to a CSV file.

        The CSV file is organized as follows:
        One per individual: decision variables, objective values, fitness mode.

        :param f_sim_archive: filename of the CSV file.
        :type f_sim_archive: str
        """

        assert type(f_sim_archive)==str
        assert self.dvec.shape[0]==self.obj_vals.shape[0]

        idx_sim = np.where(self.fitness_modes==True)[0]
        idx_sim = np.unique(idx_sim) # for the multi-objective case
        with open(f_sim_archive, 'a') as my_file:
            # mono-objective
            if self.obj_vals.ndim==1: 
                for (dvec, obj_val) in zip(self.dvec[idx_sim], self.obj_vals[idx_sim]):
                    my_file.write(" ".join(map(str, dvec))+" "+str(obj_val)+"\n")
            # multi-objective
            else:
                for (dvec, obj_val) in zip(self.dvec[idx_sim], self.obj_vals[idx_sim]):
                    my_file.write(" ".join(map(str, dvec))+" "+" ".join(map(str, obj_val))+"\n")


    #-------------save_to_pickle_file-------------#
    def save_to_pickle_file(self, f_pop_archive):
        """Saves the population to a pickle file.

        :param f_pop_archive: filename of the pickle file
        :type f_pop_archive: str
        """

        assert type(f_pop_archive)==str
        assert self.check_integrity()

        with open(f_pop_archive, 'wb') as my_file:
            pickle.dump(self.__dict__, my_file)


    #-------------load_from_pickle_file-------------#
    def load_from_pickle_file(self, f_pop_archive):
        """Loads a population from a pickle file.

        :param f_pop_archive: filename of the pickle file
        :type f_pop_archive: str
        """

        assert type(f_pop_archive)==str

        with open(f_pop_archive, 'rb') as my_file:
            self.__dict__.update(pickle.load(my_file))

        assert self.check_integrity()
