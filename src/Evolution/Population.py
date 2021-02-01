import csv
import pickle
import numpy as np
import itertools

from Problems.Problem import Problem
from Global_Var import *


#------------------------------------------#
#-------------class Population-------------#
#------------------------------------------#
class Population:
    """Class for the population of an evolutionary algorithm.

    :param dvec: decision vectors of the individuals
    :type dvec: np.ndarray
    :param costs: costs associated with each individual
    :type costs: np.ndarray
    :param fitness_modes: evaluation mode associated with each individual: True for simulation (i.e. real evaluation) and False for prediction (i.e. surrogate evaluation)
    :type fitness_modes: np.ndarray
    """
    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#

    #-------------__init__-------------#
    def __init__(self, n_dvar):
        """
        __init__ method's input
        
        :param n_dvar: number of decision variables
        :type n_dvar: positive int, not zero
        """
        
        assert type(n_dvar)==int
        
        self.dvec=np.empty((0,n_dvar))
        self.costs=np.empty((0,))
        self.fitness_modes=np.empty((0,), dtype=bool)

    #-------------__del__-------------#
    def __del__(self):
        del self.dvec
        del self.costs
        del self.fitness_modes

    #-------------__str__-------------#
    def __str__(self):
        return "Population\n  Decision vectors:\n  "+str(self.dvec)+"\n  Costs:\n  "+str(self.costs)+"\n  Fitness modes:\n  "+str(self.fitness_modes)


    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#
    
    #-------------print_shapes-------------#
    def print_shapes(self):
        """Prints the shapes of the arrays `dvec`, `costs` and `fitness_modes` forming the population."""
        
        print(self.dvec.shape)
        print(self.costs.shape)
        print(self.fitness_modes.shape)
       
    #-------------check_integrity-------------#
    def check_integrity(self, pb):
        """Checks arrays' shapes are consistent.

        :param pb: the problem the population is associated with
        :type pb: Problem
        :returns: True for arrays' consistency and False otherwise
        :rtype: bool
        """
        
        assert isinstance(pb, Problem)

        return (self.dvec.shape[1]==pb.n_dvar) and ((self.costs.size==0 and self.fitness_modes.size==0) or (self.costs.shape[0]==self.dvec.shape[0] and self.fitness_modes.size==self.dvec.shape[0]))

    #-------------append-------------#
    def append(self, pop):
        """Appends individuals to the current population.
        
        :param pop: indivials to be appended
        :type pop: Population
        """
        
        assert isinstance(pop, Population)

        self.dvec = np.vstack( (self.dvec, pop.dvec) )
        self.costs = np.concatenate( (self.costs, pop.costs) )
        self.fitness_modes = np.concatenate( (self.fitness_modes, pop.fitness_modes) )
        
    #-------------sort-------------#
    def sort(self):
        """Sorts the population according to individuals' costs (ascending order)."""
        
        self.dvec = self.dvec[np.argsort(self.costs)]
        self.fitness_modes = self.fitness_modes[np.argsort(self.costs)]
        self.costs = self.costs[np.argsort(self.costs)]

    #-------------split_in_batches-------------#
    def split_in_batches(self, n_batch):
        """Splits the population in batches.

        :param n_batch: number of batches
        :type n_batch: positive int, not zero
        :returns: list of batches
        :rtype: list(Population)
        """
        
        assert self.costs.size==0 and self.fitness_modes.size==0

        batches = [Population(self.dvec.shape[0]) for i in range(n_batch)]
        batches_dvec = np.split(self.dvec, n_batch)

        for (batch, batch_dvec) in zip(batches, batches_dvec):
            batch.dvec = batch_dvec

        return batches
            
    #-------------update_best_sim-------------#
    def update_best_sim(self, f_best_profile):
        """Updates the best individual and logs.

        The best individual is a simulated individual (its `fitness_modes` is True).
        The best decision vector (minimisation assumed) is saved in `Global_Var.dvec_min`.
        The best cost is saved in `Global_Var.cost_min`.
        Both best decision vector and cost are printed to a file.

        :param f_best_profile: filename for logging
        :type f_best_profile: str
        """
        
        assert self.dvec.shape[0]==self.costs.size
        
        # sorting
        tmp_dvec = self.dvec[np.argsort(self.costs)]
        tmp_fitness_modes = self.fitness_modes[np.argsort(self.costs)]
        tmp_costs = self.costs[np.argsort(self.costs)]
        
        if np.where(tmp_fitness_modes==True)[0].size>0:
            best_idx = np.where(tmp_fitness_modes==True)[0][0]

            if tmp_costs[best_idx]<Global_Var.cost_min:
                Global_Var.cost_min = tmp_costs[best_idx]
                Global_Var.dvec_min = tmp_dvec[best_idx]

            with open(f_best_profile, 'a') as my_file:
                my_file.write(" ".join(map(str, Global_Var.dvec_min))+" "+str(Global_Var.cost_min)+"\n")

    #-------------save_to_csv_file-------------#
    def save_to_csv_file(self, f_pop_archive, pb):
        """Prints the population to a CSV file.

        The CSV file is organized as follows:
        First row: number of decision variables, number of objectives, number of fitness modes
        Second row: lower bounds of the decision variables
        Thrid row: upper bounds of the decision variables
        Remaining rows (one per individual): decision variables, costs, fitness mode

        :param f_pop_archive: filename of the CSV file.
        :type f_pop_archive: str
        :param pb: problem the population is assocated with
        :type pb: Problem
        """
        
        assert type(f_pop_archive)==str
        assert isinstance(pb, Problem)
        assert self.check_integrity(pb)
        
        with open(f_pop_archive, 'w') as my_file:
            # writing number of decision variables, number of objectives and number of fitness modes
            my_file.write(str(self.dvec.shape[1])+" "+str(self.costs.shape[1] if len(self.costs.shape)>1 else 1 if self.costs.shape[0]>0 else 0)+" "+str(self.fitness_modes.shape[1] if len(self.fitness_modes.shape)>1 else 1 if self.fitness_modes.shape[0]>0 else 0)+"\n")
            # writing bounds
            my_file.write(" ".join(map(str, pb.get_bounds()[0]))+"\n")
            my_file.write(" ".join(map(str, pb.get_bounds()[1]))+"\n")
            # writing each individuals
            for (dvec, cost, fitness_mode) in itertools.zip_longest(self.dvec, self.costs, self.fitness_modes, fillvalue=''):
                my_file.write(" ".join(map(str, dvec))+" "+str(cost)+(" "+str(int(fitness_mode)) if type(fitness_mode)==np.bool_ else "")+"\n")
                
    #-------------load_from_csv_file-------------#
    # to load from the initial population archive
    def load_from_csv_file(self, f_pop_archive, pb):
        """Loads the population from a CSV file.

        The CSV file has to be organized as follows:
        First row: number of decision variables, number of objectives, number of fitness modes
        Second row: lower bounds of the decision variables
        Thrid row: upper bounds of the decision variables
        Remaining rows (one per individual): decision variables, costs, fitness mode
        
        :param f_pop_archive: filename of the CSV file
        :type f_pop_archive: str
        :param pb: problem the population is associated with
        :type pb: Problem
        """
        
        assert type(f_pop_archive)==str
        assert isinstance(pb, Problem)

        with open(f_pop_archive, 'r') as my_file:
            # Counting the number of lines.
            reader = csv.reader(my_file, delimiter=' ')
            n_samples = sum(1 for line in reader) - 3
            my_file.seek(0)
        
            # First line: number of decision variables, number of costs and number of fitness modes
            line = next(reader)
            n_dvar = int(line[0])
            n_obj = int(line[1])
            n_fm = int(line[2])
        
            # Second line: lower bounds
            lower_bounds = np.zeros((n_dvar,))
            lower_bounds[0:n_dvar] = np.asarray(next(reader))
            assert lower_bounds.all()==pb.get_bounds()[0].all()

            # Third line: upper bounds
            upper_bounds = np.zeros((n_dvar,))
            upper_bounds[0:n_dvar] = np.asarray(next(reader))
            assert upper_bounds.all()==pb.get_bounds()[1].all()

            # Following lines contain (dvec, cost)
            self.dvec = np.zeros((n_samples, n_dvar))
            self.costs = np.zeros((n_samples,n_obj))
            self.fitness_modes = np.ones((n_samples, n_fm), dtype=bool)
            for i, line in enumerate(reader):
                self.dvec[i] = np.asarray(line[0:n_dvar])
                self.costs[i,0:n_obj] = np.asarray(line[n_dvar:n_dvar+n_obj])
                self.fitness_modes[i,0:n_fm] = np.asarray(line[n_dvar+n_obj:n_dvar+n_obj+n_fm], dtype=int)
            if self.costs.shape[1]<2:
                self.costs = np.ndarray.flatten(self.costs)
            if self.fitness_modes.shape[1]<2:
                self.fitness_modes = np.ndarray.flatten(self.fitness_modes)

        assert self.check_integrity(pb)
            
    #-------------save_sim_archive-------------#
    def save_sim_archive(self, f_sim_archive):
        """Prints the population's simulated individuals to a CSV file.

        The CSV file is organized as follows:
        One per individual: decision variables, costs, fitness mode.

        :param f_sim_archive: filename of the CSV file.
        :type f_sim_archive: str
        """

        assert type(f_sim_archive)==str
        assert self.dvec.shape[0]==self.costs.shape[0]

        idx_sim = np.where(self.fitness_modes==True)[0]
        with open(f_sim_archive, 'a') as my_file:
            for (dvec, cost) in zip(self.dvec[idx_sim], self.costs[idx_sim]):
                my_file.write(" ".join(map(str, dvec))+" "+str(cost)+"\n")

    #-------------save_to_pickle_file-------------#
    def save_to_pickle_file(self, f_pop_archive, pb):
        """Saves the population to a pickle file.

        :param f_pop_archive: filename of the pickle file
        :type f_pop_archive: str
        :param pb: problem the population is associated with
        :type pb: Problem
        """
        
        assert type(f_pop_archive)==str
        assert isinstance(pb, Problem)
        assert self.check_integrity(pb)

        with open(f_pop_archive, 'wb') as my_file:
            pickle.dump(self.__dict__, my_file)

    #-------------load_from_pickle_file-------------#
    def load_from_pickle_file(self, f_pop_archive, pb):
        """Loads a population from a pickle file.

        :param f_pop_archive: filename of the pickle file
        :type f_pop_archive: str
        :param pb: problem the population is associated with
        :type pb: Problem
        """

        assert type(f_pop_archive)==str
        assert isinstance(pb, Problem)

        with open(f_pop_archive, 'rb') as my_file:
            self.__dict__.update(pickle.load(my_file))

        assert self.check_integrity(pb)
