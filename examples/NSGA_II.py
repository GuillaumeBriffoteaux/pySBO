"""``NSGA_II.py`` Script running the synchronous parallel surrogate-free Non-Domiated Sorted Genetic Algorithm for multi-objective optimization.

NSGA_II is described in:
`K. Deb, A. Pratap, S. Agarwal and T. Meyarivan. A fast and elitist multiobjective genetic algorithm: NSGA_II. IEEE Transactions on Evolutionary Computation,
6(2):182–197, 2002. <https://doi.org/10.1109/4235.996017>`_

Execution on Linux:
  * To run sequentially: ``python ./NSGA_II.py``
  * To run in parallel (in 4 computational units): ``mpiexec -n 4 python NSGA_II.py``
  * To run in parallel (in 4 computational units) specifying the units in `./hosts.txt`: ``mpiexec --machinefile ./host.txt -n 4 python NSGA_II.py``

Execution on Windows:
  * To run sequentially: ``python ./NSGA_II.py``
  * To run in parallel (in 4 computational units): ``mpiexec /np 4 python NSGA_II.py``
"""

import shutil
import sys
sys.path.append('../src/')
import os
import time
import numpy as np
from mpi4py import MPI
import pygmo

from Problems.DTLZ import DTLZ
from Problems.DoE import DoE

from Evolution.Population import Population
from Evolution.Tournament import Tournament
from Evolution.Tournament_Position import Tournament_Position
from Evolution.SBX import SBX
from Evolution.Two_Points import Two_Points
from Evolution.Intermediate import Intermediate
from Evolution.Polynomial import Polynomial
from Evolution.Elitist import Elitist

from Global_Var import *


def main():

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    # Multi-objective Problem
    p = DTLZ(5,4,3)

    
    #--------------------------------#
    #-------------MASTER-------------#
    #--------------------------------#
    if rank==0:
        
        # Arguments of the search
        TIME_BUDGET = 0
        SIM_TIME = 60
        N_GEN = 50
        if TIME_BUDGET>0:
            assert TIME_BUDGET>SIM_TIME
            N_GEN = 1000000000000
        POP_SIZE=108

        # Reference point for HV computation
        Global_Var.ref_point=np.array([100., 100., 100.])
        
        # Files
        DIR_STORAGE="outputs"
        F_SIM_ARCHIVE=DIR_STORAGE+"/sim_archive.csv"
        F_BEST_PROFILE=DIR_STORAGE+"/best_profile.csv"
        F_INIT_POP=DIR_STORAGE+"/init_pop.csv"
        F_HYPERVOLUME=DIR_STORAGE+"/hypervolume.csv"
        shutil.rmtree(DIR_STORAGE, ignore_errors=True)
        os.makedirs(DIR_STORAGE, exist_ok=True)
        
        # Population initialization / Parallel DoE
        sampler = DoE(p)
        pop = Population(p)
        pop.dvec = sampler.latin_hypercube_sampling(POP_SIZE)
        nb_sim_per_proc = (POP_SIZE//nprocs)*np.ones((nprocs,), dtype=int) # number of simulations per proc
        for i in range(POP_SIZE%nprocs):
            nb_sim_per_proc[i+1]+=1
        for i in range(1,nprocs): # sending to workers
            comm.send(nb_sim_per_proc[i], dest=i, tag=10)
            comm.send(pop.dvec[np.sum(nb_sim_per_proc[:i]):np.sum(nb_sim_per_proc[:i+1])], dest=i, tag=11)
        pop.obj_vals = np.zeros((pop.dvec.shape[0],pop.pb.n_obj))
        pop.obj_vals[0:nb_sim_per_proc[0]] = p.perform_real_evaluation(pop.dvec[0:nb_sim_per_proc[0]])
        for i in range(1,nprocs): # receiving from workers
            pop.obj_vals[np.sum(nb_sim_per_proc[:i]):np.sum(nb_sim_per_proc[:i+1])] = comm.recv(source=i, tag=12)
        pop.fitness_modes = True*np.ones(pop.obj_vals.shape, dtype=bool)
        
        # Logging
        pop.save_to_csv_file(F_INIT_POP)
        pop.save_sim_archive(F_SIM_ARCHIVE)
        pop.update_best_sim(F_BEST_PROFILE, F_HYPERVOLUME)
        
        # Start chronometer
        if TIME_BUDGET>0:
            t_start = time.time()

        # Operators
        pop.sort()
        select_op = Tournament_Position(2)
        crossover_op = SBX(0.9, 10)
        mutation_op = Polynomial(0.1, 50)
        replace_op = Elitist()

        #----------------------Start Generation loop----------------------#
        for curr_gen in range(N_GEN):
            print("generation "+str(curr_gen))

            # Acquisition Process
            parents = select_op.perform_selection(pop, POP_SIZE)
            children = crossover_op.perform_crossover(parents)
            children = mutation_op.perform_mutation(children)
            assert p.is_feasible(children.dvec)

            # Computing number of affordable simulations
            if TIME_BUDGET>0:
                t_now = time.time()
                remaining_time = TIME_BUDGET-(t_now-t_start)
                if remaining_time<=SIM_TIME:
                    break
                sim_afford = int(remaining_time//SIM_TIME)
                if np.max(nb_sim_per_proc)>sim_afford: # setting nb_sim_per_proc according to the remaining simulation budget
                    nb_sim_per_proc=sim_afford*np.ones((nprocs,), dtype=int)
                    SIM_TIME = 10000

            # Parallel simulations
            for i in range(1,nprocs): # sending to workers
                comm.send(nb_sim_per_proc[i], dest=i, tag=10)
                comm.send(children.dvec[np.sum(nb_sim_per_proc[:i]):np.sum(nb_sim_per_proc[:i+1])], dest=i, tag=11)
            children.obj_vals = np.zeros((np.sum(nb_sim_per_proc), pop.pb.n_obj))
            children.obj_vals[0:nb_sim_per_proc[0]] = p.perform_real_evaluation(children.dvec[0:nb_sim_per_proc[0]])
            for i in range(1,nprocs): # receiving from workers
                children.obj_vals[np.sum(nb_sim_per_proc[:i]):np.sum(nb_sim_per_proc[:i+1])] = comm.recv(source=i, tag=12)
            children.dvec = children.dvec[:np.sum(nb_sim_per_proc)]
            children.fitness_modes = True*np.ones(children.obj_vals.shape, dtype=bool)
            children.save_sim_archive(F_SIM_ARCHIVE) # logging

            # Replacement
            replace_op.perform_replacement(pop, children)
            assert p.is_feasible(pop.dvec)
            del children
            pop.update_best_sim(F_BEST_PROFILE, F_HYPERVOLUME)

            # Exit loop if budget time exhausted
            if TIME_BUDGET>0:
                t_now = time.time()
                if TIME_BUDGET-(t_now-t_start)<SIM_TIME:
                    break

        #----------------------End Generation loop----------------------#
        
        # Stop workers
        for i in range(1,nprocs):
            comm.send(-1, dest=i, tag=10)

    #---------------------------------#
    #-------------WORKERS-------------#
    #---------------------------------#
    else:
        nsim = comm.recv(source=0, tag=10)
        while nsim!=-1:
            candidates = np.empty((nsim, p.n_dvar))
            candidates = comm.recv(source=0, tag=11)
            obj_vals = p.perform_real_evaluation(candidates)
            comm.send(obj_vals, dest=0, tag=12)
            nsim = comm.recv(source=0, tag=10)


if __name__ == "__main__":
    main()
