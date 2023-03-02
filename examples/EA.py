"""``EA.py`` Script running a synchronous parallel Evolutionary Algorithm for single-objective optimization.

EA is described in:
`E. G. Talbi. Metaheuristics: From Design to Implementation. Wiley Series on Parallel and Distributed Computing. Wiley, 2009. ISBN: 9780470496909. 
<https://books.google.fr/books?id=SIsa6zi5XV8C>`_

Execution on Linux:
  * To run sequentially: ``python ./EA.py``
  * To run in parallel (in 4 computational units): ``mpiexec -n 4 python EA.py``
  * To run in parallel (in 4 computational units) specifying the units in `./hosts.txt`: ``mpiexec --machinefile ./host.txt -n 4 python EA.py``

Execution on Windows:
  * To run sequentially: ``python ./EA.py``
  * To run in parallel (in 4 computational units): ``mpiexec /np 4 python EA.py``
"""

import sys
sys.path.append('../src/')
import shutil
import os
import time
import numpy as np
from mpi4py import MPI

from Problems.Schwefel import Schwefel
from Problems.Ackley import Ackley
from Problems.Xiong import Xiong
from Problems.Rastrigin import Rastrigin
from Problems.Rosenbrock import Rosenbrock
from Problems.CEC2013 import CEC2013
from Problems.CEC2014 import CEC2014
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

    # Problem
    p = Schwefel(6)

    #--------------------------------#
    #-------------MASTER-------------#
    #--------------------------------#
    if rank==0:

        # Arguments of the search
        POP_SIZE=100
        N_GEN=10
        TIME_BUDGET=0 # in seconds (int), DoE excluded (if 0 the search stops after N_GEN generations, that corresponds to N_GEN*N_SIM*POP_SIZE simulations)
        SIM_TIME=0.0001 # in seconds, duration of 1 simulation on 1 core
        if TIME_BUDGET>0:
            assert TIME_BUDGET>SIM_TIME
            N_GEN=1000000000000

        # Files
        DIR_STORAGE="outputs"
        F_SIM_ARCHIVE=DIR_STORAGE+"/sim_archive.csv"
        F_BEST_PROFILE=DIR_STORAGE+"/best_profile.csv"
        F_INIT_POP=DIR_STORAGE+"/init_pop.csv"
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
        pop.obj_vals = np.zeros((pop.dvec.shape[0],))
        pop.obj_vals[0:nb_sim_per_proc[0]] = p.perform_real_evaluation(pop.dvec[0:nb_sim_per_proc[0]])
        for i in range(1,nprocs): # receiving from workers
            pop.obj_vals[np.sum(nb_sim_per_proc[:i]):np.sum(nb_sim_per_proc[:i+1])] = comm.recv(source=i, tag=12)
        pop.fitness_modes = True*np.ones(pop.obj_vals.shape, dtype=bool)

        # Logging
        pop.save_to_csv_file(F_INIT_POP)
        pop.save_sim_archive(F_SIM_ARCHIVE)
        pop.update_best_sim(F_BEST_PROFILE)

        # Operators
        select_op = Tournament(2)
        crossover_op = Intermediate(0.9)
        crossover_op = Two_Points(0.9)
        mutation_op = Polynomial(0.1, 50)
        replace_op = Elitist()

        if TIME_BUDGET>0:
            t_start = time.time()

        #----------------------Start Generation loop----------------------#
        for curr_gen in range(N_GEN):
            print("generation "+str(curr_gen))

            # Acquisition Process
            parents = select_op.perform_selection(pop, POP_SIZE)
            children = crossover_op.perform_crossover(parents)
            children = mutation_op.perform_mutation(children)
            assert p.is_feasible(children.dvec)

            # Parallel evaluations
            if TIME_BUDGET>0:
                t_now = time.time()
                remaining_time = TIME_BUDGET-(t_now-t_start)
                if remaining_time<=SIM_TIME:
                    break
                sim_afford = int(remaining_time//SIM_TIME)
                if np.max(nb_sim_per_proc)>sim_afford: # setting nb_sim_per_proc according to the remaining simulation budget
                    nb_sim_per_proc=sim_afford*np.ones((nprocs,), dtype=int)
                    
            for i in range(1,nprocs): # sending to workers
                comm.send(nb_sim_per_proc[i], dest=i, tag=10)
                comm.send(children.dvec[np.sum(nb_sim_per_proc[:i]):np.sum(nb_sim_per_proc[:i+1])], dest=i, tag=11)
            children.obj_vals = np.zeros((children.dvec.shape[0],))
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
            pop.update_best_sim(F_BEST_PROFILE)
        
            # Exit Generation loop if budget time exhausted
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
