#!/usr/bin/python3.7
"""Script running an asynchronous parallel evolutionary algorithm.

Cannot be run sequentially.

To run in parallel (in 3 computational units): ``mpiexec -n 3 python3.7 par_async_EA.py``

To run in parallel (in 3 computational units) specifying the units in `./hosts.txt`: ``mpiexec --machinefile ./host.txt -n 3 python3.7 par_async_EA.py``

Each worker performs an acquisition process and its associated simulations. Master retrieves the simulated candidates and performs replacement.
"""

import sys
sys.path.append('../src')
import os
import sys
import time
import numpy as np
from mpi4py import MPI

from Problems.Schwefel import Schwefel
from Problems.Ackley import Ackley
from Problems.Xiong import Xiong
from Problems.Rastrigin import Rastrigin
from Problems.Rosenbrock import Rosenbrock
from Problems.CEC2013 import CEC2013
from Problems.DoE import DoE

from Evolution.Population import Population
from Evolution.Tournament import Tournament
from Evolution.SBX import SBX
from Evolution.Polynomial import Polynomial
from Evolution.Elitist import Elitist

from Global_Var import *


def main():
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
    buf = np.zeros((1,), dtype=int)

    # Problem
    N_DV = 6
    p = Schwefel(N_DV)
    # p = Ackley(N_DV)
    # N_DV = 1
    # p = Xiong()
    # p = Rastrigin(N_DV)
    # p = Rosenbrock(N_DV)
    # N_DV = 5
    # p = CEC2013(1, N_DV)

    # Arguments of the search
    POP_SIZE=100
    N_CHLD=100
    N_GEN=4 # minimum total number of generation
    # WARNING !!! the number of generations could be greater than N_GEN because if two or more procs ends at the same time, two or more generations are done
    TIME_BUDGET=0 # in seconds (int), DoE excluded (if 0 the search is limited by N_GEN, that corresponds to at least N_GEN*POP_SIZE simulations)
    if TIME_BUDGET>0:
        N_GEN=1000000000000

    # Files
    DIR_STORAGE="./outputs"
    F_SIM_ARCHIVE=DIR_STORAGE+"/sim_archive.csv"
    
    # Population
    pop = Population(p.n_dvar)


    #--------------------------------#
    #-------------MASTER-------------#
    #--------------------------------#
    if rank==0:

        # Files
        F_BEST_PROFILE=DIR_STORAGE+"/best_profile.csv"
        F_INIT_POP=DIR_STORAGE+"/init_pop.csv"
        os.system("rm -rf "+DIR_STORAGE+"/*")

        # Population initialization / Parallel DoE
        d = DoE(p)
        pop.dvec = d.latin_hypercube_sampling(POP_SIZE)
        nb_sim_per_proc = (POP_SIZE//nprocs)*np.ones((nprocs,), dtype=int) # number of simulations per proc
        for i in range(POP_SIZE%nprocs):
            nb_sim_per_proc[i+1]+=1
        for i in range(1,nprocs): # sending to workers
            comm.send(nb_sim_per_proc[i], dest=i, tag=10)
            comm.send(pop.dvec[np.sum(nb_sim_per_proc[:i]):np.sum(nb_sim_per_proc[:i+1])], dest=i, tag=11)
        pop.costs = np.zeros((pop.dvec.shape[0],))
        pop.costs[0:nb_sim_per_proc[0]] = p.perform_real_evaluation(pop.dvec[0:nb_sim_per_proc[0]])
        for i in range(1,nprocs): # receiving from workers
            pop.costs[np.sum(nb_sim_per_proc[:i]):np.sum(nb_sim_per_proc[:i+1])] = comm.recv(source=i, tag=12)
        pop.fitness_modes = True*np.ones(pop.costs.shape, dtype=bool)
        for i in range(1,nprocs): # inform workers that population initialization is done
            comm.send(-1, dest=i, tag=10)
        pop.save_to_csv_file(F_INIT_POP, p)
        pop.save_sim_archive(F_SIM_ARCHIVE)
        pop.update_best_sim(F_BEST_PROFILE)
        comm.bcast(pop, root=rank)

        # # Population initialization / Loading from a file
        # pop.load_from_csv_file(F_INIT_POP, p)
        # pop.save_sim_archive(F_SIM_ARCHIVE)
        # pop.update_best_sim(F_BEST_PROFILE)
        # comm.bcast(pop, root=rank)

        if TIME_BUDGET>0:
            t_start = time.time()

        # Operator
        replace_op = Elitist()

        # Request to workers
        req = [MPI.Request() for i in range(1,nprocs)]
        for i in range(0,nprocs-1):
            req[i] = comm.Irecv([buf,MPI.INT], source=i+1, tag=10)
        status = MPI.Status()

        #----------------------Generation loop----------------------#
        curr_gen=0
        while curr_gen<N_GEN:
            print("generation "+str(curr_gen))

            # Exit Generation loop if time budget is consumed
            if TIME_BUDGET>0:
                t_now = time.time()
                remaining_time = int(TIME_BUDGET-(t_now-t_start))
                if remaining_time<0:
                    break

            # We don't know how long we're going to wait at this point...
            MPI.Request.Waitany(req, status)

            # Determining which workers are done
            done_workers=np.empty((0,0), dtype=int)
            for i,r in enumerate(req):
                if MPI.Request.Get_status(r):
                    done_workers = np.append(done_workers, i+1)

            # For each "done workers", retrieve its associated candidates
            for actual_src in done_workers:
                assert actual_src<nprocs
                batch = Population(p.n_dvar)
                batch.load_from_pickle_file(DIR_STORAGE+"/batch"+str(actual_src)+".pkl", p)        
                batch.save_sim_archive(F_SIM_ARCHIVE)
                batch.update_best_sim(F_BEST_PROFILE)

                # Replacement
                replace_op.perform_replacement(pop, batch)
                assert p.is_feasible(pop.dvec)
                del batch

                # Sending new pop and surrogate to the worker
                comm.send(pop, dest=actual_src, tag=10)
                
                req[actual_src-1] = comm.Irecv([buf, MPI.INT], source=actual_src, tag=10)
                
                curr_gen+=1

        #----------------------End Generation loop----------------------#
            
        # Exit the program (even if workers are not done)
        os.system("kill -KILL "+str(os.getpid()))
        
    #---------------------------------#
    #-------------WORKERS-------------#
    #---------------------------------#
    else:

        # Initial population / Parallel simulations
        nsim = comm.recv(source=0, tag=10)
        while nsim!=-1:
            candidates = np.empty((nsim, p.n_dvar))
            candidates = comm.recv(source=0, tag=11)
            costs = p.perform_real_evaluation(candidates)
            comm.send(costs, dest=0, tag=12)
            nsim = comm.recv(source=0, tag=10)

        # Files
        F_BATCH_PKL_FILE=DIR_STORAGE+"/batch"+str(rank)+".pkl"
        buf[0]=rank

        # Population
        pop=comm.bcast(pop, root=0)

        # Operators
        select_op = Tournament(2)
        crossover_op = SBX(0.9, 10)
        mutation_op = Polynomial(0.1, 50)
    
        #----------------------Generation loop----------------------#
        while pop is not None:
    
            # Acquisition Process
            parents = select_op.perform_selection(pop, N_CHLD)
            children = crossover_op.perform_crossover(parents, p.get_bounds())
            children = mutation_op.perform_mutation(children, p.get_bounds())
            assert p.is_feasible(children.dvec)

            children.costs = p.perform_real_evaluation(children.dvec)
            children.fitness_modes = True*np.ones(children.costs.shape, dtype=bool)
            children.save_to_pickle_file(F_BATCH_PKL_FILE, p)
            
            # Sending to master
            comm.Isend([buf[0], MPI.INT], dest=0, tag=10)

            # Waiting for new population
            pop=comm.recv(source=0, tag=10)
        #----------------------End Generation Process loop----------------------#


if __name__ == "__main__":
    main()
