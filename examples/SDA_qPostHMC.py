"""``SDA_qPostHMC.py`` Script running a synchronous parallel Surrogate-Driven Algorithm based on a sub-GPs acquisition process for single-objective optimization.

The sub-GPs acquisition process is described in:
`G. Briffoteaux. Parallel surrogate-based algorithms for solving expensive optimization problems. Thesis. University of Mons (Belgium) and University of Lille (France). 2022. <https://hal.science/tel-03853862>`_

This algorithm is only meant to be run in parallel.

Execution on Linux:
  * To run in parallel (in 2 computational units): ``mpiexec -n 2 python SDA_qPostHMC.py``
  * To run in parallel (in 2 computational units) specifying the units in `./hosts.txt`: ``mpiexec --machinefile ./host.txt -n 2 python SDA_qPostHMC.py``

Execution on Windows:
  * To run in parallel (in 2 computational units): ``mpiexec /np 2 python SDA_qPostHMC.py``
"""

import shutil
import sys
sys.path.append('../src')
import os
import time
import numpy as np
from mpi4py import MPI

from Problems.Schwefel import Schwefel
from Problems.DoE import DoE

from Evolution.Population import Population
from Evolution.Tournament_Position import Tournament_Position
from Evolution.SBX import SBX
from Evolution.Polynomial import Polynomial
from Evolution.Custom_Elitism import Custom_Elitism

from Surrogates.GP_post_HMC import GP_post_HMC

from Evolution_Controls.POV_EC import POV_EC

from Global_Var import *


def main():

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    # Problem
    p = Schwefel(16)

    # Argument of the search
    q=2 # number of simulations per cycle (could be less for the last cycle according to time budget)
    POP_SIZE=50
    if q%nprocs!=0:
        print("[SDA_qPostHMC.py] error q should be a multiple of the number of processors")
        assert False
    nb_sim_per_proc=q//nprocs
    if nb_sim_per_proc>POP_SIZE:
        print("[SDA_qPostHMC.py] number of candidate per proc cannot be greater than the population size")
        assert False
    N_GEN=100
    N_CHLD=POP_SIZE


    # Files
    DIR_STORAGE="outputs"
    if rank==0:
        F_BEST_PROFILE=DIR_STORAGE+"/best_profile.csv"
        F_INIT_DB=DIR_STORAGE+"/init_db.csv"
        F_INIT_WEIGHTS=DIR_STORAGE+"/init_weights"
        shutil.rmtree(DIR_STORAGE, ignore_errors=True)
        os.makedirs(DIR_STORAGE, exist_ok=True)
    F_SIM_ARCHIVE=DIR_STORAGE+"/sim_archive.csv"
    F_TRAIN_LOG=DIR_STORAGE+"/training_log.csv"
    F_TRAINED_MODEL=DIR_STORAGE+"/trained_model"
    
    # Barrier
    comm.Barrier()

    #--------------------------------#
    #-------------MASTER-------------#
    #--------------------------------#
    if rank==0:

        TIME_BUDGET = 0
        SIM_TIME = 15
        N_CYCLES=2
        INIT_DB_SIZE=72
        if TIME_BUDGET>0:
            assert TIME_BUDGET>SIM_TIME
            N_CYCLES=1000000000000

        # Database initialization / Parallel DoE
        sampler = DoE(p)
        db = Population(p)
        db.dvec = sampler.latin_hypercube_sampling(INIT_DB_SIZE)
        nb_sim_per_proc_init = (INIT_DB_SIZE//nprocs)*np.ones((nprocs,), dtype=int) # number of simulations per proc
        for i in range(INIT_DB_SIZE%nprocs):
            nb_sim_per_proc_init[i+1]+=1
        for i in range(1,nprocs): # sending to workers
            comm.send(nb_sim_per_proc_init[i], dest=i, tag=10)
            comm.send(db.dvec[np.sum(nb_sim_per_proc_init[:i]):np.sum(nb_sim_per_proc_init[:i+1])], dest=i, tag=11)
        db.obj_vals = np.zeros((db.dvec.shape[0],))
        db.obj_vals[0:nb_sim_per_proc_init[0]] = p.perform_real_evaluation(db.dvec[0:nb_sim_per_proc_init[0]])
        for i in range(1,nprocs): # receiving from workers
            db.obj_vals[np.sum(nb_sim_per_proc_init[:i]):np.sum(nb_sim_per_proc_init[:i+1])] = comm.recv(source=i, tag=12)
        db.fitness_modes = True*np.ones(db.obj_vals.shape, dtype=bool)
        for i in range(1,nprocs): # sending to workers
            comm.send(-1, dest=i, tag=10)

        # Logging
        db.save_sim_archive(F_SIM_ARCHIVE)
        db.update_best_sim(F_BEST_PROFILE)
        del db

        # inform the worker F_SIM_ARCHIVE is ready
        for i in range(1,nprocs): # sending to workers
            comm.send(1, dest=i, tag=10)

        if TIME_BUDGET>0:
            t_start = time.time()

        # Creating surrogate
        surr = GP_post_HMC(F_SIM_ARCHIVE, p, 256, F_TRAIN_LOG, F_TRAINED_MODEL, rank, q)
        surr.perform_training()

        # Evolution Controls
        ec_op = POV_EC(surr)

        # Operators
        select_op = Tournament_Position(2)
        crossover_op = SBX(0.9, 10)
        mutation_op = Polynomial(1.0/p.n_dvar, 50)
        replace_op = Custom_Elitism(ec_op)

        #----------------------Loop over Cycles----------------------#    
        for curr_cycle in range(N_CYCLES):
            print("cycle "+str(curr_cycle))
        
            q_cands = Population(p)
            q_cands.dvec = np.zeros((q,p.n_dvar))
            q_cands.obj_vals = np.zeros((q_cands.dvec.shape[0],))

            # inform the worker a new cycle begins
            for i in range(1,nprocs): # sending to workers
                comm.send(1, dest=i, tag=10)

            # Population initialization
            pop = Population(p)
            pop.dvec = sampler.latin_hypercube_sampling(POP_SIZE)
            pop.dvec = pop.dvec[ec_op.get_sorted_indexes(pop)]

            #----------------------Evolution loop----------------------#
            for curr_gen in range(N_GEN):

                # Acquisition Process
                parents = select_op.perform_selection(pop, N_CHLD)
                children = crossover_op.perform_crossover(parents)
                children = mutation_op.perform_mutation(children)
                assert p.is_feasible(children.dvec)

                # Replacement
                replace_op.perform_replacement(pop, children)
                assert p.is_feasible(pop.dvec)
                del children
            #----------------------End evolution loop----------------------#

            # Simulating the best nb_sim_per_proc individuals
            q_cands.dvec[0:nb_sim_per_proc,:] = pop.dvec[0:nb_sim_per_proc,:]
            del pop
            q_cands.obj_vals[0:nb_sim_per_proc] = p.perform_real_evaluation(q_cands.dvec[0:nb_sim_per_proc,:])
        
            # Receiving from workers         
            for i in range(1,nprocs):
                q_cands.dvec[i*nb_sim_per_proc:(i+1)*nb_sim_per_proc,:] = comm.recv(source=i, tag=11)
                q_cands.obj_vals[i*nb_sim_per_proc:(i+1)*nb_sim_per_proc] = comm.recv(source=i, tag=12)
            q_cands.fitness_modes = True*np.ones(q_cands.obj_vals.shape, dtype=bool)

            # New simulations added to database
            q_cands.save_sim_archive(F_SIM_ARCHIVE)
            q_cands.update_best_sim(F_BEST_PROFILE)
            del q_cands

            # Exit loop over cycles if budget time exhausted
            if TIME_BUDGET>0:
                t_now = time.time()
                if TIME_BUDGET-(t_now-t_start)-(curr_cycle+1)*nb_sim_per_proc*SIM_TIME<SIM_TIME:
                    break

            # Surrogate update
            if curr_cycle!=N_CYCLES-1:
                surr.perform_training()
        
        #----------------------End loop over Cycles----------------------#
    
        # Stop workers
        for i in range(1,nprocs):
            comm.send(0, dest=i, tag=10)


    #---------------------------------#
    #-------------WORKERS-------------#
    #---------------------------------#
    else:

        # Simulating the initial database
        nsim = comm.recv(source=0, tag=10)
        while nsim!=-1:
            candidates = np.empty((nsim, p.n_dvar))
            candidates = comm.recv(source=0, tag=11)
            obj_vals = p.perform_real_evaluation(candidates)
            comm.send(obj_vals, dest=0, tag=12)
            nsim = comm.recv(source=0, tag=10)

        ctrl = comm.recv(source=0, tag=10)

        # Creating surrogate
        surr = GP_post_HMC(F_SIM_ARCHIVE, p, 256, F_TRAIN_LOG, F_TRAINED_MODEL, rank, q)

        # Evolution Controls
        ec_op = POV_EC(surr)

        # Operators
        sampler = DoE(p)
        select_op = Tournament_Position(2)
        crossover_op = SBX(0.9, 10)
        mutation_op = Polynomial(1.0/p.n_dvar, 50)
        replace_op = Custom_Elitism(ec_op)
        
        ctrl = comm.recv(source=0, tag=10)
        #----------------------Loop over Cycles----------------------#
        while ctrl!=0:
        
            # Load surrogate's new hyper-params
            surr.load_trained_model()

            # Population initialization
            pop = Population(p)
            pop.dvec = sampler.latin_hypercube_sampling(POP_SIZE)
            pop.dvec = pop.dvec[ec_op.get_sorted_indexes(pop)]

            #----------------------Evolution loop----------------------#
            for curr_gen in range(N_GEN):

                # Acquisition Process
                parents = select_op.perform_selection(pop, N_CHLD)
                children = crossover_op.perform_crossover(parents)
                children = mutation_op.perform_mutation(children)
                assert p.is_feasible(children.dvec)

                # Replacement
                replace_op.perform_replacement(pop, children)
                assert p.is_feasible(pop.dvec)
                del children
            #----------------------End evolution loop----------------------#

            # Simulating the best nb_sim_per_proc individuals
            cost = p.perform_real_evaluation(pop.dvec[0:nb_sim_per_proc,:])
        
            # Sending to master
            comm.send(pop.dvec[0:nb_sim_per_proc,:], dest=0, tag=11)
            comm.send(cost, dest=0, tag=12)
            del pop

            # Wait until a new message comes
            ctrl = comm.recv(source=0, tag=10)
        #----------------------End loop over Cycles----------------------#

    comm.Barrier()

if __name__ == "__main__":
    main()
