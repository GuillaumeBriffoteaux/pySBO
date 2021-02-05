#!/usr/bin/python3.7
"""Script running a synchronous parallel evolutionary algorithm.

To run sequentially: ``./par_sync_NSGA-II.py``

To run in parallel (in 4 computational units): ``mpiexec -n 4 python3.7 par_sync_NSGA-II.py``

To run in parallel (in 4 computational units) specifying the units in `./hosts.txt`: ``mpiexec --machinefile ./host.txt -n 4 python3.7 par_sync_NSGA-II.py``

Only the simulations (i.e. real evaluations) are executed in parallel.
"""

import sys
sys.path.append('../src/')
import os
import time
import numpy as np
from mpi4py import MPI

from Problems.Schwefel import Schwefel
from Problems.ZDT import ZDT
from Problems.DTLZ import DTLZ
from Problems.DoE import DoE

from Evolution.Population import Population
from Evolution.Tournament_Position import Tournament_Position
from Evolution.SBX import SBX
from Evolution.Polynomial import Polynomial
from Evolution.Elitist import Elitist

from Surrogates.BNN_MCD import BNN_MCD
from Surrogates.GP_Multitask_RBF import GP_Multitask_RBF
from Surrogates.RF import RF

from Evolution_Controls.Random_EC import Random_EC
from Evolution_Controls.Distance_EC import Distance_EC

from Global_Var import *


def main():

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    # # Bi-objective Problem
    # N_DV = 3
    # p = ZDT(4, N_DV)
    # print(p)

    # Multi-objective Problem
    N_DV = 4
    N_OBJ = 3
    p = DTLZ(1, N_DV, N_OBJ)
    print(p)

    
    #--------------------------------#
    #-------------MASTER-------------#
    #--------------------------------#
    if rank==0:

        # Arguments of the search
        POP_SIZE=126
        N_CHLD=252 # number of children issued per generation
        N_BATCH=2 # number of batches per generation
        BATCH_SIZE=126 # number of solutions per batch
        assert N_CHLD==N_BATCH*BATCH_SIZE
        N_SIM=48 # number of simulations per batch
        N_PRED=15 # number of predictions per batch
        N_DISC=63 # number of rejections per batch 
        assert BATCH_SIZE==N_SIM+N_PRED+N_DISC
        N_GEN=10
        TIME_BUDGET=0 # in seconds (int), DoE excluded (if 0 the search stops after N_GEN generations, that corresponds to N_GEN*N_BATCH*N_SIM simulations)
        SIM_TIME=0.5 # in seconds, duration of 1 simulation on 1 core
        if TIME_BUDGET>0:
            assert TIME_BUDGET>SIM_TIME
            N_GEN=1000000000000

        # Files
        DIR_STORAGE="./outputs"
        F_SIM_ARCHIVE=DIR_STORAGE+"/sim_archive.csv"
        F_BEST_PROFILE=DIR_STORAGE+"/best_profile.csv"
        F_INIT_POP=DIR_STORAGE+"/init_pop.csv"
        F_TRAIN_LOG=DIR_STORAGE+"/training_log.csv"
        F_TRAINED_MODEL=DIR_STORAGE+"/trained_model"
        os.system("rm -rf "+DIR_STORAGE+"/*")

        # Population initialization / Parallel DoE
        d = DoE(p)
        pop = Population(p)
        pop.dvec = d.latin_hypercube_sampling(POP_SIZE)
        nb_sim_per_proc = (POP_SIZE//nprocs)*np.ones((nprocs,), dtype=int) # number of simulations per proc
        for i in range(POP_SIZE%nprocs):
            nb_sim_per_proc[i+1]+=1
        for i in range(1,nprocs): # sending to workers
            comm.send(nb_sim_per_proc[i], dest=i, tag=10)
            comm.send(pop.dvec[np.sum(nb_sim_per_proc[:i]):np.sum(nb_sim_per_proc[:i+1])], dest=i, tag=11)
        pop.costs = np.zeros((pop.dvec.shape[0],pop.pb.n_obj))
        pop.costs[0:nb_sim_per_proc[0]] = p.perform_real_evaluation(pop.dvec[0:nb_sim_per_proc[0]])
        for i in range(1,nprocs): # receiving from workers
            pop.costs[np.sum(nb_sim_per_proc[:i]):np.sum(nb_sim_per_proc[:i+1])] = comm.recv(source=i, tag=12)
        pop.fitness_modes = True*np.ones((pop.costs.shape[0],), dtype=bool)

        pop.save_to_csv_file(F_INIT_POP)
        pop.save_sim_archive(F_SIM_ARCHIVE)
        pop.update_best_sim(F_BEST_PROFILE)

        # # Population initialization / Loading from a file
        # pop = Population(p)
        # pop.load_from_csv_file(F_INIT_POP)
        # pop.save_sim_archive(F_SIM_ARCHIVE)
        # pop.update_best_sim(F_BEST_PROFILE)

        # Number of simulations per proc
        nb_sim_per_proc = (N_SIM//nprocs)*np.ones((nprocs,), dtype=int)
        for i in range(N_SIM%nprocs):
            nb_sim_per_proc[i+1]+=1

        if TIME_BUDGET>0:
            t_start = time.time()

        # Surrogate
        # surrogate = BNN_MCD(F_SIM_ARCHIVE, p, 2*POP_SIZE, F_TRAIN_LOG, F_TRAINED_MODEL)
        surrogate = GP_Multitask_RBF(F_SIM_ARCHIVE, p, 16, F_TRAIN_LOG, F_TRAINED_MODEL)
        # surrogate = RF(F_SIM_ARCHIVE, p, 2*POP_SIZE, F_TRAIN_LOG, F_TRAINED_MODEL)
        surrogate.perform_training()

        # Evolution Controls
        # ec_op = Random_EC()
        ec_op = Distance_EC(surrogate)
        
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

            #------------Start batches loop------------#
            batches = children.split_in_batches(N_BATCH)
            children = Population(p)
            for curr_batch,batch in enumerate(batches):

                # Evolution Control
                idx_split = ec_op.get_sorted_indexes(batch)
                subbatch_to_simulate = Population(p)
                subbatch_to_simulate.dvec = batch.dvec[idx_split[0:N_SIM]]
                subbatch_to_predict = Population(p)
                subbatch_to_predict.dvec = batch.dvec[idx_split[N_SIM:N_SIM+N_PRED]]

                if N_SIM>0:
                    # Parallel Simulations
                    if TIME_BUDGET>0:
                        t_now = time.time()
                        remaining_time = TIME_BUDGET-(t_now-t_start)
                        if remaining_time<SIM_TIME:
                            break
                        sim_afford = int(remaining_time//SIM_TIME)
                        if np.max(nb_sim_per_proc)>sim_afford: # setting nb_sim_per_proc according to the remaining simulation budget
                            nb_sim_per_proc=sim_afford*np.ones((nprocs,), dtype=int)
                            
                    for i in range(1,nprocs): # sending to workers
                        comm.send(nb_sim_per_proc[i], dest=i, tag=10)
                        comm.send(subbatch_to_simulate.dvec[np.sum(nb_sim_per_proc[:i]):np.sum(nb_sim_per_proc[:i+1])], dest=i, tag=11)
                    subbatch_to_simulate.costs = np.zeros((np.sum(nb_sim_per_proc),subbatch_to_simulate.pb.n_obj))
                    subbatch_to_simulate.costs[0:nb_sim_per_proc[0]] = p.perform_real_evaluation(subbatch_to_simulate.dvec[0:nb_sim_per_proc[0]])
                    for i in range(1,nprocs): # receiving from workers
                        subbatch_to_simulate.costs[np.sum(nb_sim_per_proc[:i]):np.sum(nb_sim_per_proc[:i+1])] = comm.recv(source=i, tag=12)
                    subbatch_to_simulate.dvec = subbatch_to_simulate.dvec[:np.sum(nb_sim_per_proc)]
                    subbatch_to_simulate.fitness_modes = True*np.ones(np.sum(nb_sim_per_proc), dtype=bool)
                    subbatch_to_simulate.save_sim_archive(F_SIM_ARCHIVE) # logging
                    # subbatch_to_simulate.update_best_sim(F_BEST_PROFILE)

                if not (N_SIM==0 or (N_PRED==0 and N_DISC==0)):
        	    # Surrogate update
                    surrogate.perform_training()

                if N_PRED>0:
                    # Predictions
                    subbatch_to_predict.costs = surrogate.perform_prediction(subbatch_to_predict.dvec)[0]
                    subbatch_to_predict.fitness_modes = False*np.ones((subbatch_to_predict.costs.shape[0],), dtype=bool)

                # Merging evaluated batches
                children.append(subbatch_to_simulate)
                children.append(subbatch_to_predict)
                del subbatch_to_simulate
                del subbatch_to_predict

                # Exit batch loop if budget time exhausted
                if TIME_BUDGET>0:
                    t_now = time.time()
                    if TIME_BUDGET-(t_now-t_start)<SIM_TIME:
                        break
                    
            #------------End batches loop------------#

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
            costs = p.perform_real_evaluation(candidates)
            comm.send(costs, dest=0, tag=12)
            nsim = comm.recv(source=0, tag=10)


if __name__ == "__main__":
    main()
