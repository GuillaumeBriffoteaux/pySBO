#!/usr/bin/python3.7
"""Script running a synchronous parallel surrogate-assisted evolutionary algorithm.

To run sequentially: ``./par_sync_SAEA.py``

To run in parallel (in 4 computational units): ``mpiexec -n 4 python3.7 par_sync_SAEA.py``

To run in parallel (in 4 computational units) specifying the units in `./hosts.txt`: ``mpiexec --machinefile ./host.txt -n 4 python3.7 par_sync_SAEA.py``

Only the simulations (i.e. real evaluations) are executed in parallel.
"""

import sys
sys.path.append('../src')
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
from Problems.DoE import DoE

from Evolution.Population import Population
from Evolution.Tournament import Tournament
from Evolution.SBX import SBX
from Evolution.Polynomial import Polynomial
from Evolution.Elitist import Elitist
from Evolution.Elitist_Multiobj import Elitist_Multiobj

from Surrogates.BNN_MCD import BNN_MCD
from Surrogates.BNN_BLR import BNN_BLR
from Surrogates.KRG import KRG
from Surrogates.GP_SMK import GP_SMK
from Surrogates.GP_Matern import GP_Matern
from Surrogates.RF import RF

from Evolution_Controls.Pareto_Tian2018_EC import Pareto_Tian2018_EC
from Evolution_Controls.Probability_Improvement_EC import Probability_Improvement_EC
from Evolution_Controls.Expected_Improvement_EC import Expected_Improvement_EC
from Evolution_Controls.Lower_Confident_Bound_EC import Lower_Confident_Bound_EC
from Evolution_Controls.Variance_EC import Variance_EC
from Evolution_Controls.Distance_EC import Distance_EC
from Evolution_Controls.Best_Predicted_EC import Best_Predicted_EC
from Evolution_Controls.Random_EC import Random_EC
from Evolution_Controls.Adaptive_Wang2020_EC import Adaptive_Wang2020_EC
from Evolution_Controls.Committee_EC import Committee_EC
from Evolution_Controls.Dynamic_EC import Dynamic_EC
from Evolution_Controls.Adaptive_EC import Adaptive_EC
from Evolution_Controls.Pareto_EC import Pareto_EC

from Global_Var import *


def main():
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

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
        N_GEN=1
        TIME_BUDGET=0 # in seconds (int), DoE excluded (if 0 the search stops after N_GEN generations, that corresponds to N_GEN*N_BATCH*N_SIM simulations)
        SIM_TIME=0.001 # in seconds, duration of 1 simulation on 1 core
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
        pop = Population(p.n_dvar)
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
        pop.save_to_csv_file(F_INIT_POP, p)
        pop.save_sim_archive(F_SIM_ARCHIVE)
        pop.update_best_sim(F_BEST_PROFILE)

        # # Population initialization / Loading from a file
        # pop = Population(p.n_dvar)
        # pop.load_from_csv_file(F_INIT_POP, p)
        # pop.save_sim_archive(F_SIM_ARCHIVE)
        # pop.update_best_sim(F_BEST_PROFILE)
    
        # Number of simulations per proc
        nb_sim_per_proc = (N_SIM//nprocs)*np.ones((nprocs,), dtype=int)
        for i in range(N_SIM%nprocs):
            nb_sim_per_proc[i+1]+=1

        if TIME_BUDGET>0:
            t_start = time.time()

        # Surrogate
        surrogate = BNN_MCD(F_SIM_ARCHIVE, p, 2*POP_SIZE, F_TRAIN_LOG, F_TRAINED_MODEL)
        # surrogate = BNN_BLR(F_SIM_ARCHIVE, p, 2*POP_SIZE, F_TRAIN_LOG, F_TRAINED_MODEL)
        # surrogate = KRG(F_SIM_ARCHIVE, p, 16, F_TRAIN_LOG, F_TRAINED_MODEL)
        # surrogate = GP_SMK(F_SIM_ARCHIVE, p, 16, F_TRAIN_LOG, F_TRAINED_MODEL)
        # surrogate = GP_Matern(F_SIM_ARCHIVE, p, 16, F_TRAIN_LOG, F_TRAINED_MODEL)
        # surrogate = RF(F_SIM_ARCHIVE, p, 2*POP_SIZE, F_TRAIN_LOG, F_TRAINED_MODEL)
        surrogate.perform_training()

        # Evolution Controls
        ec_base_f = Best_Predicted_EC(surrogate)
        ec_base_s = Variance_EC(surrogate)
        ec_base_d = Distance_EC(surrogate)
        ec_base_md = Pareto_EC([1, 1], 'hvc', ec_base_f, ec_base_d) # min pred cost, max distance
        ec_base_ms = Pareto_EC([1, 1], 'hvc', ec_base_f, ec_base_s) # min pred cost, max variance
    
        # ec_op = Random_EC()
        # ec_op = ec_base_f
        # ec_op = ec_base_s
        # ec_op = ec_base_d
    
        # ec_op = Lower_Confident_Bound_EC(surrogate)
        # ec_op = Expected_Improvement_EC(surrogate)
        # ec_op = Probability_Improvement_EC(surrogate)

        ec_op = Pareto_Tian2018_EC(surrogate)
        # ec_op = ec_base_md
        # ec_op = ec_base_ms

        # if TIME_BUDGET>0:
        #     ec_op = Adaptive_Wang2020_EC(surrogate, TIME_BUDGET, "min")
        # else:
        #     ec_op = Adaptive_Wang2020_EC(surrogate, N_GEN*N_BATCH, "min")
        # if TIME_BUDGET>0:
        #     ec_op = Adaptive_Wang2020_EC(surrogate, TIME_BUDGET, "max")
        # else:
        #     ec_op = Adaptive_Wang2020_EC(surrogate, N_GEN*N_BATCH, "max")

        # if TIME_BUDGET>0:
        #     ec_op = Dynamic_EC(TIME_BUDGET, [0.25, 0.5, 0.25], ec_base_d, ec_base_md, ec_base_f)
        # else:
        #     ec_op = Dynamic_EC(N_GEN, [0.25, 0.5, 0.25], ec_base_d, ec_base_md, ec_base_f)

        # pred_costs = surrogate.perform_prediction(pop.dvec)[0]
        # ec_op = Adaptive_EC(N_SIM, N_PRED, N_DISC, "tanh", pop.costs, pred_costs, ec_base_f, ec_base_s)

        # ec_op = Committee_EC(N_SIM, ec_base_s, ec_base_ms, ec_base_f)

        # Operators
        select_op = Tournament(2)
        crossover_op = SBX(0.9, 10)
        mutation_op = Polynomial(0.1, 50)
        replace_op = Elitist()
        # replace_op = Elitist_Multiobj(ec_base_md)


        #----------------------Start Generation loop----------------------#
        for curr_gen in range(N_GEN):
            print("generation "+str(curr_gen))

            # Acquisition Process
            parents = select_op.perform_selection(pop, N_CHLD)
            children = crossover_op.perform_crossover(parents, p.get_bounds())
            children = mutation_op.perform_mutation(children, p.get_bounds())
            assert p.is_feasible(children.dvec)

            # Update active EC in Dynamic_EC
            if isinstance(ec_op, Dynamic_EC):
                if TIME_BUDGET>0:
                    t_now = time.time()
                    elapsed_time = int(t_now-t_start)
                    ec_op.update_active(elapsed_time)
                else:
                    ec_op.update_active(curr_gen)

            #------------Start batches loop------------#
            batches = children.split_in_batches(N_BATCH)
            children = Population(p.n_dvar)
            for curr_batch,batch in enumerate(batches):

                # Update Adaptive_Wang2020_EC (batch level)
                if isinstance(ec_op, Adaptive_Wang2020_EC):
                    if TIME_BUDGET>0:
                        t_now = time.time()
                        elapsed_time = int(t_now-t_start)
                        ec_op.update_EC(elapsed_time)
                    else:
                        ec_op.update_EC((curr_gen*N_BATCH+curr_batch))

                # Evolution Control
                idx_split = ec_op.get_sorted_indexes(batch)
                subbatch_to_simulate = Population(N_DV)
                subbatch_to_simulate.dvec = batch.dvec[idx_split[0:N_SIM]]
                subbatch_to_predict = Population(N_DV)
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
                    subbatch_to_simulate.costs = np.zeros((np.sum(nb_sim_per_proc),))
                    subbatch_to_simulate.costs[0:nb_sim_per_proc[0]] = p.perform_real_evaluation(subbatch_to_simulate.dvec[0:nb_sim_per_proc[0]])
                    for i in range(1,nprocs): # receiving from workers
                        subbatch_to_simulate.costs[np.sum(nb_sim_per_proc[:i]):np.sum(nb_sim_per_proc[:i+1])] = comm.recv(source=i, tag=12)
                    subbatch_to_simulate.dvec = subbatch_to_simulate.dvec[:np.sum(nb_sim_per_proc)]
                    subbatch_to_simulate.fitness_modes = True*np.ones(np.sum(nb_sim_per_proc), dtype=bool)
                    subbatch_to_simulate.save_sim_archive(F_SIM_ARCHIVE) # logging
                    subbatch_to_simulate.update_best_sim(F_BEST_PROFILE)

                # Update active EC in Adaptive_EC
                if isinstance(ec_op, Adaptive_EC):
                    pred_costs = surrogate.perform_prediction(subbatch_to_simulate.dvec)[0]
                    ec_op.update_active(subbatch_to_simulate.costs, pred_costs)

                if not (N_SIM==0 or (N_PRED==0 and N_DISC==0)):
		    # Surrogate update
                    surrogate.perform_training()

                if N_PRED>0:
                    # Predictions
                    subbatch_to_predict.costs = surrogate.perform_prediction(subbatch_to_predict.dvec)[0]
                    subbatch_to_predict.fitness_modes = False*np.ones(subbatch_to_predict.costs.shape, dtype=bool)

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
        
            # Exit evolution loop if budget time exhausted
            if TIME_BUDGET>0:
                t_now = time.time()
                if TIME_BUDGET-(t_now-t_start)<SIM_TIME:
                    break
            
        #----------------------End Generation loop----------------------#

        # Simulate best predicted candidate from the population when optimisation has only been performed
        # on the surrogate
        if N_SIM==0:
            pop.sort()
            if np.where(pop.fitness_modes==False)[0].size>0:
                idx_best_pred = np.where(pop.fitness_modes==False)[0][0]
                pop.costs[idx_best_pred] = p.perform_real_evaluation(pop.dvec[idx_best_pred])
                pop.fitness_modes[idx_best_pred] = True
                pop.update_best_sim(F_BEST_PROFILE)
                with open(F_SIM_ARCHIVE, 'a') as my_file:
                    my_file.write(" ".join(map(str, pop.dvec[idx_best_pred]))+" "+str(pop.costs[idx_best_pred])+"\n")

        # Stop workers
        for i in range(1,nprocs):
            comm.send(-1, dest=i, tag=10)

    #---------------------------------#
    #-------------WORKERS-------------#
    #---------------------------------#
    else:
        nsim = comm.recv(source=0, tag=10)
        while nsim!=-1:
            candidates = np.empty((nsim, N_DV))
            candidates = comm.recv(source=0, tag=11)
            costs = p.perform_real_evaluation(candidates)
            comm.send(costs, dest=0, tag=12)
            nsim = comm.recv(source=0, tag=10)

            
if __name__ == "__main__":
    main()
            
