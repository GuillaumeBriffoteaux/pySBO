#!/usr/bin/python3.7
"""Script running an asynchronous parallel surrogate-assisted evolutionary algorithm.

Cannot be run sequentially.

To run in parallel (in 3 computational units): ``mpiexec -n 3 python3.7 par_async_SAEA.py``

To run in parallel (in 3 computational units) specifying the units in `./hosts.txt`: ``mpiexec --machinefile ./host.txt -n 3 python3.7 par_async_SAEA.py``

Each worker performs an acquisition process and its associated simulations. Master retrieves the simulated candidates and performs predictions, replacement and surrogate training.
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
from Evolution_Controls.Committee_EC import Committee_EC
from Evolution_Controls.Dynamic_EC import Dynamic_EC
from Evolution_Controls.Adaptive_EC import Adaptive_EC
from Evolution_Controls.Pareto_EC import Pareto_EC

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
    POP_SIZE=126
    N_CHLD=252 # number of children issued per generation
    N_SIM=96 # number of simulations per generation
    N_PRED=30 # number of predictions per generation
    N_DISC=126 # number of rejections per generation
    assert N_CHLD>0
    assert N_CHLD==N_SIM+N_PRED+N_DISC and N_DISC!=N_CHLD
    N_GEN=4 # minimum total number of generationes
    # WARNING !!! the number of generations could be greater than N_GEN because if two or more procs ends at the same time, two or more generations are done
    TIME_BUDGET=0 # in seconds (int), DoE excluded (if 0 the search is limited by N_GEN, that corresponds to at least N_GEN*N_SIM simulations)
    if TIME_BUDGET>0:
        N_GEN=1000000000000

    # Files
    DIR_STORAGE="./outputs"
    F_SIM_ARCHIVE=DIR_STORAGE+"/sim_archive.csv"
    
    # Population
    pop = Population(p.n_dvar)
    search_progress=0.0 # to update Dynamic_EC on the workers' side


    #--------------------------------#
    #-------------MASTER-------------#
    #--------------------------------#
    if rank==0:

        # Files
        F_BEST_PROFILE=DIR_STORAGE+"/best_profile.csv"
        F_INIT_POP=DIR_STORAGE+"/init_pop.csv"
        F_TRAIN_LOG=DIR_STORAGE+"/training_log.csv"
        F_TRAINED_MODEL=DIR_STORAGE+"/trained_model"
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

        # Surrogate
        surrogate = BNN_MCD(F_SIM_ARCHIVE, p, 2*POP_SIZE, F_TRAIN_LOG, F_TRAINED_MODEL)
        # surrogate = BNN_BLR(F_SIM_ARCHIVE, p, 2*POP_SIZE, F_TRAIN_LOG, F_TRAINED_MODEL)
        # surrogate = KRG(F_SIM_ARCHIVE, p, 16, F_TRAIN_LOG, F_TRAINED_MODEL)
        # surrogate = GP_SMK(F_SIM_ARCHIVE, p, 16, F_TRAIN_LOG, F_TRAINED_MODEL)
        # surrogate = GP_Matern(F_SIM_ARCHIVE, p, 16, F_TRAIN_LOG, F_TRAINED_MODEL)
        # surrogate = RF(F_SIM_ARCHIVE, p, 2*POP_SIZE, F_TRAIN_LOG, F_TRAINED_MODEL)
        surrogate.perform_training()
        for i in range(1,nprocs):
            os.system("cp -rf "+F_TRAINED_MODEL+" "+DIR_STORAGE+"/trained_model_"+str(i)+".pkl")
            comm.send(buf[0], dest=i, tag=10)

        # Operator
        # replace_op = Elitist()
        ec_base_f = Best_Predicted_EC(surrogate)
        ec_base_d = Distance_EC(surrogate)
        ec_base_md = Pareto_EC([1, 1], 'hvc', ec_base_f, ec_base_d) # min pred cost, max distance
        replace_op = Elitist_Multiobj(ec_base_md)

        # Send search progress indicator to worker (used to update Dynamic_EC)
        curr_gen=0
        if TIME_BUDGET>0:
            t_now = time.time()
            remaining_time = int(TIME_BUDGET-(t_now-t_start))
            search_progress = remaining_time
        else:
            search_progress = curr_gen
        comm.bcast(search_progress, root=rank)

        # Request to workers
        req = [MPI.Request() for i in range(1,nprocs)]
        for i in range(0,nprocs-1):
            req[i] = comm.Irecv([buf,MPI.INT], source=i+1, tag=10)
        status = MPI.Status()

        #----------------------Generation loop----------------------#
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
        
                # Logging
                if N_SIM>0:
                    batch.save_sim_archive(F_SIM_ARCHIVE)
                    batch.update_best_sim(F_BEST_PROFILE)
        
                # Surrogate update
                if not (N_SIM==0 or (N_PRED==0 and N_DISC==0)):
                    surrogate.perform_training()
                    os.system("cp -rf "+F_TRAINED_MODEL+" "+DIR_STORAGE+"/trained_model_"+str(actual_src)+".pkl")
        
                # Predictions
                if N_PRED>0:
                    batch.costs[N_SIM:N_SIM+N_PRED] = surrogate.perform_prediction(batch.dvec[N_SIM:N_SIM+N_PRED])[0]

                # Replacement
                replace_op.perform_replacement(pop, batch)
                assert p.is_feasible(pop.dvec)
                del batch

                # Sending new pop and surrogate to the worker
                comm.send(pop, dest=actual_src, tag=10)
                req[actual_src-1] = comm.Irecv([buf, MPI.INT], source=actual_src, tag=10)

                # Sending search progress indicator to the worker (to update Dynamic_EC)
                if TIME_BUDGET>0:
                    t_now = time.time()
                    remaining_time = int(TIME_BUDGET-(t_now-t_start))
                    search_progress = remaining_time
                else:
                    search_progress = curr_gen+1
                comm.send(search_progress, dest=actual_src, tag=13)

                curr_gen+=1

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
        F_TRAINED_MODEL=DIR_STORAGE+"/trained_model_"+str(rank)+".pkl"
        F_BATCH_PKL_FILE=DIR_STORAGE+"/batch"+str(rank)+".pkl"
        buf[0]=rank

        # Population
        pop=comm.bcast(pop, root=0)
        search_progress=comm.bcast(search_progress, root=0) # to update Dynamic_EC

        # Building surrogate
        _ = comm.recv(source=0, tag=10)
        surrogate = BNN_MCD(F_SIM_ARCHIVE, p, 2*POP_SIZE, "", F_TRAINED_MODEL)
        # surrogate = BNN_BLR(F_SIM_ARCHIVE, p, 2*POP_SIZE, "", F_TRAINED_MODEL)
        # surrogate = KRG(F_SIM_ARCHIVE, p, 16, "", F_TRAINED_MODEL)
        # surrogate = GP_SMK(F_SIM_ARCHIVE, p, 16, "", F_TRAINED_MODEL)
        # surrogate = GP_Matern(F_SIM_ARCHIVE, p, 16, "", F_TRAINED_MODEL)
        # surrogate = RF(F_SIM_ARCHIVE, p, 2*POP_SIZE, "", F_TRAINED_MODEL)

        # Evolution Controls
        ec_base_f = Best_Predicted_EC(surrogate)
        ec_base_s = Variance_EC(surrogate)
        ec_base_d = Distance_EC(surrogate)
        ec_base_md = Pareto_EC([1, 1], 'hvc', ec_base_f, ec_base_d) # min pred cost, max distance
        ec_base_ms = Pareto_EC([1, 1], 'hvc', ec_base_f, ec_base_s) # min pred cost, max variance
    
        # ec_op = Random_EC()
        ec_op = ec_base_f
        # ec_op = ec_base_s
        # ec_op = ec_base_d
    
        # ec_op = Lower_Confident_Bound_EC(surrogate)
        # ec_op = Expected_Improvement_EC(surrogate)
        # ec_op = Probability_Improvement_EC(surrogate)

        # ec_op = Pareto_Tian2018_EC(surrogate)
        # ec_op = ec_base_md
        # ec_op = ec_base_ms

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
    
        #----------------------Generation loop----------------------#
        while pop is not None:

            # Updating surrogate
            surrogate.load_trained_model()
    
            # Acquisition Process
            parents = select_op.perform_selection(pop, N_CHLD)
            children = crossover_op.perform_crossover(parents, p.get_bounds())
            children = mutation_op.perform_mutation(children, p.get_bounds())
            assert p.is_feasible(children.dvec)

            # Update active EC in Dynamic_EC
            if isinstance(ec_op, Dynamic_EC):
                ec_op.update_active(search_progress)

            # Evolution Control
            idx_split = ec_op.get_sorted_indexes(children)
            batch_to_simulate = Population(p.n_dvar)
            batch_to_simulate.dvec = children.dvec[idx_split[0:N_SIM]]
            batch_to_predict = Population(p.n_dvar)
            batch_to_predict.dvec = children.dvec[idx_split[N_SIM:N_SIM+N_PRED]]

            # Simulations
            if N_SIM>0:
                batch_to_simulate.costs = p.perform_real_evaluation(batch_to_simulate.dvec)
                batch_to_simulate.fitness_modes = True*np.ones(batch_to_simulate.costs.shape, dtype=bool)
            
                # Update active EC in Adaptive_EC
                if isinstance(ec_op, Adaptive_EC):
                    pred_costs = surrogate.perform_prediction(batch_to_simulate.dvec)[0]
                    ec_op.update_active(batch_to_simulate.costs, pred_costs)

            batch_to_predict.costs = float("inf")*np.ones((batch_to_predict.dvec.shape[0],), dtype=float)
            batch_to_predict.fitness_modes = False*np.ones(batch_to_predict.costs.shape, dtype=bool)

            # Merging
            batch = Population(p.n_dvar)
            batch.append(batch_to_simulate)
            batch.append(batch_to_predict)
            del batch_to_simulate
            del batch_to_predict
            batch.save_to_pickle_file(F_BATCH_PKL_FILE, p)

            # Sending to master
            comm.Isend([buf[0], MPI.INT], dest=0, tag=10)

            # Waiting for new population
            pop=comm.recv(source=0, tag=10)

            # Get the search progress (to update Dynamic_EC)
            search_progress=comm.recv(source=0, tag=13)
        #----------------------End Generation loop----------------------#


if __name__ == "__main__":
    main()
