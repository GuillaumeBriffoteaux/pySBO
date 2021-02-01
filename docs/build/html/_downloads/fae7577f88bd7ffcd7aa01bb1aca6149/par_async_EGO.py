#!/usr/bin/python3.7
"""Script running an asynchronous parallel efficient global optimization algorithm.

Cannot be run sequentially.

To run in parallel (in 3 computational units): ``mpiexec -n 3 python3.7 par_async_EGO.py``

To run in parallel (in 3 computational units) specifying the units in `./hosts.txt`: ``mpiexec --machinefile ./host.txt -n 3 python3.7 par_async_EGO.py``

Each worker performs an acquisition process and its associated simulations. Master retrieves the simulated candidates and performs predictions and surrogate training.
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
    INIT_DB_SIZE=32
    N_CYCLES=4 # minimum total number of Cycles
    # WARNING !!! the number of cycles could be greater than N_CYCLES because if two or more procs ends at the same time, two or more cycles are done
    q=16 # number of simulated solutions returned by each worker per cycle
    assert q>0
    TIME_BUDGET=0 # in seconds (int), DoE excluded (if 0 the search is limited by N_CYCLES, that corresponds to at least N_CYCLES*q simulations)
    if TIME_BUDGET>0:
        N_CYCLES=1000000000000
    assert N_CYCLES>0
    POP_SIZE=32
    N_GEN=5
    N_CHLD=32

    # Files
    DIR_STORAGE="./outputs"
    F_SIM_ARCHIVE=DIR_STORAGE+"/sim_archive.csv"

    # Initial database
    d = DoE(p)
    db = Population(p.n_dvar)
    search_progress = 0.0 # to update Dynamic_EC on the workers' side


    #--------------------------------#
    #-------------MASTER-------------#
    #--------------------------------#
    if rank==0:

        # Files
        F_BEST_PROFILE=DIR_STORAGE+"/best_profile.csv"
        F_INIT_DB=DIR_STORAGE+"/init_pop.csv"
        F_TRAIN_LOG=DIR_STORAGE+"/training_log.csv"
        F_TRAINED_MODEL=DIR_STORAGE+"/trained_model"
        os.system("rm -rf "+DIR_STORAGE+"/*")
        
        # Database initialization / Parallel DoE
        db.dvec = d.latin_hypercube_sampling(INIT_DB_SIZE)
        nb_sim_per_proc = (INIT_DB_SIZE//nprocs)*np.ones((nprocs,), dtype=int) # number of simulations per proc
        for i in range(INIT_DB_SIZE%nprocs):
            nb_sim_per_proc[i+1]+=1
        for i in range(1,nprocs): # sending to workers
            comm.send(nb_sim_per_proc[i], dest=i, tag=10)
            comm.send(db.dvec[np.sum(nb_sim_per_proc[:i]):np.sum(nb_sim_per_proc[:i+1])], dest=i, tag=11)
        db.costs = np.zeros((db.dvec.shape[0],))
        db.costs[0:nb_sim_per_proc[0]] = p.perform_real_evaluation(db.dvec[0:nb_sim_per_proc[0]])
        for i in range(1,nprocs): # receiving from workers
            db.costs[np.sum(nb_sim_per_proc[:i]):np.sum(nb_sim_per_proc[:i+1])] = comm.recv(source=i, tag=12)
        db.fitness_modes = True*np.ones(db.costs.shape, dtype=bool)
        for i in range(1,nprocs): # inform workers that population initialization is done
            comm.send(-1, dest=i, tag=10)
        db.save_to_csv_file(F_INIT_DB, p)
        db.save_sim_archive(F_SIM_ARCHIVE)
        db.update_best_sim(F_BEST_PROFILE)
        comm.bcast(db, root=rank) # to update Adaptive_EC on the workers' side
        del db

        # # Database initialization / Loading from a file
        # db = Population(p.n_dvar)
        # db.load_from_csv_file(F_INIT_DB, p)
        # db.save_sim_archive(F_SIM_ARCHIVE)
        # db.update_best_sim(F_BEST_PROFILE)
        # for i in range(1,nprocs): # inform workers that population initialization is done
        #     comm.send(-1, dest=i, tag=10)
        # comm.bcast(db, root=rank)
        # del db

        if TIME_BUDGET>0:
            t_start = time.time()
        
        # Surrogate
        surrogate = BNN_MCD(F_SIM_ARCHIVE, p, INIT_DB_SIZE, F_TRAIN_LOG, F_TRAINED_MODEL)
        # surrogate = BNN_BLR(F_SIM_ARCHIVE, p, INIT_DB_SIZE, F_TRAIN_LOG, F_TRAINED_MODEL)
        # surrogate = KRG(F_SIM_ARCHIVE, p, 16, F_TRAIN_LOG, F_TRAINED_MODEL)
        # surrogate = GP_SMK(F_SIM_ARCHIVE, p, 16, F_TRAIN_LOG, F_TRAINED_MODEL)
        # surrogate = GP_Matern(F_SIM_ARCHIVE, p, 16, F_TRAIN_LOG, F_TRAINED_MODEL)
        # surrogate = RF(F_SIM_ARCHIVE, p, INIT_DB_SIZE, F_TRAIN_LOG, F_TRAINED_MODEL)
        surrogate.perform_training()
        for i in range(1,nprocs):
            os.system("cp -rf "+F_TRAINED_MODEL+" "+DIR_STORAGE+"/trained_model_"+str(i)+".pkl")
            comm.send(buf[0], dest=i, tag=10)

        # Send search progress indicator to worker (used to update Dynamic_EC)
        curr_cycle=0
        if TIME_BUDGET>0:
            t_now = time.time()
            remaining_time = int(TIME_BUDGET-(t_now-t_start))
            search_progress = remaining_time
        else:
            search_progress = curr_cycle
        comm.bcast(search_progress, root=rank)

        # Request to workers
        req = [MPI.Request() for i in range(1,nprocs)]
        for i in range(0,nprocs-1):
            req[i] = comm.Irecv([buf,MPI.INT], source=i+1, tag=10)
        status = MPI.Status()

        #----------------------Cycle loop----------------------#
        while curr_cycle<N_CYCLES:
            print("cycle "+str(curr_cycle))

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
        
                # Update the database
                batch.save_sim_archive(F_SIM_ARCHIVE)
                batch.update_best_sim(F_BEST_PROFILE)
        
                # Surrogate update
                surrogate.perform_training()
                os.system("cp -rf "+F_TRAINED_MODEL+" "+DIR_STORAGE+"/trained_model_"+str(actual_src)+".pkl")

                # Sending new surrogate to the worker
                comm.send(buf[0], dest=actual_src, tag=10)
                req[actual_src-1] = comm.Irecv([buf, MPI.INT], source=actual_src, tag=10)

                # Sending search progress indicator to the worker (to update Dynamic_EC)
                if TIME_BUDGET>0:
                    t_now = time.time()
                    remaining_time = int(TIME_BUDGET-(t_now-t_start))
                    search_progress = remaining_time
                else:
                    search_progress = curr_cycle+1
                comm.send(search_progress, dest=actual_src, tag=13)
                
                curr_cycle+=1

        #----------------------End Cycle Process loop----------------------#

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
    
        buf[0]=rank
        # Retrieving initial database to initialize Adaptive_EC
        db=comm.bcast(db, root=0)

        # Files
        F_TRAINED_MODEL=DIR_STORAGE+"/trained_model_"+str(rank)+".pkl"
        F_BATCH_PKL_FILE=DIR_STORAGE+"/batch"+str(rank)+".pkl"

        # Surrogate
        ctrl = comm.recv(source=0, tag=10)
        surrogate = BNN_MCD(F_SIM_ARCHIVE, p, INIT_DB_SIZE, "", F_TRAINED_MODEL)
        # surrogate = BNN_BLR(F_SIM_ARCHIVE, p, INIT_DB_SIZE, "", F_TRAINED_MODEL)
        # surrogate = KRG(F_SIM_ARCHIVE, p, 16, "", F_TRAINED_MODEL)
        # surrogate = GP_SMK(F_SIM_ARCHIVE, p, 16, "", F_TRAINED_MODEL)
        # surrogate = GP_Matern(F_SIM_ARCHIVE, p, 16, "", F_TRAINED_MODEL)
        # surrogate = RF(F_SIM_ARCHIVE, p, INIT_DB_SIZE, "", F_TRAINED_MODEL)

        # Infill Criteria
        ic_base_f = Best_Predicted_EC(surrogate)
        ic_base_s = Variance_EC(surrogate)
        ic_base_d = Distance_EC(surrogate)
        ic_base_md = Pareto_EC([1, 1], 'hvc', ic_base_f, ic_base_d) # min pred cost, max distance
        ic_base_ms = Pareto_EC([1, 1], 'hvc', ic_base_f, ic_base_s) # min pred cost, max variance
    
        # ic_op = Random_EC()
        # ic_op = ic_base_f
        # ic_op = ic_base_s
        ic_op = ic_base_d
    
        # ic_op = Lower_Confident_Bound_EC(surrogate)
        # ic_op = Expected_Improvement_EC(surrogate)
        # ic_op = Probability_Improvement_EC(surrogate)

        # ic_op = Pareto_Tian2018_EC(surrogate)
        # ic_op = ic_base_md
        # ic_op = ic_base_ms

        # if TIME_BUDGET>0:
        #     ic_op = Dynamic_EC(TIME_BUDGET, [0.25, 0.5, 0.25], ic_base_d, ic_base_md, ic_base_f)
        # else:
        #     ic_op = Dynamic_EC(N_CYCLES, [0.25, 0.5, 0.25], ic_base_d, ic_base_md, ic_base_f)

        # pred_costs = surrogate.perform_prediction(db.dvec)[0]
        # ic_op = Adaptive_EC(0, 0, 0, "tanh", db.costs, pred_costs, ic_base_f, ic_base_s)

        # ic_op = Committee_EC(0, ic_base_s, ic_base_ms, ic_base_f)

        # Operators for the Acquisition Process
        select_op = Tournament(2)
        crossover_op = SBX(0.9, 10)
        mutation_op = Polynomial(0.1, 50)
        replace_op = Elitist()
        # replace_op = Elitist_Multiobj(ic_base_md)

        # Population initialization
        pop = Population(p.n_dvar)

        del db # has been used to initiate Adaptive_EC, no usefull anymore
        search_progress=comm.bcast(search_progress, root=0) # to update Dynamic_EC

        #----------------Cycle loop----------------#
        while ctrl is not None:

            # Updating the surrogate
            surrogate.load_trained_model()
    
            # Population initialization
            pop.dvec = d.random_uniform_sampling(POP_SIZE)
            pop.costs = ic_op.get_IC_value(pop.dvec)
            if isinstance(ic_op, Adaptive_EC):
                ic_op.to_be_updated=False
            pop.fitness_modes = False*np.ones(pop.costs.shape, dtype=bool)

            # Update active EC in Dynamic_EC
            if isinstance(ic_op, Dynamic_EC):
                ic_op.update_active(search_progress)

            #----------------Evolution loop----------------#
            for curr_gen in range(N_GEN):

                # Reproduction
                parents = select_op.perform_selection(pop, N_CHLD)
                children = crossover_op.perform_crossover(parents, p.get_bounds())
                children = mutation_op.perform_mutation(children, p.get_bounds())
                assert p.is_feasible(children.dvec)

                # Children evaluation
                children.costs = ic_op.get_IC_value(children.dvec)
                children.fitness_modes = False*np.ones(children.costs.shape, dtype=bool)

                # Replacement
                replace_op.perform_replacement(pop, children)
                assert p.is_feasible(pop.dvec)
                del children
            #----------------end Evolution loop----------------#

            # q simulations
            pop.costs[0:q] = p.perform_real_evaluation(pop.dvec[0:q])
            pop.fitness_modes[0:q] = True

            # Saving the q simulated candidates and sending to master
            batch = Population(p.n_dvar)
            batch.dvec = pop.dvec[0:q]
            batch.costs = pop.costs[0:q]
            batch.fitness_modes = pop.fitness_modes[0:q]
            batch.save_to_pickle_file(F_BATCH_PKL_FILE, p)
            comm.Isend([buf[0], MPI.INT], dest=0, tag=10)

            # Update active EC in Adaptive_EC
            if isinstance(ic_op, Adaptive_EC):
                pred_costs = surrogate.perform_prediction(batch.dvec)[0]
                ic_op.update_active(batch.costs, pred_costs)

            # Updating surrogate
            ctrl = comm.recv(source=0, tag=10)

            # Get the search progress (to update Dynamic_EC)
            search_progress=comm.recv(source=0, tag=13)

        #----------------Cycle Process loop----------------#


if __name__ == "__main__":
    main()
        
