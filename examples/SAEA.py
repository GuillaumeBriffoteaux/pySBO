"""``SAEA.py`` Script running a synchronous parallel Surrogate-Assisted Evolutionary Algorithm for single-objective optimization.
The surrogate is used as an evaluator and/or a filter.

SAEA is described in:
`G. Briffoteaux, R. Ragonnet, M. Mezmaz, N. Melab and D. Tuyttens. Evolution Control Ensemble Models for Surrogate-Assisted Evolutionary Algorithms. In HPCS 2020 - International Conference on High Performance Computing and Simulation, 22-27 March 2021, Online conference. <https://hal.inria.fr/hal-03332521>`_


Execution on Linux:
  * To run sequentially: ``python ./SAEA.py``
  * To run in parallel (in 4 computational units): ``mpiexec -n 4 python SAEA.py``
  * To run in parallel (in 4 computational units) specifying the units in `./hosts.txt`: ``mpiexec --machinefile ./host.txt -n 4 python SAEA.py``

Execution on Windows:
  * To run sequentially: ``python ./SAEA.py``
  * To run in parallel (in 4 computational units): ``mpiexec /np 4 python SAEA.py``
"""

import shutil
import sys
sys.path.append('../src')
import os
import time
import numpy as np
from mpi4py import MPI
import itertools

from Problems.Schwefel import Schwefel
from Problems.DoE import DoE

from Evolution.Population import Population
from Evolution.Tournament import Tournament
from Evolution.SBX import SBX
from Evolution.Polynomial import Polynomial
from Evolution.Elitist import Elitist

from Surrogates.BNN_MCD import BNN_MCD
from Surrogates.BLR_ANN import BLR_ANN
from Surrogates.iKRG import iKRG
from Surrogates.rKRG import rKRG
from Surrogates.GP import GP

from Evolution_Controls.Random_EC import Random_EC
from Evolution_Controls.POV_EC import POV_EC
from Evolution_Controls.Distance_EC import Distance_EC
from Evolution_Controls.Pred_Stdev_EC import Pred_Stdev_EC
from Evolution_Controls.Expected_Improvement_EC import Expected_Improvement_EC
from Evolution_Controls.Probability_Improvement_EC import Probability_Improvement_EC
from Evolution_Controls.Lower_Confident_Bound_EC import Lower_Confident_Bound_EC
from Evolution_Controls.Pareto_EC import Pareto_EC
from Evolution_Controls.Pareto_Tian2018_EC import Pareto_Tian2018_EC
from Evolution_Controls.Dynamic_Exclusive_EC import Dynamic_Exclusive_EC
from Evolution_Controls.Dynamic_Inclusive_EC import Dynamic_Inclusive_EC
from Evolution_Controls.Adaptive_EC import Adaptive_EC
from Evolution_Controls.Adaptive_Wang2020_EC import Adaptive_Wang2020_EC
from Evolution_Controls.Committee_EC import Committee_EC

from Global_Var import *

def main():

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    # Problem
    p = Schwefel(16)
    
    #--------------------------------#
    #-------------MASTER-------------#
    #--------------------------------#
    if rank==0:
    
        # Search arguments
        TIME_BUDGET = 0
        SIM_TIME = 15
        POP_SIZE = 72
        N_CHLD = 288 # number of children issued per generation
        N_BATCH = 1 # number of batches per generation
        N_SIM = 72 # number of simulations per batch
        N_PRED = 72 # number of predictions per batch
        N_DISC = 144 # number of rejections per batch
        assert (N_SIM+N_PRED+N_DISC)*N_BATCH==N_CHLD
        assert N_SIM!=0
        N_GEN = 2
        if TIME_BUDGET > 0:
            assert TIME_BUDGET > SIM_TIME
            N_GEN = 1000000000000

        # Files
        DIR_STORAGE="outputs"
        F_SIM_ARCHIVE=DIR_STORAGE+"/sim_archive.csv"
        F_BEST_PROFILE=DIR_STORAGE+"/best_profile.csv"
        F_INIT_POP=DIR_STORAGE+"/init_pop.csv"
        F_TRAIN_LOG=DIR_STORAGE+"/training_log.csv"
        F_TRAINED_MODEL=DIR_STORAGE+"/trained_model"
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

        # Number of simulations per proc
        nb_sim_per_proc = (N_SIM//nprocs)*np.ones((nprocs,), dtype=int)
        for i in range(N_SIM%nprocs):
            nb_sim_per_proc[i+1]+=1

        # Operators
        select_op = Tournament(2)
        crossover_op = SBX(0.9, 10)
        mutation_op = Polynomial(1.0/p.n_dvar, 50)
        replace_op = Elitist()

        # Start chronometer
        if TIME_BUDGET>0:
            t_start = time.time()

        # Creating surrogate            
        # surr = BNN_MCD(F_SIM_ARCHIVE, p, float('inf'), F_TRAIN_LOG, F_TRAINED_MODEL, 5)
        # surr = BLR_ANN(F_SIM_ARCHIVE, p, 256, F_TRAIN_LOG, F_TRAINED_MODEL)
        # surr = iKRG(F_SIM_ARCHIVE, p, 18, F_TRAIN_LOG, F_TRAINED_MODEL)
        # surr = rKRG(F_SIM_ARCHIVE, p, 18, F_TRAIN_LOG, F_TRAINED_MODEL)
        surr = GP(F_SIM_ARCHIVE, p, 72, F_TRAIN_LOG, F_TRAINED_MODEL, 'rbf')
        surr.perform_training()
        
        # Evolution Controls
        ec_op = Expected_Improvement_EC(surr)
        # ec_base_y = POV_EC(surr)
        # ec_base_d = Distance_EC(surr)
        # ec_base_s = Pred_Stdev_EC(surr)
        # if sys.argv[1]=="rand":
        #     ec_op = Random_EC()
        # elif sys.argv[1]=="pov":
        #     ec_op = ec_base_y
        # elif sys.argv[1]=="dist":
        #     ec_op = ec_base_d
        # elif sys.argv[1]=="stdev":
        #     ec_op = ec_base_s
        # elif sys.argv[1]=="ei":
        #     ec_op = Expected_Improvement_EC(surr)
        # elif sys.argv[1]=="pi":
        #     ec_op = Probability_Improvement_EC(surr)
        # elif sys.argv[1]=="lcb":
        #     ec_op = Lower_Confident_Bound_EC(surr)
        # elif sys.argv[1]=="par-fd-cd":
        #     ec_op = Pareto_EC([1.0, 1.0], "cd", ec_base_y, ec_base_d)
        # elif sys.argv[1]=="par-fs-hvc":
        #     ec_op = Pareto_EC([1.0, 1.0], "hvc", ec_base_y, ec_base_s)
        # elif sys.argv[1]=="par-tian":
        #     ec_op = Pareto_Tian2018_EC([1.0, -1.0], ec_base_y, ec_base_s)
        # elif sys.argv[1]=="dyn-df-excl":
        #     if TIME_BUDGET>0:
        #         ec_op = Dynamic_Exclusive_EC(TIME_BUDGET, [0.5, 0.5], ec_base_d, ec_base_y)
        #     else:
        #         ec_op = Dynamic_Exclusive_EC(N_GEN*N_BATCH, [0.5, 0.5], ec_base_d, ec_base_y)
        # elif sys.argv[1]=="dyn-dpf-excl":
        #     if TIME_BUDGET>0:
        #         ec_tmp = Pareto_EC([1.0, 1.0], "cd", ec_base_y, ec_base_d)
        #         ec_op = Dynamic_Exclusive_EC(TIME_BUDGET, [0.25, 0.5, 0.25], ec_base_d, ec_tmp, ec_base_y)
        #     else:
        #         ec_tmp = Pareto_EC([1.0, 1.0], "cd", ec_base_y, ec_base_d)
        #         ec_op = Dynamic_Exclusive_EC(N_GEN*N_BATCH, [0.25, 0.5, 0.25], ec_base_d, ec_tmp, ec_base_y)
        # elif sys.argv[1]=="dyn-df-incl":
        #     if TIME_BUDGET>0:
        #         ec_op = Dynamic_Inclusive_EC(TIME_BUDGET, N_SIM, N_PRED, ec_base_d, ec_base_y)
        #     else:
        #         ec_op = Dynamic_Inclusive_EC(N_GEN*N_BATCH, N_SIM, N_PRED, ec_base_d, ec_base_y)
        # elif sys.argv[1]=="ada-dpf":
        #     norm_pred_obj_vals = surr.perform_prediction(pop.dvec)[0]
        #     norm_pop_obj_vals = surr.normalize_obj_vals(pop.obj_vals)
        #     norm_y_min = surr.normalize_obj_vals(Global_Var.obj_val_min)[0]
        #     ec_tmp = Pareto_EC([1.0, 1.0], "cd", ec_base_y, ec_base_d)
        #     ec_op = Adaptive_EC(norm_pop_obj_vals, norm_pred_obj_vals, norm_y_min, ec_base_d, ec_tmp, ec_base_y)
        # elif sys.argv[1]=="ada-wang":
        #     if TIME_BUDGET>0:
        #         ec_op = Adaptive_Wang2020_EC(surr, TIME_BUDGET, "max")
        #     else:
        #         ec_op = Adaptive_Wang2020_EC(surr, N_GEN*N_BATCH, "max")
        # elif sys.argv[1]=="com-dpf":
        #     ec_tmp = Pareto_EC([1.0, 1.0], "cd", ec_base_y, ec_base_d)
        #     ec_op = Committee_EC(N_SIM, ec_base_d, ec_tmp, ec_base_y)
        # else:
        #     print("[SAEA.py] error invalid ec name")
        #     assert False

        
        #----------------------Start Generation loop----------------------#
        for curr_gen in range(N_GEN):
            print("generation "+str(curr_gen))

            # Acquisition Process
            parents = select_op.perform_selection(pop, N_CHLD)
            children = crossover_op.perform_crossover(parents)
            children = mutation_op.perform_mutation(children)
            assert p.is_feasible(children.dvec)

            #------------Start batches loop------------#
            batches = children.split_in_batches(N_BATCH)
            children = Population(p)
            for curr_batch,batch in enumerate(batches):
                print("    batch "+str(curr_batch))

                # Setting number of simulations per proc and update dynamic and adaptive ECs
                if TIME_BUDGET>0:
                    t_now = time.time()
                    elapsed_time = (t_now-t_start)

                    # Update active EC in Dynamic_EC or Adaptive_Wang2020_EC (batch level)
                    if isinstance(ec_op, Dynamic_Exclusive_EC) or isinstance(ec_op, Dynamic_Inclusive_EC):
                        ec_op.update_active(elapsed_time)
                    if isinstance(ec_op, Adaptive_Wang2020_EC):
                        ec_op.update_EC(elapsed_time)
                
                    remaining_time = TIME_BUDGET-elapsed_time
                    if remaining_time<=SIM_TIME:
                        break
                    sim_afford = int(remaining_time//SIM_TIME)
                    if np.max(nb_sim_per_proc)>sim_afford: # setting nb_sim_per_proc according to the remaining simulation budget
                        nb_sim_per_proc = sim_afford*np.ones((nprocs,), dtype=int)
                        SIM_TIME = 10000
                else:
                    # Update active EC in Dynamic_EC or Adaptive_Wang2020_EC (batch level)
                    if isinstance(ec_op, Dynamic_Exclusive_EC) or isinstance(ec_op, Dynamic_Inclusive_EC):
                        ec_op.update_active(curr_gen*N_BATCH+curr_batch)
                    if isinstance(ec_op, Adaptive_Wang2020_EC):
                        ec_op.update_EC(curr_gen*N_BATCH+curr_batch)
                        
                # Evolution Control
                idx_split = ec_op.get_sorted_indexes(batch)
                subbatch_to_simulate = Population(p)
                subbatch_to_simulate.dvec = batch.dvec[idx_split[0:N_SIM]]
                subbatch_to_predict = Population(p)
                subbatch_to_predict.dvec = batch.dvec[idx_split[N_SIM:N_SIM+N_PRED]]

                # Parallel simulations
                for i in range(1,nprocs): # sending to workers
                    comm.send(nb_sim_per_proc[i], dest=i, tag=10)
                    comm.send(subbatch_to_simulate.dvec[np.sum(nb_sim_per_proc[:i]):np.sum(nb_sim_per_proc[:i+1])], dest=i, tag=11)
                subbatch_to_simulate.obj_vals = np.zeros((np.sum(nb_sim_per_proc),))
                subbatch_to_simulate.obj_vals[0:nb_sim_per_proc[0]] = p.perform_real_evaluation(subbatch_to_simulate.dvec[0:nb_sim_per_proc[0]])
                for i in range(1,nprocs): # receiving from workers
                    subbatch_to_simulate.obj_vals[np.sum(nb_sim_per_proc[:i]):np.sum(nb_sim_per_proc[:i+1])] = comm.recv(source=i, tag=12)
                subbatch_to_simulate.dvec = subbatch_to_simulate.dvec[:np.sum(nb_sim_per_proc)]
                subbatch_to_simulate.fitness_modes = np.ones(subbatch_to_simulate.obj_vals.shape, dtype=bool)
                subbatch_to_simulate.save_sim_archive(F_SIM_ARCHIVE) # logging
                subbatch_to_simulate.update_best_sim(F_BEST_PROFILE)

                # Update active EC in Adaptive_EC
                if isinstance(ec_op, Adaptive_EC):
                    norm_pred_obj_vals = surr.perform_prediction(subbatch_to_simulate.dvec)[0]
                    norm_subbatch_to_simulate_obj_vals = surr.normalize_obj_vals(subbatch_to_simulate.obj_vals)
                    norm_y_min = surr.normalize_obj_vals(Global_Var.obj_val_min)[0]                
                    ec_op.update_active(norm_subbatch_to_simulate_obj_vals, norm_pred_obj_vals, norm_y_min)

                # Surrogate update
                surr.perform_training()

                # Predictions
                subbatch_to_predict.obj_vals = surr.perform_prediction(subbatch_to_predict.dvec)[0]
                subbatch_to_predict.obj_vals = surr.denormalize_predictions(subbatch_to_predict.obj_vals)
                subbatch_to_predict.fitness_modes = False*np.ones(subbatch_to_predict.obj_vals.shape, dtype=bool)

                # Merging evaluated batches
                children.append(subbatch_to_simulate)
                children.append(subbatch_to_predict)
                del subbatch_to_simulate
                del subbatch_to_predict
            #------------End batches loop------------#

            # Replacement
            replace_op.perform_replacement(pop, children)
            assert p.is_feasible(pop.dvec)
            del children

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
