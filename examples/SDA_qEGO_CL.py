"""``SDA_qEGO_CL.py`` Script running a synchronous parallel Surrogate-Driven Algorithm similar to q-EGO with constant liar for single-objective optimization.

q-EGO with Constant Liar is described in:
`D. Ginsbourger, R. Le Riche, and L. Carraro. Kriging is well-suited to parallelize optimization. In Computational Intelligence in Expensive Optimization Problems. Springer, 2010,  pp. 131–162. <https://hal-emse.ccsd.cnrs.fr/emse-00436126>`_

Execution on Linux:
  * To run sequentially: ``python ./SDA_qEGO_CL.py``
  * To run in parallel (in 4 computational units): ``mpiexec -n 4 python SDA_qEGO_CL.py``
  * To run in parallel (in 4 computational units) specifying the units in `./hosts.txt`: ``mpiexec --machinefile ./host.txt -n 4 python SDA_qEGO_CL.py``

Execution on Windows:
  * To run sequentially: ``python ./SDA_qEGO_CL.py``
  * To run in parallel (in 4 computational units): ``mpiexec /np 4 python SDA_qEGO_CL.py``
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

        # Argument of the search
        TIME_BUDGET = 0
        SIM_TIME = 15
        q=2 # number of simulations per cycle (could be less for the last cycle according to time budget)
        N_CYCLES=2
        INIT_DB_SIZE=72
        if TIME_BUDGET>0:
            assert TIME_BUDGET>SIM_TIME
            N_CYCLES=1000000000000
        POP_SIZE=50
        N_GEN=100
        N_CHLD=POP_SIZE

        # Files
        DIR_STORAGE="outputs"
        F_SIM_ARCHIVE=DIR_STORAGE+"/sim_archive.csv"
        F_BEST_PROFILE=DIR_STORAGE+"/best_profile.csv"
        F_INIT_DB=DIR_STORAGE+"/init_db.csv"
        F_TMP_DB=DIR_STORAGE+"/tmp_db.csv"
        F_TRAIN_LOG=DIR_STORAGE+"/training_log.csv"
        F_TRAINED_MODEL=DIR_STORAGE+"/trained_model"
        shutil.rmtree(DIR_STORAGE, ignore_errors=True)
        os.makedirs(DIR_STORAGE, exist_ok=True)

        # Database initialization / Parallel DoE
        sampler = DoE(p)
        db = Population(p)
        db.dvec = sampler.latin_hypercube_sampling(INIT_DB_SIZE)
        nb_sim_per_proc = (INIT_DB_SIZE//nprocs)*np.ones((nprocs,), dtype=int) # number of simulations per proc
        for i in range(INIT_DB_SIZE%nprocs):
            nb_sim_per_proc[i+1]+=1
        for i in range(1,nprocs): # sending to workers
            comm.send(nb_sim_per_proc[i], dest=i, tag=10)
            comm.send(db.dvec[np.sum(nb_sim_per_proc[:i]):np.sum(nb_sim_per_proc[:i+1])], dest=i, tag=11)
        db.obj_vals = np.zeros((db.dvec.shape[0],))
        db.obj_vals[0:nb_sim_per_proc[0]] = p.perform_real_evaluation(db.dvec[0:nb_sim_per_proc[0]])
        for i in range(1,nprocs): # receiving from workers
            db.obj_vals[np.sum(nb_sim_per_proc[:i]):np.sum(nb_sim_per_proc[:i+1])] = comm.recv(source=i, tag=12)
        db.fitness_modes = True*np.ones(db.obj_vals.shape, dtype=bool)

        # Logging
        db.save_to_csv_file(F_INIT_DB)
        db.save_sim_archive(F_SIM_ARCHIVE)
        db.save_sim_archive(F_TMP_DB)
        db.update_best_sim(F_BEST_PROFILE)

        # Set the constant for Constant Liar
        # cl_mode = sys.argv[2]
        cl_mode = "cl-mean"
        if cl_mode=="cl-min":
            L = Global_Var.obj_val_min
        elif cl_mode=="cl-mean":
            L = [np.sum(db.obj_vals), db.obj_vals.shape[0]]
        elif cl_mode=="cl-max":
            L = np.max(db.obj_vals)
        else:
            print("[SDA_qEGO_CL.py] error invalid strategy name")
            assert False
        
        # Number of simulations per proc
        nb_sim_per_proc = (q//nprocs)*np.ones((nprocs,), dtype=int)
        for i in range(q%nprocs):
            nb_sim_per_proc[i+1]+=1

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
        #         ec_op = Dynamic_Exclusive_EC(N_CYCLES, [0.5, 0.5], ec_base_d, ec_base_y)
        # elif sys.argv[1]=="dyn-dpf-excl":
        #     if TIME_BUDGET>0:
        #         ec_tmp = Pareto_EC([1.0, 1.0], "cd", ec_base_y, ec_base_d)
        #         ec_op = Dynamic_Exclusive_EC(TIME_BUDGET, [0.25, 0.5, 0.25], ec_base_d, ec_tmp, ec_base_y)
        #     else:
        #         ec_tmp = Pareto_EC([1.0, 1.0], "cd", ec_base_y, ec_base_d)
        #         ec_op = Dynamic_Exclusive_EC(N_CYCLES, [0.25, 0.5, 0.25], ec_base_d, ec_tmp, ec_base_y)
        # elif sys.argv[1]=="dyn-df-incl":
        #     if TIME_BUDGET>0:
        #         ec_op = Dynamic_Inclusive_EC(TIME_BUDGET, q, (POP_SIZE-q), ec_base_d, ec_base_y)
        #     else:
        #         ec_op = Dynamic_Inclusive_EC(N_CYCLES, q, (POP_SIZE-q), ec_base_d, ec_base_y)
        # elif sys.argv[1]=="ada-dpf":
        #     norm_pred_obj_vals = surr.perform_prediction(db.dvec)[0]
        #     norm_pop_obj_vals = surr.normalize_obj_vals(db.obj_vals)
        #     norm_y_min = surr.normalize_obj_vals(Global_Var.obj_val_min)[0]
        #     ec_tmp = Pareto_EC([1.0, 1.0], "cd", ec_base_y, ec_base_d)
        #     ec_op = Adaptive_EC(norm_pop_obj_vals, norm_pred_obj_vals, norm_y_min, ec_base_d, ec_tmp, ec_base_y)
        # elif sys.argv[1]=="ada-wang":
        #     if TIME_BUDGET>0:
        #         ec_op = Adaptive_Wang2020_EC(surr, TIME_BUDGET, "max")
        #     else:
        #         ec_op = Adaptive_Wang2020_EC(surr, N_CYCLES, "max")
        # elif sys.argv[1]=="com-dpf":
        #     ec_tmp = Pareto_EC([1.0, 1.0], "cd", ec_base_y, ec_base_d)
        #     ec_op = Committee_EC(POP_SIZE, ec_base_d, ec_tmp, ec_base_y)
        # else:
        #     print("[SDA_qEGO_CL.py] error invalid ec name")
        #     assert False

        # Operators
        select_op = Tournament_Position(2)
        crossover_op = SBX(0.9, 10)
        mutation_op = Polynomial(1.0/p.n_dvar, 50)
        replace_op = Custom_Elitism(ec_op)
        
        del db

        #----------------------Loop over Cycles----------------------#    
        for curr_cycle in range(N_CYCLES):
            print("\ncycle "+str(curr_cycle))

            q_cands = Population(p)
            q_cands.dvec = np.zeros((q,p.n_dvar))

            #----------------------Loop over the q sub-cycles----------------------#    
            for curr_sub_cycle in range(q):
                print("    sub-cycle "+str(curr_sub_cycle))

                # Population initialization
                pop = Population(p)
                pop.dvec = sampler.latin_hypercube_sampling(POP_SIZE)
                pop.dvec = pop.dvec[ec_op.get_sorted_indexes(pop)]

                #----------------------Evolution loop----------------------#
                for curr_gen in range(N_GEN):
                    # print("generation "+str(curr_gen))

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

                # Retaining the best individual
                pop.obj_vals = np.zeros((pop.dvec.shape[0],))
                if cl_mode=="cl-mean":
                    pop.obj_vals[0:1] = L[0]/L[1]
                elif cl_mode=="cl-min" or cl_mode=="cl-max":
                    pop.obj_vals[0:1] = L
                else:
                    print("[SDA_qEGO_CL.py] error invalid strategy name")
                    assert False
                pop.fitness_modes = False*np.ones(pop.obj_vals.shape, dtype=bool)
                pop.fitness_modes[0:1] = True
                pop.save_sim_archive(F_TMP_DB)
                q_cands.dvec[curr_sub_cycle,:] = pop.dvec[0,:]

                # Exit loop over the q sub-cycles if budget time exhausted
                if TIME_BUDGET>0:
                    t_now = time.time()
                    if TIME_BUDGET-(t_now-t_start)<SIM_TIME:
                        break

                # Surrogate partial update
                if curr_sub_cycle!=q-1:
                    if isinstance(surr, iKRG):
                        surr.add_point(pop.dvec[0,:], pop.obj_vals[0:1])
                    if isinstance(surr, rKRG):
                        surr.add_point(pop.dvec[0,:], pop.obj_vals[0:1])
                    elif isinstance(surr, GP):
                        surr.perform_partial_training()
                    else:
                        surr.perform_training() # training on the liar(s)

                del pop
            #----------------------End loop over the q sub-cycles----------------------#

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

            # q(=np.sum(np_sim_per_proc)) best candidates from the population are simulated in parallel
            for i in range(1,nprocs): # sending to workers
                comm.send(nb_sim_per_proc[i], dest=i, tag=10)
                comm.send(q_cands.dvec[np.sum(nb_sim_per_proc[:i]):np.sum(nb_sim_per_proc[:i+1])], dest=i, tag=11)
            q_cands.obj_vals = np.zeros((q_cands.dvec.shape[0],))
            q_cands.obj_vals[0:nb_sim_per_proc[0]] = p.perform_real_evaluation(q_cands.dvec[0:nb_sim_per_proc[0]])
            for i in range(1,nprocs): # receiving from workers
                q_cands.obj_vals[np.sum(nb_sim_per_proc[:i]):np.sum(nb_sim_per_proc[:i+1])] = comm.recv(source=i, tag=12)
            q_cands.fitness_modes = False*np.ones(q_cands.obj_vals.shape, dtype=bool)
            q_cands.fitness_modes[0:np.sum(nb_sim_per_proc)] = True

            # New simulations added to database
            q_cands.save_sim_archive(F_SIM_ARCHIVE)
            q_cands.update_best_sim(F_BEST_PROFILE)
            # Temporary database is re-created
            shutil.copy(F_SIM_ARCHIVE, F_TMP_DB)

            # Update active EC in Adaptive_EC
            if isinstance(ec_op, Adaptive_EC):
                norm_pred_obj_vals = surr.perform_prediction(q_cands.dvec)[0]
                norm_subbatch_to_simulate_obj_vals = surr.normalize_obj_vals(q_cands.obj_vals)
                norm_y_min = surr.normalize_obj_vals(Global_Var.obj_val_min)[0]                
                ec_op.update_active(norm_subbatch_to_simulate_obj_vals, norm_pred_obj_vals, norm_y_min)

            # Constant in Constant Liar is updated
            if cl_mode=="cl-min":
                L = Global_Var.obj_val_min
            elif cl_mode=="cl-max":
                L = max(np.max(q_cands.obj_vals[0:np.sum(nb_sim_per_proc)]), L)
            elif cl_mode=="cl-mean":
                L[0] += np.sum(q_cands.obj_vals[0:np.sum(nb_sim_per_proc)])
                L[1] += np.sum(nb_sim_per_proc)

            del q_cands

            # Exit loop over cycles if budget time exhausted
            if TIME_BUDGET>0:
                t_now = time.time()
                elapsed_time = (t_now-t_start)
            
                # Update active EC in Dynamic_EC (cycle level) or Adaptive_Wang2020_EC
                if isinstance(ec_op, Dynamic_Exclusive_EC) or isinstance(ec_op, Dynamic_Inclusive_EC):
                    ec_op.update_active(elapsed_time)
                if isinstance(ec_op, Adaptive_Wang2020_EC):
                    ec_op.update_EC(elapsed_time)

                if TIME_BUDGET-elapsed_time<SIM_TIME:
                    break
            else:
                # Update active EC in Dynamic_EC (cycle level) or Adaptive_Wang2020_EC
                if isinstance(ec_op, Dynamic_Exclusive_EC) or isinstance(ec_op, Dynamic_Inclusive_EC):
                    ec_op.update_active(curr_cycle)
                if isinstance(ec_op, Adaptive_Wang2020_EC):
                    ec_op.update_EC(curr_cycle)

            # Surrogate update
            if curr_cycle!=N_CYCLES-1:
                surr.perform_training() # (with simulated training samples only)
        
        #----------------------End loop over Cycles----------------------#

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
