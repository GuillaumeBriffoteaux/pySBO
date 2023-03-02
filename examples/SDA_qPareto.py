"""``SDA_qPareto.py`` Script running a synchronous parallel Surrogate-Driven Algorithm based on a Pareto acquisition process for single-objective optimization.

The Pareto acquisition process is described in:
`Z. Feng, Q. Zhang, Q. Zhang, Q. Tang, T. Yang and Y. Ma. A multi-objective optimization based framework to balance the global exploration and local exploitation in expensive optimization. In Journal of Global Optimization 61.4 (Apr. 2015), pp. 677–694. <https://doi.org/10.1007/s10898-014-0210-2>`_

Execution on Linux:
  * To run sequentially: ``python ./SDA_qPareto.py``
  * To run in parallel (in 4 computational units): ``mpiexec -n 4 python SDA_qPareto.py``
  * To run in parallel (in 4 computational units) specifying the units in `./hosts.txt`: ``mpiexec --machinefile ./host.txt -n 4 python SDA_qPareto.py``

Execution on Windows:
  * To run sequentially: ``python ./SDA_qPareto.py``
  * To run in parallel (in 4 computational units): ``mpiexec /np 4 python SDA_qPareto.py``
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

from Evolution_Controls.POV_EC import POV_EC
from Evolution_Controls.Distance_EC import Distance_EC
from Evolution_Controls.Pred_Stdev_EC import Pred_Stdev_EC
from Evolution_Controls.Pareto_EC import Pareto_EC
from Evolution_Controls.Pareto_Tian2018_EC import Pareto_Tian2018_EC

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
        POP_SIZE=150
        assert q<=POP_SIZE
        N_GEN=50
        N_CHLD=POP_SIZE

        # Files
        DIR_STORAGE="outputs"
        F_SIM_ARCHIVE=DIR_STORAGE+"/sim_archive.csv"
        F_BEST_PROFILE=DIR_STORAGE+"/best_profile.csv"
        F_INIT_DB=DIR_STORAGE+"/init_db.csv"
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
        db.update_best_sim(F_BEST_PROFILE)
        
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
        ec_base_y = POV_EC(surr)
        ec_base_d = Distance_EC(surr)
        ec_base_s = Pred_Stdev_EC(surr)
        ec_op = Pareto_EC([1.0, 1.0], "cd", ec_base_y, ec_base_d)
        # if sys.argv[1]=="par-fd-cd":
        #     ec_op = Pareto_EC([1.0, 1.0], "cd", ec_base_y, ec_base_d)
        # elif sys.argv[1]=="par-fs-hvc":
        #     ec_op = Pareto_EC([1.0, 1.0], "hvc", ec_base_y, ec_base_s)
        # elif sys.argv[1]=="par-tian":
        #     ec_op = Pareto_Tian2018_EC([1.0, -1.0], ec_base_y, ec_base_s)
        # else:
        #     print("[SDA_qPareto.py] error invalid ec name")
        #     assert False

        # Operators
        select_op = Tournament_Position(2)
        crossover_op = SBX(0.9, 10)
        mutation_op = Polynomial(1.0/p.n_dvar, 50)
        replace_op = Custom_Elitism(ec_op)

        del db


        #----------------------Loop over Cycles----------------------#    
        for curr_cycle in range(N_CYCLES):
            print("cycle "+str(curr_cycle))

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
                comm.send(pop.dvec[np.sum(nb_sim_per_proc[:i]):np.sum(nb_sim_per_proc[:i+1])], dest=i, tag=11)
            pop.obj_vals = np.zeros((pop.dvec.shape[0],))
            pop.obj_vals[0:nb_sim_per_proc[0]] = p.perform_real_evaluation(pop.dvec[0:nb_sim_per_proc[0]])
            for i in range(1,nprocs): # receiving from workers
                pop.obj_vals[np.sum(nb_sim_per_proc[:i]):np.sum(nb_sim_per_proc[:i+1])] = comm.recv(source=i, tag=12)
            pop.fitness_modes = False*np.ones(pop.obj_vals.shape, dtype=bool)
            pop.fitness_modes[0:np.sum(nb_sim_per_proc)] = True

            # New simulations added to database
            pop.save_sim_archive(F_SIM_ARCHIVE)
            pop.update_best_sim(F_BEST_PROFILE)
            del pop

            # Exit loop over cycles if budget time exhausted
            if TIME_BUDGET>0:
                t_now = time.time()
                if TIME_BUDGET-(t_now-t_start)<SIM_TIME:
                    break

            # Surrogate update (with simulated training samples only)
            if curr_cycle!=N_CYCLES-1:
                surr.perform_training()        
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
