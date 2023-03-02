"""``SAEA_ME.py`` Script running the synchronous parallel Surrogate-Assisted Evolutionary Algorithm for Medium Scale Expensive multi-objective problems.

SAEA_ME is described in:
`X. Ruan, K. Li, B. Derbel, and A. Liefooghe. Surrogate assisted evolutionary algorithm for medium scale multi-objective optimisation problems. In Proceedings of the 2020 Genetic and Evolutionary Computation Conference, GECCO 2020, page560–568, New York, NY, USA, 2020. Association for Computing Machinery <https://hal.archives-ouvertes.fr/hal-02932303v1>`_

The dimensionality reduction technique proposed in the paper is not reproduced.

Execution on Linux:
  * To run sequentially: ``python ./SAEA_ME.py``
  * To run in parallel (in 4 computational units): ``mpiexec -n 4 python SAEA_ME.py``
  * To run in parallel (in 4 computational units) specifying the units in `./hosts.txt`: ``mpiexec --machinefile ./host.txt -n 4 python SAEA_ME.py``

Execution on Windows:
  * To run sequentially: ``python ./SAEA_ME.py``
  * To run in parallel (in 4 computational units): ``mpiexec /np 4 python SAEA_ME.py``
"""

import shutil
import sys
sys.path.append('../src')
import os
import time
import numpy as np
from mpi4py import MPI
import pygmo

from Problems.DTLZ import DTLZ
from Problems.DoE import DoE

from Evolution.Population import Population
from Evolution.Tournament_Position import Tournament_Position
from Evolution.SBX import SBX
from Evolution.Polynomial import Polynomial
from Evolution.Custom_Elitism import Custom_Elitism

from Surrogates.BNN_MCD import BNN_MCD
from Surrogates.GP_MO import GP_MO

from Evolution_Controls.MO_POV_LCB_IC import MO_POV_LCB_IC
from Evolution_Controls.MO_POV_LCB_EC import MO_POV_LCB_EC

from Global_Var import *


def main():
   
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    # Multi-objective Problem
    p = DTLZ(5,4,3)


    #--------------------------------#
    #-------------MASTER-------------#
    #--------------------------------#
    if rank==0:

        # Argument of the search
        TIME_BUDGET = 0
        SIM_TIME = 60
        q=4 # number of simulations per cycle (could be less for the last cycle according to time budget)
        N_CYCLES=2
        if TIME_BUDGET>0:
            assert TIME_BUDGET>SIM_TIME
            N_CYCLES=1000000000000
        INIT_DB_SIZE=105
        POP_SIZE=76
        N_GEN=100
        N_CHLD=POP_SIZE

        # Reference point for HV computation
        Global_Var.ref_point=np.array([100., 100., 100.])

        # Files
        DIR_STORAGE="outputs"
        F_SIM_ARCHIVE=DIR_STORAGE+"/sim_archive.csv"
        F_BEST_PROFILE=DIR_STORAGE+"/best_profile.csv"
        F_INIT_DB=DIR_STORAGE+"/init_db.csv"
        F_HYPERVOLUME=DIR_STORAGE+"/hypervolume.csv"
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
        db.obj_vals = np.zeros((db.dvec.shape[0],p.n_obj))
        db.obj_vals[0:nb_sim_per_proc[0]] = p.perform_real_evaluation(db.dvec[0:nb_sim_per_proc[0]])
        for i in range(1,nprocs): # receiving from workers
            db.obj_vals[np.sum(nb_sim_per_proc[:i]):np.sum(nb_sim_per_proc[:i+1])] = comm.recv(source=i, tag=12)
        db.fitness_modes = True*np.ones(db.obj_vals.shape, dtype=bool)

        # Logging
        db.save_to_csv_file(F_INIT_DB)
        db.save_sim_archive(F_SIM_ARCHIVE)
        db.update_best_sim(F_BEST_PROFILE, F_HYPERVOLUME)

        # Number of simulations per proc
        nb_sim_per_proc = (q//nprocs)*np.ones((nprocs,), dtype=int)
        for i in range(q%nprocs):
            nb_sim_per_proc[i+1]+=1

        # Start chronometer
        if TIME_BUDGET>0:
            t_start = time.time()

        # Creating surrogate
        # surr = BNN_MCD(F_SIM_ARCHIVE, p, float('inf'), F_TRAIN_LOG, F_TRAINED_MODEL, 5)
        surr = GP_MO(F_SIM_ARCHIVE, p, 72, F_TRAIN_LOG, F_TRAINED_MODEL, 'rbf')
        surr.perform_training()

        # Evolution Control / Infill Criterion
        ic_op = MO_POV_LCB_IC(surr)
        ec_op = MO_POV_LCB_EC(surr, q)
        
        # Operators
        select_op = Tournament_Position(2)
        crossover_op = SBX(0.9, 10)
        mutation_op = Polynomial(0.1, 50)
        replace_op = Custom_Elitism(ic_op)

        #----------------------Loop over Cycles----------------------#    
        for curr_cycle in range(N_CYCLES):
            print("\ncycle "+str(curr_cycle))

            # Population initialization
            pop = Population(p)
            pop.dvec = db.dvec[db.dvec.shape[0]-POP_SIZE:db.dvec.shape[0],:]
            pop.dvec = pop.dvec[ic_op.get_sorted_indexes(pop)]

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

            # Retaining the q best individual according to MO_POV_LCB_EC
            q_cands = Population(p)
            idx = ec_op.get_sorted_indexes(pop)
            q_cands.dvec = pop.dvec[idx[0:q]]
            del pop

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
            q_cands.obj_vals = np.zeros((q_cands.dvec.shape[0], p.n_obj))
            q_cands.obj_vals[0:nb_sim_per_proc[0]] = p.perform_real_evaluation(q_cands.dvec[0:nb_sim_per_proc[0]])
            for i in range(1,nprocs): # receiving from workers
                q_cands.obj_vals[np.sum(nb_sim_per_proc[:i]):np.sum(nb_sim_per_proc[:i+1])] = comm.recv(source=i, tag=12)
            q_cands.fitness_modes = False*np.ones(q_cands.obj_vals.shape, dtype=bool)
            q_cands.fitness_modes[0:np.sum(nb_sim_per_proc)] = True

            # New simulations added to database
            db.dvec = np.vstack((db.dvec, q_cands.dvec))
            db.obj_vals = np.vstack((db.obj_vals, q_cands.obj_vals))
            db.fitness_modes = np.vstack((db.fitness_modes, q_cands.fitness_modes))
        
            q_cands.save_sim_archive(F_SIM_ARCHIVE)
            db.update_best_sim(F_BEST_PROFILE, F_HYPERVOLUME)
            del q_cands

            # Exit loop over cycles if budget time exhausted
            if TIME_BUDGET>0:
                t_now = time.time()
                elapsed_time = (t_now-t_start)
                if TIME_BUDGET-elapsed_time<SIM_TIME:
                    break

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
