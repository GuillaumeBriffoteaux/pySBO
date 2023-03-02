"""``HSAP.py`` Script running the parallel Hybrid Successive Acquition Process for single-objective optimization.

The Hybrid Successive Acquition Process is described in:
`G. Briffoteaux, N. Melab, M. Mezmaz et D. Tuyttens. Hybrid Acquisition Processes in Surrogate-based Optimization. Application to Covid-19 Contact Reduction. International Conference on Bioinspired Optimisation Methods and Their Applications, BIOMA, 2022, Maribor, Slovenia, Lecture Notes in Computer Science, vol 13627. Springer, pages 127-141 <https://doi.org/10.1007/978-3-031-21094-5_10>`_

You must set the SIM_TIME variable to a non-zero positive value. This represents the evaluation time of the objective function (which is fictitious in case of artificial benchmark functions).

Execution on Linux:
  * To run sequentially: ``python ./HSAP.py``
  * To run in parallel (in 4 computational units): ``mpiexec -n 4 python HSAP.py``
  * To run in parallel (in 4 computational units) specifying the units in `./hosts.txt`: ``mpiexec --machinefile ./host.txt -n 4 python HSAP.py``

Execution on Windows:
  * To run sequentially: ``python ./HSAP.py``
  * To run in parallel (in 4 computational units): ``mpiexec /np 4 python HSAP.py``
"""

import shutil
import sys
sys.path.append('../src')
import os
import time
import numpy as np
from mpi4py import MPI
from sklearn.cluster import KMeans

from Problems.Schwefel import Schwefel
from Problems.DoE import DoE

from Evolution.Population import Population
from Evolution.Tournament import Tournament
from Evolution.Tournament_Position import Tournament_Position
from Evolution.SBX import SBX
from Evolution.Polynomial import Polynomial
from Evolution.Elitist import Elitist
from Evolution.Custom_Elitism import Custom_Elitism

from Surrogates.BNN_MCD import BNN_MCD
from Surrogates.GP import GP

from Evolution_Controls.Expected_Improvement_EC import Expected_Improvement_EC
from Evolution_Controls.POV_EC import POV_EC
from Evolution_Controls.Distance_EC import Distance_EC
from Evolution_Controls.Dynamic_Inclusive_EC import Dynamic_Inclusive_EC

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
    
        # Budget
        TIME_BUDGET = 300
        SIM_TIME = 15
        assert TIME_BUDGET > 0
        assert TIME_BUDGET > SIM_TIME
        N_SIM_REALIZED=0
        INIT_DB_SIZE = 36

        # Files
        DIR_STORAGE = "outputs"
        shutil.rmtree(DIR_STORAGE, ignore_errors=True)
        os.makedirs(DIR_STORAGE, exist_ok=True)
        F_SIM_ARCHIVE = DIR_STORAGE+"/sim_archive.csv"
        F_TRAIN_LOG_BNN = DIR_STORAGE+"/training_log_BNN.csv"
        F_TRAIN_LOG_GP = DIR_STORAGE+"/training_log_GP.csv"
        F_TRAINED_MODEL_BNN = DIR_STORAGE+"/trained_model_BNN"
        F_TRAINED_MODEL_GP = DIR_STORAGE+"/trained_model_GP"
        F_TMP_DB=DIR_STORAGE+"/tmp_db.csv"
        F_BEST_PROFILE = DIR_STORAGE+"/best_profile.csv"
        F_INIT_DB = DIR_STORAGE+"/init_db.csv"

        # Parameter for q-EGO
        q=18
        POP_SIZE=50
        N_GEN=100
        N_CHLD=POP_SIZE

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
        db.save_sim_archive(F_TMP_DB)

        # Sim per proc for q-EGO
        nb_sim_per_proc = (q//nprocs)*np.ones((nprocs,), dtype=int)
        for i in range(q%nprocs):
            nb_sim_per_proc[i+1]+=1

        # Set the constant for cl-mean
        L = [np.sum(db.obj_vals), db.obj_vals.shape[0]]

        # Start chronometer
        if TIME_BUDGET>0:
            t_start = time.time()

        # Surrogate for q-EGO
        surr = GP(F_TMP_DB, p, float('inf'), F_TRAIN_LOG_GP, F_TRAINED_MODEL_GP, "rbf")
        surr.perform_training()

        # Evolution Control for q-EGO
        ec_op = Expected_Improvement_EC(surr)

        # EA operators
        select_op = Tournament_Position(2)
        crossover_op = SBX(0.9, 10)
        mutation_op = Polynomial(1.0/p.n_dvar, 50)
        replace_op = Custom_Elitism(ec_op)

    
        #----------------------q-EGO----------------------#
        counter=0
        switch=False
        while switch is False:
            print("q-EGO cycle "+str(counter))
        
            q_cands = Population(p)
            q_cands.dvec = np.zeros((q,p.n_dvar))

            #----------------------q-loop
            for curr_sub_cycle in range(q):

                # Population initialization
                pop = Population(p)
                pop.dvec = sampler.latin_hypercube_sampling(POP_SIZE)
                pop.dvec = pop.dvec[ec_op.get_sorted_indexes(pop)]

                #----------------------Evolution loop
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
                #----------------------End evolution loop

                # Retaining the best individual
                pop.obj_vals = np.zeros((pop.dvec.shape[0],))
                pop.obj_vals[0:1] = L[0]/L[1]
                pop.fitness_modes = False*np.ones(pop.obj_vals.shape, dtype=bool)
                pop.fitness_modes[0:1] = True
                pop.save_sim_archive(F_TMP_DB)
                q_cands.dvec[curr_sub_cycle,:] = pop.dvec[0,:]

                # Surrogate partial update
                if curr_sub_cycle!=q-1:
                    surr.perform_partial_training()

            #----------------------End q-loop

            # Parallel simulations
            for i in range(1,nprocs): # sending to workers
                comm.send(nb_sim_per_proc[i], dest=i, tag=10)
                comm.send(q_cands.dvec[np.sum(nb_sim_per_proc[:i]):np.sum(nb_sim_per_proc[:i+1])], dest=i, tag=11)
            q_cands.obj_vals = np.zeros((np.sum(nb_sim_per_proc),))
            q_cands.obj_vals[0:nb_sim_per_proc[0]] = p.perform_real_evaluation(q_cands.dvec[0:nb_sim_per_proc[0]])
            for i in range(1,nprocs): # receiving from workers
                q_cands.obj_vals[np.sum(nb_sim_per_proc[:i]):np.sum(nb_sim_per_proc[:i+1])] = comm.recv(source=i, tag=12)
            q_cands.dvec = q_cands.dvec[:np.sum(nb_sim_per_proc)]
            q_cands.fitness_modes = np.ones(q_cands.obj_vals.shape, dtype=bool)
            N_SIM_REALIZED=N_SIM_REALIZED+q

            # Updating db
            db.obj_vals = np.append(db.obj_vals, q_cands.obj_vals)
            db.dvec = np.vstack((db.dvec, q_cands.dvec))
            db.fitness_modes = np.append(db.fitness_modes, q_cands.fitness_modes)

            # Logging
            q_cands.save_sim_archive(F_SIM_ARCHIVE) 
            q_cands.update_best_sim(F_BEST_PROFILE)
            shutil.copy(F_SIM_ARCHIVE, F_TMP_DB)
            del q_cands
        
            # Check for AP switch
            if N_SIM_REALIZED>=100:
                switch=True
            else:
                # Surrogate update
                surr.perform_training()

            counter = counter+1
        #----------------------End q-EGO----------------------#



    
        # Timing
        T_START_PSAEA=time.time()
    
        # Parameters for P-SAEA
        POP_SIZE = 72
        N_CHLD = 288
        N_SIM = 72
        N_DISC = 216
        N_GEN = 60 # upper bound
        if N_CHLD!=N_SIM+N_DISC:
            print("[HSAP.py] number of children in SAEA does not match number of simulations and discardings")
            assert False        

        # Number of simulations per proc
        nb_sim_per_proc = (N_SIM//nprocs)*np.ones((nprocs,), dtype=int)
        for i in range(N_SIM%nprocs):
            nb_sim_per_proc[i+1]+=1

        # Population initialization
        pop = Population(p)
        db.sort() # sort db and retain the 10 best candidates
        pop.obj_vals = db.obj_vals[:10]
        pop.dvec = db.dvec[:10]
        pop.fitness_modes = db.fitness_modes[:10]
        n_clusters=62 # clustering the database
        max_iter=300
        clust = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=max_iter)
        clust.fit(db.dvec)
        for i in range(n_clusters): # random sampling of 1 candidate per cluster
            idxs = np.random.choice(np.where(clust.labels_==i)[0], size=1, replace=False)
            for j in idxs:
                pop.obj_vals = np.append(pop.obj_vals, db.obj_vals[j])
                pop.dvec = np.vstack((pop.dvec, db.dvec[j]))
                pop.fitness_modes = np.append(pop.fitness_modes, db.fitness_modes[j])
        del db
        del clust

        # Surrogate
        surr = BNN_MCD(F_SIM_ARCHIVE, p, float('inf'), F_TRAIN_LOG_BNN, F_TRAINED_MODEL_BNN, 5)
        surr.perform_training()
    
        # Evolution Controls                                                                      
        ec_base_y = POV_EC(surr)
        ec_base_d = Distance_EC(surr)
        ec_op = Dynamic_Inclusive_EC(int(TIME_BUDGET-T_START_PSAEA), N_SIM, 0, ec_base_d, ec_base_y)
        
        # Operators
        select_op = Tournament(2)
        crossover_op = SBX(0.9, 10)
        mutation_op = Polynomial(1.0/p.n_dvar, 50)
        replace_op = Elitist()


        #----------------------Start Generation loop----------------------#
        for curr_gen in range(N_GEN):
            print("P-SAEA generation "+str(curr_gen))

            # Reproduction operators
            parents = select_op.perform_selection(pop, N_CHLD)
            children = crossover_op.perform_crossover(parents)
            children = mutation_op.perform_mutation(children)
            assert p.is_feasible(children.dvec)

            # Evolution Control
            idx_split = ec_op.get_sorted_indexes(children)
            batch_to_simulate = Population(p)
            batch_to_simulate.dvec = children.dvec[idx_split[0:N_SIM]]

            # Update active EC in Dynamic_EC (batch level)
            if TIME_BUDGET>0:
                if isinstance(ec_op, Dynamic_Inclusive_EC):
                    t_now = time.time()
                    elapsed_time = (t_now-T_START_PSAEA)
                    ec_op.update_active(elapsed_time)

            # Parallel simulations
            for i in range(1,nprocs): # sending to workers
                comm.send(nb_sim_per_proc[i], dest=i, tag=10)
                comm.send(batch_to_simulate.dvec[np.sum(nb_sim_per_proc[:i]):np.sum(nb_sim_per_proc[:i+1])], dest=i, tag=11)
            batch_to_simulate.obj_vals = np.zeros((np.sum(nb_sim_per_proc),))
            batch_to_simulate.obj_vals[0:nb_sim_per_proc[0]] = p.perform_real_evaluation(batch_to_simulate.dvec[0:nb_sim_per_proc[0]])
            for i in range(1,nprocs): # receiving from workers
                batch_to_simulate.obj_vals[np.sum(nb_sim_per_proc[:i]):np.sum(nb_sim_per_proc[:i+1])] = comm.recv(source=i, tag=12)
            batch_to_simulate.dvec = batch_to_simulate.dvec[:np.sum(nb_sim_per_proc)]
            batch_to_simulate.fitness_modes = np.ones(batch_to_simulate.obj_vals.shape, dtype=bool)

            # Logging
            batch_to_simulate.save_sim_archive(F_SIM_ARCHIVE) 
            batch_to_simulate.update_best_sim(F_BEST_PROFILE)

            # Exit Generation loop if budget time exhausted
            if TIME_BUDGET>0:
                t_now = time.time()
                if TIME_BUDGET-(t_now-t_start)<SIM_TIME:
                    break

            # Surrogate update
            surr.perform_training()

            # Replacement
            replace_op.perform_replacement(pop, batch_to_simulate)
            assert p.is_feasible(pop.dvec)
            del batch_to_simulate
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
