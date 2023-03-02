"""``AB_MOEA.py`` Script running the synchronous parallel surrogate-based Adaptive Bayesian Multi-Objective Evolutionary Algorithm.

AB_MOEA is described in:
`X. Wang, Y. Jin, S. Schmitt and M. Olhofer. An adaptive Bayesian approach to surrogate-assisted evolutionary multi-objective optimization. In Information Sciences 519 (2020), pp. 317–331. ISSN: 0020-0255. <https://doi.org/10.1016/j.ins.2020.01.048>`_

Execution on Linux:
  * To run sequentially: ``python ./AB_MOEA.py``
  * To run in parallel (in 4 computational units): ``mpiexec -n 4 python AB_MOEA.py``
  * To run in parallel (in 4 computational units) specifying the units in `./hosts.txt`: ``mpiexec --machinefile ./host.txt -n 4 python AB_MOEA.py``

Execution on Windows:
  * To run sequentially: ``python ./AB_MOEA.py``
  * To run in parallel (in 4 computational units): ``mpiexec /np 4 python AB_MOEA.py``
"""

import shutil
import sys
sys.path.append('../src/')
import os
import time
import math
import numpy as np
import itertools
from mpi4py import MPI
from scipy.special import comb
import scipy

from Problems.DTLZ import DTLZ
from Problems.DoE import DoE

from Evolution.Reference_Vector_Set import Reference_Vector_Set
from Evolution.Population import Population
from Evolution.Tournament_Position import Tournament_Position
from Evolution.SBX import SBX
from Evolution.Polynomial import Polynomial

from Surrogates.BNN_MCD import BNN_MCD
from Surrogates.GP_MO import GP_MO

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

        # Arguments of the search
        TIME_BUDGET = 0
        SIM_TIME = 60
        u = 4 # number of simulations per cycle (could be less for the last cycle according to time budget)
        N_CYCLES=2
        if TIME_BUDGET>0:
            assert TIME_BUDGET>SIM_TIME
            N_CYCLES=1000000000000
        INIT_DB_SIZE=105

        # Reference point for HV computation
        Global_Var.ref_point=np.array([100., 100., 100.])

        # Argument for RVEA
        H = 13 # simplex lattice parameter
        POP_SIZE = int(comb(H+p.n_obj-1, p.n_obj-1))
        N_CHLD = 2*math.floor(POP_SIZE/2)
        N_GEN = 1 # 20
        F_UPD = 0.1 # frequency for reference vector update

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
        nb_sim_per_proc = (u//nprocs)*np.ones((nprocs,), dtype=int)
        for i in range(u%nprocs):
            nb_sim_per_proc[i+1]+=1

        # Start chronometer
        if TIME_BUDGET>0:
            t_start = time.time()

        # Creating surrogate
        # surr = BNN_MCD(F_SIM_ARCHIVE, p, float('inf'), F_TRAIN_LOG, F_TRAINED_MODEL, 5)
        surr = GP_MO(F_SIM_ARCHIVE, p, 72, F_TRAIN_LOG, F_TRAINED_MODEL, 'rbf')
        surr.perform_training()
        
        # Operators
        select_op = Tournament_Position(2)
        crossover_op = SBX(0.9, 10)
        mutation_op = Polynomial(0.1, 50)

        #----------------------Loop over Cycles----------------------#    
        for curr_cycle in range(N_CYCLES):
            print("\ncycle "+str(curr_cycle))

            # Population initialization
            pop = Population(p)
            pop.dvec = db.dvec[db.dvec.shape[0]-POP_SIZE:db.dvec.shape[0],:]
            pop.obj_vals = surr.perform_prediction(pop.dvec)[0]
            # pop.obj_vals = surr.denormalize_predictions(pop.obj_vals)
            pop.fitness_modes = False*np.ones((pop.obj_vals.shape[0],p.n_obj), dtype=bool)
        
            # Reference vectors initialization
            V = Reference_Vector_Set(H, p)
            V_init = Reference_Vector_Set(H, p)

            #----------------------Start Generation loop----------------------#
            for curr_gen in range(N_GEN):            
                # print("generation "+str(curr_gen))

                # Acquisition Process
                parents = select_op.perform_selection(pop, N_CHLD)
                children = crossover_op.perform_crossover(parents)
                children = mutation_op.perform_mutation(children)
                assert p.is_feasible(children.dvec)

                # Predictions
                children.obj_vals = np.zeros((children.dvec.shape[0],children.pb.n_obj))
                children.obj_vals = surr.perform_prediction(children.dvec)[0]
                # children.obj_vals = surr.denormalize_predictions(children.obj_vals)
                children.fitness_modes = False*np.ones(children.obj_vals.shape, dtype=bool)
                pop.append(children)
                del children

                # Reference Vector guided Replacement
                pop = V.reference_vector_guided_replacement(pop, curr_gen, N_GEN)
                assert p.is_feasible(pop.dvec)

                # Reference Vector update
                if int(N_GEN*F_UPD)>0:
                    if curr_gen%int(N_GEN*F_UPD)==0 and curr_gen!=N_GEN-1:
                        V.rv = V_init.reference_vector_update(pop)
            #----------------------End Generation loop----------------------#

            # Exit loop if budget time exhausted
            if TIME_BUDGET>0:
                t_now = time.time()
                if TIME_BUDGET-(t_now-t_start)<SIM_TIME:
                    break



                
            #-------------------Adaptive acquisition function
            if TIME_BUDGET==0:
                alpha = -0.5*np.cos(np.pi*curr_cycle/N_CYCLES)+0.5
            else:
                t_now = time.time()
                elapsed_time = (t_now-t_start)
                alpha = -0.5*np.cos(np.pi*elapsed_time/TIME_BUDGET)+0.5

            preds_max = np.amax(pop.obj_vals, axis=0)
            stdevs = surr.perform_prediction(pop.dvec)[1]            
            variances = np.power(stdevs, 2)
            variances_max = np.amax(variances, axis=0)
            pop.obj_vals = (1-alpha)*np.divide(pop.obj_vals, preds_max) + alpha*np.divide(variances, variances_max)
            
            #-------------------Adaptive Sampling Selection Criterion

            # Translate the objective values
            z_min = np.amin(pop.obj_vals, axis=0)
            pop.obj_vals = pop.obj_vals - z_min
            
            # Angle from each reference vector to the remaining reference vectors
            theta_rv = np.array([])
            for (rv1, rv2) in itertools.permutations(V.rv, 2):
                theta_rv = np.append(theta_rv, np.arccos(np.dot(rv1,rv2)))
            theta_rv = np.reshape(theta_rv, (V.rv.shape[0], V.rv.shape[0]-1))

            # Minimum angle from each reference vector to the remaining reference vectors
            min_theta_rv = np.amin(theta_rv, axis=1)

            # Compute the cosine angles
            cos_theta = np.array([]) # first index: translated objective vector | second index: reference vector
            for (pc, rv) in itertools.product(pop.obj_vals, V.rv):
                cos_theta = np.append(cos_theta, np.dot(pc, rv)/scipy.linalg.norm(pc))
            cos_theta = np.reshape(cos_theta, (pop.obj_vals.shape[0], V.rv.shape[0]))

            # Compute the metric used to order the new candidates according to decreasing promise
            Angle_metric = np.empty(())
            if alpha<0.5:            
                # Compute the angles
                Angles = np.array([])
                idx_subpop = -1*np.ones((pop.obj_vals.shape[0],), dtype=int) # contains the index of the sub-population for each objective vector
                for i in range(pop.obj_vals.shape[0]):
                    idx_subpop[i] = np.argmax(cos_theta[i,:])
                    theta = np.arccos(cos_theta[i, idx_subpop[i]])
                    Angles = np.append(Angles, p.n_obj*theta/min_theta_rv[idx_subpop[i]])
                Angle_metric = Angles
            else:
                # Compute the APD
                APD = np.array([])
                if TIME_BUDGET==0:
                    budget_coeff = pow(float(curr_cycle)/float(N_CYCLES),2)
                else:
                    t_now = time.time()
                    elapsed_time = (t_now-t_start)
                    budget_coeff = pow(float(elapsed_time)/float(TIME_BUDGET),2)
                idx_subpop = -1*np.ones((pop.obj_vals.shape[0],), dtype=int) # contains the index of the sub-population for each objective vector
                for i in range(pop.obj_vals.shape[0]):
                    idx_subpop[i] = np.argmax(cos_theta[i,:])
                    theta = np.arccos(cos_theta[i, idx_subpop[i]])
                    APD = np.append( APD, ( 1.0 + p.n_obj * budget_coeff * (theta/min_theta_rv[idx_subpop[i]]) ) * scipy.linalg.norm(pop.obj_vals[i]) )
                Angle_metric = APD

            # Order the objective vectors: first the lowest Angle_metric for each sub-population, second the lowest Angle_metric among the remaining objective vectors
            idx_pop_obj_vals = np.array([i for i in range(pop.obj_vals.shape[0])])
            
            # Ordering according to increasing Angle_metric
            idx_sorted_angles = np.argsort(Angle_metric)
            Angle_metric = np.sort(Angle_metric)
            idx_subpop = idx_subpop[idx_sorted_angles]
            idx_pop_obj_vals = idx_pop_obj_vals[idx_sorted_angles]            
            del idx_sorted_angles
            
            # Get the index corresponding to lowest Angle_metric for each subpopulation
            unique_idx_subpop = np.unique(idx_subpop)
            idx_tmp = np.array([], dtype=int)
            for i,idx in enumerate(idx_subpop):
                if idx in unique_idx_subpop:
                    idx_tmp = np.append(idx_tmp, i)
                    unique_idx_subpop = unique_idx_subpop[unique_idx_subpop!=idx]
            del unique_idx_subpop
            
            # idx_new_sort contains the indexes of the objective vectors ordered in decreasing promise
            idx_new_sort = idx_pop_obj_vals[idx_tmp]
            Angle_metric = np.delete(Angle_metric, idx_tmp)
            idx_pop_obj_vals = np.delete(idx_pop_obj_vals, idx_tmp)
            idx_subpop = np.delete(idx_subpop, idx_tmp)
            idx_new_sort = np.append(idx_new_sort, idx_pop_obj_vals)
            
            del Angle_metric
            del idx_pop_obj_vals
            del idx_subpop




            #-------------------Parallel simulations and surrogate update

            batch_to_simulate = Population(p)
            batch_to_simulate.dvec = pop.dvec[idx_new_sort[0:u]]

            # Number of simulations per proc
            nb_sim_per_proc = (batch_to_simulate.dvec.shape[0]//nprocs)*np.ones((nprocs,), dtype=int)
            for i in range(batch_to_simulate.dvec.shape[0]%nprocs):
                nb_sim_per_proc[i+1]+=1

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

            # Parallel simulations
            for i in range(1,nprocs): # sending to workers
                comm.send(nb_sim_per_proc[i], dest=i, tag=10)
                comm.send(batch_to_simulate.dvec[np.sum(nb_sim_per_proc[:i]):np.sum(nb_sim_per_proc[:i+1])], dest=i, tag=11)
            batch_to_simulate.obj_vals = np.zeros((np.sum(nb_sim_per_proc),p.n_obj))
            batch_to_simulate.obj_vals[0:nb_sim_per_proc[0]] = p.perform_real_evaluation(batch_to_simulate.dvec[0:nb_sim_per_proc[0]])
            for i in range(1,nprocs): # receiving from workers
                batch_to_simulate.obj_vals[np.sum(nb_sim_per_proc[:i]):np.sum(nb_sim_per_proc[:i+1])] = comm.recv(source=i, tag=12)
            batch_to_simulate.dvec = batch_to_simulate.dvec[:np.sum(nb_sim_per_proc)]
            batch_to_simulate.fitness_modes = True*np.ones((batch_to_simulate.obj_vals.shape[0],p.n_obj), dtype=bool)
            
            batch_to_simulate.save_sim_archive(F_SIM_ARCHIVE) # logging
            db.dvec = np.vstack( (db.dvec, batch_to_simulate.dvec) )
            db.obj_vals = np.vstack( (db.obj_vals, batch_to_simulate.obj_vals) )
            db.fitness_modes = np.vstack( (db.fitness_modes, batch_to_simulate.fitness_modes) )
            db.update_best_sim(F_BEST_PROFILE, F_HYPERVOLUME)

            # Exit loop over the q sub-cycles if budget time exhausted
            if TIME_BUDGET>0:
                t_now = time.time()
                if TIME_BUDGET-(t_now-t_start)<SIM_TIME:
                    break

            # Surrogate update
            surr.perform_training()

            del batch_to_simulate
            del pop

        #----------------------Loop over Cycles----------------------#        

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
