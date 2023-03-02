"""``SMBOEA.py`` Script running the parallel Surrogate-Model-Based Optimization + Evolutionary Algorithm for single-objective optimization.

The Surrogate-Model-Based Optimization + Evolutionary Algorithm is described in:
`F. Rehback, M. Zaefferer, J. Stork, and T. Bartz-Beielstein. Comparison of parallel surrogate-assisted optimization approaches. In Proceedings of the Genetic and Evolutionary Computation Conference, GECCO ’18, page 1348–1355, New York, NY, USA, 2018. Association for Computing Machinery. <http://www.cmap.polytechnique.fr/~nikolaus.hansen/proceedings/2018/GECCO/proceedings/proceedings_files/pap500s3-file1.pdf>`_

This algorithm is only meant to be run in parallel in at least 3 computing units.

Execution on Linux:
  * To run in parallel (in 4 computational units): ``mpiexec -n 4 python SMBOEA.py``
  * To run in parallel (in 4 computational units) specifying the units in `./hosts.txt`: ``mpiexec --machinefile ./host.txt -n 4 python SMBOEA.py``

Execution on Windows:
  * To run in parallel (in 4 computational units): ``mpiexec /np 4 python SMBOEA.py``
"""

# Hybrid AP per cycle:
#  rank 0 issues N_SIM new candidates through reproduction operators
#  rank (nprocs-1) issues 1 new candidate through EI maximization
#  rank (nprocs-2) issues 1 new candidate through POV minimization
# N_SIM and N_IC_OPT should be properly set

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
from Evolution.Tournament import Tournament
from Evolution.Tournament_Position import Tournament_Position
from Evolution.SBX import SBX
from Evolution.Polynomial import Polynomial
from Evolution.Elitist import Elitist
from Evolution.Custom_Elitism import Custom_Elitism

from Surrogates.GP import GP

from Evolution_Controls.POV_EC import POV_EC
from Evolution_Controls.Expected_Improvement_EC import Expected_Improvement_EC

from Global_Var import *


def main():
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    # Problem
    p = Schwefel(16)

    # Files
    DIR_STORAGE="outputs"
    if rank==0:
        F_BEST_PROFILE=DIR_STORAGE+"/best_profile.csv"
        F_INIT_POP=DIR_STORAGE+"/init_db.csv"
        shutil.rmtree(DIR_STORAGE, ignore_errors=True)
        os.makedirs(DIR_STORAGE, exist_ok=True)
    F_SIM_ARCHIVE=DIR_STORAGE+"/sim_archive.csv"
    F_TRAIN_LOG=DIR_STORAGE+"/training_log.csv"
    F_TRAINED_MODEL=DIR_STORAGE+"/trained_model"


    
    #--------------------------------#
    #-------------MASTER-------------#
    #--------------------------------#
    if rank==0:
    
        # Search arguments
        TIME_BUDGET = 0
        N_GEN = 2
        SIM_TIME = 15
        POP_SIZE = 72
        N_SIM = 2 # number of new candidates issued via the reproduction operators
        N_IC_OPT = 2 # number of new candidates issued via IC optimization
        if N_SIM+N_IC_OPT>nprocs:
            print("[SMBOEA.py] 1 simulation per core per cycle at most")
            assert False
        if TIME_BUDGET > 0:
            assert TIME_BUDGET > SIM_TIME
            N_GEN = 1000000000000

        # Population initialization / Parallel DoE
        sampler = DoE(p)
        pop = Population(p)
        pop.dvec = sampler.latin_hypercube_sampling(POP_SIZE)
        nb_sim_per_proc_init = (POP_SIZE//nprocs)*np.ones((nprocs,), dtype=int) # number of simulations per proc
        for i in range(POP_SIZE%nprocs):
            nb_sim_per_proc_init[i+1]+=1
        for i in range(1,nprocs): # sending to workers
            comm.send(nb_sim_per_proc_init[i], dest=i, tag=10)
            comm.send(pop.dvec[np.sum(nb_sim_per_proc_init[:i]):np.sum(nb_sim_per_proc_init[:i+1])], dest=i, tag=11)
        pop.obj_vals = np.zeros((pop.dvec.shape[0],))
        pop.obj_vals[0:nb_sim_per_proc_init[0]] = p.perform_real_evaluation(pop.dvec[0:nb_sim_per_proc_init[0]])
        for i in range(1,nprocs): # receiving from workers
            pop.obj_vals[np.sum(nb_sim_per_proc_init[:i]):np.sum(nb_sim_per_proc_init[:i+1])] = comm.recv(source=i, tag=12)
        pop.fitness_modes = True*np.ones(pop.obj_vals.shape, dtype=bool)
        for i in range(1,nprocs): # sending to workers
            comm.send(-2, dest=i, tag=10)

        # Logging
        pop.save_sim_archive(F_SIM_ARCHIVE)
        pop.update_best_sim(F_BEST_PROFILE)

        # Number of simulations per proc (only concern procs allocated to simulate the candidates issued per reproduction operators)
        nb_sim_per_proc = (N_SIM//(nprocs-N_IC_OPT))*np.ones(((nprocs-N_IC_OPT),), dtype=int)
        for i in range(N_SIM%(nprocs-N_IC_OPT)):
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
        surr = GP(F_SIM_ARCHIVE, p, float('inf'), F_TRAIN_LOG, F_TRAINED_MODEL, "rbf")
        surr.perform_training()

        # Inform that surrogate is ready
        for i in range((nprocs-N_IC_OPT), nprocs):
            comm.send(1, dest=i, tag=10)
            comm.send(Global_Var.obj_val_min, dest=i, tag=10)
        
        #----------------------Start Generation loop----------------------#
        for curr_gen in range(N_GEN):
            print("generation "+str(curr_gen))

            # Acquisition Process
            parents = select_op.perform_selection(pop, N_SIM)
            children = crossover_op.perform_crossover(parents)
            children.dvec = children.dvec[0:N_SIM,:]
            children = mutation_op.perform_mutation(children)
            assert p.is_feasible(children.dvec)

            # Parallel simulations of candidates issued via reproduction operators
            for i in range(1,(nprocs-N_IC_OPT)): # sending to workers
                comm.send(nb_sim_per_proc[i], dest=i, tag=10)
                comm.send(children.dvec[np.sum(nb_sim_per_proc[:i]):np.sum(nb_sim_per_proc[:i+1])], dest=i, tag=11)
            children.obj_vals = np.zeros((np.sum(nb_sim_per_proc),))
            children.obj_vals[0:nb_sim_per_proc[0]] = p.perform_real_evaluation(children.dvec[0:nb_sim_per_proc[0]])
            for i in range(1,(nprocs-N_IC_OPT)): # receiving from workers
                children.obj_vals[np.sum(nb_sim_per_proc[:i]):np.sum(nb_sim_per_proc[:i+1])] = comm.recv(source=i, tag=12)
            children.dvec = children.dvec[:np.sum(nb_sim_per_proc)]
            children.fitness_modes = np.ones(children.obj_vals.shape, dtype=bool)

            # get results from IC OPT
            for i in range((nprocs-N_IC_OPT), nprocs):
                children.dvec = np.append(children.dvec, comm.recv(source=i, tag=12), axis=0)
                children.obj_vals = np.append(children.obj_vals, comm.recv(source=i, tag=12), axis=0)
                children.fitness_modes = np.append(children.fitness_modes, comm.recv(source=i, tag=12), axis=0)
        
            # Logging
            children.save_sim_archive(F_SIM_ARCHIVE) 
            children.update_best_sim(F_BEST_PROFILE)

            # Replacement
            replace_op.perform_replacement(pop, children)
            assert p.is_feasible(pop.dvec)
            del children

            # Exit Generation loop if budget time exhausted
            if TIME_BUDGET>0:
                t_now = time.time()
                if TIME_BUDGET-(t_now-t_start)<SIM_TIME:
                    break
            if curr_gen==N_GEN-1:
                break

            # Surrogate update
            surr.perform_training()

            # Inform that surrogate is ready
            for i in range((nprocs-N_IC_OPT), nprocs):
                comm.send(1, dest=i, tag=10)
                comm.send(Global_Var.obj_val_min, dest=i, tag=10)
        #----------------------End Generation loop----------------------#

        # Stop workers
        for i in range(1,nprocs):
            comm.send(-1, dest=i, tag=10)    


    #------------------------------------------------#
    #-------------WORKER maximization EI-------------#
    #------------------------------------------------#
    elif rank==(nprocs-1):

        # Simulating the initial database
        nsim = comm.recv(source=0, tag=10)
        while nsim!=-2:
            candidates = np.empty((nsim, p.n_dvar))
            candidates = comm.recv(source=0, tag=11)
            obj_vals = p.perform_real_evaluation(candidates)
            comm.send(obj_vals, dest=0, tag=12)
            nsim = comm.recv(source=0, tag=10)

        # GA parameters
        POP_SIZE=100
        N_GEN=50
        N_CHLD=100
        
        # wait for surrogate first training
        surr_ready = comm.recv(source=0, tag=10)
        
        # Creating surrogate
        surr = GP(F_SIM_ARCHIVE, p, float('inf'), F_TRAIN_LOG, F_TRAINED_MODEL, "rbf")

        # Evolution Control
        ec_op = Expected_Improvement_EC(surr)

        # Operators
        sampler = DoE(p)
        select_op = Tournament_Position(2)
        crossover_op = SBX(0.9, 10)
        mutation_op = Polynomial(1.0/p.n_dvar, 50)
        replace_op = Custom_Elitism(ec_op)

        #----------------------Cycle loop----------------------#
        while surr_ready!=-1:

            # Receiving current best cost found so far
            Global_Var.obj_val_min = comm.recv(source=0, tag=10)

            # Updating surrogate
            surr.init_outputs_scaler()
            surr.load_trained_model()

            # Population initialization
            pop = Population(p)
            pop.dvec = sampler.latin_hypercube_sampling(POP_SIZE)
            pop.dvec = pop.dvec[ec_op.get_sorted_indexes(pop)]

            #----------------------Evolution loop----------------------#
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
            #----------------------End evolution loop----------------------#

            # Simulating candidate with highest EI
            batch_to_simulate = Population(p)
            batch_to_simulate.dvec = pop.dvec[0,:]
            batch_to_simulate.dvec = np.reshape(batch_to_simulate.dvec, (1,p.n_dvar))
            batch_to_simulate.obj_vals = np.zeros((1,))        
            batch_to_simulate.obj_vals[0] = p.perform_real_evaluation(batch_to_simulate.dvec)
            batch_to_simulate.fitness_modes = True*np.ones(batch_to_simulate.obj_vals.shape, dtype=bool)

            # send to master
            comm.send(batch_to_simulate.dvec, dest=0, tag=12)
            comm.send(batch_to_simulate.obj_vals, dest=0, tag=12)
            comm.send(batch_to_simulate.fitness_modes, dest=0, tag=12)

            del batch_to_simulate
            del pop
            del parents

            surr_ready = comm.recv(source=0, tag=10)

        #----------------------End Cycle loop----------------------#


    #-------------------------------------------------#
    #-------------WORKER minimization POV-------------#
    #-------------------------------------------------#
    elif rank==(nprocs-2):

        # Simulating the initial database
        nsim = comm.recv(source=0, tag=10)
        while nsim!=-2:
            candidates = np.empty((nsim, p.n_dvar))
            candidates = comm.recv(source=0, tag=11)
            obj_vals = p.perform_real_evaluation(candidates)
            comm.send(obj_vals, dest=0, tag=12)
            nsim = comm.recv(source=0, tag=10)

        # GA parameters
        POP_SIZE=100
        N_GEN=50
        N_CHLD=100

        # wait for surrogate first training
        surr_ready = comm.recv(source=0, tag=10)

        # Creating surrogate
        surr = GP(F_SIM_ARCHIVE, p, float('inf'), F_TRAIN_LOG, F_TRAINED_MODEL, "rbf")

        # Evolution Control
        ec_op = POV_EC(surr)

        # Operators
        sampler = DoE(p)
        select_op = Tournament_Position(2)
        crossover_op = SBX(0.9, 10)
        mutation_op = Polynomial(1.0/p.n_dvar, 50)
        replace_op = Custom_Elitism(ec_op)

        #----------------------Cycle loop----------------------#
        while surr_ready!=-1:

            # Receiving current best cost found so far
            Global_Var.obj_val_min = comm.recv(source=0, tag=10)

            # Updating surrogate
            surr.init_outputs_scaler()
            surr.load_trained_model()

            # Population initialization
            pop = Population(p)
            pop.dvec = sampler.latin_hypercube_sampling(POP_SIZE)
            pop.dvec = pop.dvec[ec_op.get_sorted_indexes(pop)]

            #----------------------Evolution loop----------------------#
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
            #----------------------End evolution loop----------------------#

            # Simulating candidate with lowest POV
            batch_to_simulate = Population(p)
            batch_to_simulate.dvec = pop.dvec[0,:]
            batch_to_simulate.dvec = np.reshape(batch_to_simulate.dvec, (1,p.n_dvar))
            batch_to_simulate.obj_vals = np.zeros((1,))        
            batch_to_simulate.obj_vals[0] = p.perform_real_evaluation(batch_to_simulate.dvec)
            batch_to_simulate.fitness_modes = np.ones(batch_to_simulate.obj_vals.shape, dtype=bool)

            # send to master
            comm.send(batch_to_simulate.dvec, dest=0, tag=12)
            comm.send(batch_to_simulate.obj_vals, dest=0, tag=12)
            comm.send(batch_to_simulate.fitness_modes, dest=0, tag=12)

            del batch_to_simulate
            del pop
            del parents

            surr_ready = comm.recv(source=0, tag=10)

        #----------------------End Cycle loop----------------------#

    
    #---------------------------------#
    #-------------WORKERS-------------#
    #---------------------------------#
    else:
        # Simulating initial population
        nsim = comm.recv(source=0, tag=10)
        while nsim!=-2:
            candidates = np.empty((nsim, p.n_dvar))
            candidates = comm.recv(source=0, tag=11)
            obj_vals = p.perform_real_evaluation(candidates)
            comm.send(obj_vals, dest=0, tag=12)
            nsim = comm.recv(source=0, tag=10)

        # Simulating new candidates
        nsim = comm.recv(source=0, tag=10)
        while nsim!=-1:
            candidates = np.empty((nsim, p.n_dvar))
            candidates = comm.recv(source=0, tag=11)
            obj_vals = p.perform_real_evaluation(candidates)
            comm.send(obj_vals, dest=0, tag=12)
            nsim = comm.recv(source=0, tag=10)


if __name__ == "__main__":
    main()
