import numpy as np

from Evolution_Controls.Ensemble_EC import Ensemble_EC


#-------------------------------------------#
#-------------class Adaptive_EC-------------#
#-------------------------------------------#
class Adaptive_EC(Ensemble_EC):
    """Class for adaptive EC.

    Only one EC is active at a time. The active EC is switched according to a stagnation detection mechanism and a reward mechanism. The stagnation detection mechanism triggers the EC switch when the best simulated objective value found so far has not improve by at least 1e-8 during 8 batches (i.e. 8 acquisition processes). The reward mechanism chooses which EC becomes active by rewarding and penalizing all the ECs at each batch (i.e. each acquisition process) according to the error between the simulated objective values and the predicted objective values computed on the last batch of candidates.

    :param previous_best_obj_val: best simulated objective value found so far
    :type previous_best_obj_val: float
    :param counter: counter for the stagnation detection mechanism
    :type counter: int
    :param ECs_reward: rewards of each EC
    :type ECs_reward: list
    :param idx_active: index of the current active EC in ECs_list
    :type idx_active: positive int
    :param threshold: threshold value (mean error between simulated objective values and predicted objective values from the last batches of simulations)
    :type threshold: float
    :param saved_idx: 1 row per EC. A row contains the candidates indexes (into the population) ordered according to the associated EC.
    :type saved_idx: np.ndarray
    """
    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#

    #-------------__init__-------------#
    def __init__(self, sim_obj_vals, pred_obj_vals, init_best_obj_val, *ECs):
        """__init__ method's input

        :param sim_obj_vals: simulation objective values of the last batch of candidates (to initialize `threshold`)
        :type sim_obj_vals: np.ndarray
        :param pred_obj_vals: prediction objective values of the last batch of candidates (to initialize `threshold`)
        :type pred_obj_vals: np.ndarray
        :param init_best_obj_val: initial best simulated objective value found
        :type init_best_obj_val: float
        :param ECs: evolution controls
        :type ECs: list(Evolution_Control)
        """
        
        Ensemble_EC.__init__(self, *ECs)
        assert type(sim_obj_vals)==np.ndarray and type(pred_obj_vals)==np.ndarray
        assert sim_obj_vals.size==pred_obj_vals.size
        assert type(init_best_obj_val)==float or type(init_best_obj_val)==np.float64

        # attributes related to stagnation detection
        self.__previous_best_obj_val = init_best_obj_val
        self.__counter = 0

        # attributes related to reward
        self.__ECs_reward = [0 for i in range(0,len(ECs))]
        self.__idx_active=0
        self.__threshold =  np.average(np.power((sim_obj_vals-pred_obj_vals),2))
        self.__saved_idx = np.empty((0,0), dtype=int)

    
    #-------------__del__-------------#
    def __del__(self):
        Ensemble_EC.__del__(self)
        del self.__ECs_reward
        del self.__idx_active
        del self.__threshold
        del self.__saved_idx

        del self.__previous_best_obj_val
        del self.__counter


    #-------------__str__-------------#
    def __str__(self):
        res = "Adaptive Ensemble Evolution Control\n  ECs: {"
        for i,ec in enumerate(self.ECs_list):
            res+=" "+ec.__class__.__name__
        res+="}\n  ECs reward: {"
        for r in self.__ECs_reward:
            res+=" "+str(r)
        res+="}\n  index active EC: "+str(self.__idx_active)+"\n  threshold: "+str(self.__threshold)
        res+="\n  saved indexes shape: "+str(self.__saved_idx.shape)+"\n  previous best objective value: "+str(self.__previous_best_obj_val)
        res+="\n  counter: "+str(self.__counter)
        return res


    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#

    
    #-------------get_sorted_indexes-------------#
    def get_sorted_indexes(self, pop):
        Ensemble_EC.get_sorted_indexes(self, pop)
        
        self.__saved_idx = np.empty((len(self.ECs_list), pop.dvec.shape[0]), dtype=int)
        for i,ec in enumerate(self.ECs_list):
            self.__saved_idx[i] = ec.get_sorted_indexes(pop)
        
        return self.__saved_idx[self.__idx_active]

    
    #-------------update_active-------------#
    def update_active(self, sim_obj_vals, pred_obj_vals, new_best_obj_val):
        """Set the rewards/penalties, update the threshold and the counter. Potentially update the active EC.

        Reward mechanism: The error, computed as the difference between simulated and predicted objective values over the last batch of candidates, is compared to the threshold. A small error produces a reward for ECs that should have decided to simulate (including the active EC) and produces a penalty for other ECs. A large error produces a penalty for ECs that should have decided to simulate (including the active EC) and produces a reward for other ECs.

        Stagnation detection mechanism: if the best simulated objective value found so far has not improved by at least 1e-2 during 8 iteration, the active EC is updated. If the last active EC get the higher reward, it remains the active EC. When the stagnation detection occurs, the counter and the reward vector are reset.

        :param sim_obj_vals: simulation objective values of the last batch of candidates (to compute the error)
        :type sim_obj_vals: np.ndarray
        :param pred_obj_vals: prediction objective values of the last batch of candidates (to compute the error)
        :type pred_obj_vals: np.ndarray
        :param new_best_obj_val: best simulated objective value found so far
        :type new_best_obj_val: float
        """

        #----------------REWARD
        
        error = np.power((sim_obj_vals-pred_obj_vals), 2)
        rewards =np.tanh(self.__threshold-error)

        # Reward for the active EC
        self.__ECs_reward[self.__idx_active] += np.sum(rewards)

        # Reward for the deactivated ECs

        # According to simulations
        for i in range(self.__idx_active):
            intersect = np.intersect1d(self.__saved_idx[self.__idx_active,0:sim_obj_vals.shape[0]], self.__saved_idx[i,0:sim_obj_vals.shape[0]])
            idx_saved_idx = np.in1d(self.__saved_idx[self.__idx_active], intersect).nonzero()[0]
            self.__ECs_reward[i] += np.sum(rewards[idx_saved_idx])
            
        for i in range(self.__idx_active+1, len(self.ECs_list)):
            intersect = np.intersect1d(self.__saved_idx[self.__idx_active,0:sim_obj_vals.shape[0]], self.__saved_idx[i,0:sim_obj_vals.shape[0]])
            idx_saved_idx = np.in1d(self.__saved_idx[self.__idx_active], intersect).nonzero()[0]
            self.__ECs_reward[i] += np.sum(rewards[idx_saved_idx])

        # According to predictions or rejections
        for i in range(self.__idx_active):
            intersect = np.intersect1d(self.__saved_idx[self.__idx_active,0:sim_obj_vals.shape[0]], self.__saved_idx[i,sim_obj_vals.shape[0]:])
            idx_saved_idx = np.in1d(self.__saved_idx[self.__idx_active], intersect).nonzero()[0]
            self.__ECs_reward[i] -= np.sum(rewards[idx_saved_idx])
            
        for i in range(self.__idx_active+1, len(self.ECs_list)):
            intersect = np.intersect1d(self.__saved_idx[self.__idx_active,0:sim_obj_vals.shape[0]], self.__saved_idx[i,sim_obj_vals.shape[0]:])
            idx_saved_idx = np.in1d(self.__saved_idx[self.__idx_active], intersect).nonzero()[0]
            self.__ECs_reward[i] -= np.sum(rewards[idx_saved_idx])

        self.__threshold =  np.average(np.power((sim_obj_vals-pred_obj_vals),2))


        #----------------EARLY-STOPPING

        if (self.__previous_best_obj_val-new_best_obj_val)<=1e-8:
            self.__counter += 1

        self.__previous_best_obj_val=new_best_obj_val

        if self.__counter==8:
            self.__idx_active = np.argmax(self.__ECs_reward)
            self.__counter=0
            self.__ECs_reward = [0 for i in range(0,len(self.ECs_list))]
