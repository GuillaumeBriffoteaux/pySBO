import numpy as np

from Evolution_Controls.Ensemble_EC import Ensemble_EC


#-----------------------------------------#
#-------------static function-------------#
#-----------------------------------------#

#-------------sign reward function-------------#
def r_sign(t, e):
    r = np.sign(t-e)
    r[np.where(e==0.0)[0]] = 1
    return r

#-------------linear reward function-------------#
def r_linear(t, e):
    return t-e

#-------------hyperbolic tangent reward function-------------#
def r_tanh(t, e):
    return np.tanh(t-e)

#-------------asymptotic reward function-------------#
def r_asympt(t, e):
    r = -np.tan(np.pi/2.0 + (np.pi/(2.0*t))*e)
    r[np.where(e>t)[0]] = np.sinh(t-e[np.where(e>t)[0]])
    return r


#-------------------------------------------#
#-------------class Adaptive_EC-------------#
#-------------------------------------------#
class Adaptive_EC(Ensemble_EC):
    """Class for adaptive EC.

    Only one EC is active at a time. The active EC changes during the search according to a reward mechanism based on a reward function, a threshold and the error between the simulated costs and the predicted costs computed on the last batch of candidates.

    :param ECs_reward: rewards of each EC
    :type ECs_reward: list
    :param idx_active: index of the current active EC in ECs_list
    :type idx_active: positive int
    :param threshold: threshold value (mean error between simulated costs and predicted costs from the last batches of simulations)
    :type threshold: float
    :param rew_func: reward function (among `sign`, `linear`, `tanh`, `asympt`)
    :type rew_func: function
    :param saved_idx: 1 row per EC. A row contains the candidates indexes (into the population) ordered according to the associated EC.
    :type saved_idx: np.ndarray
    :param to_be_updated: True if the adaptive EC has to be updated, False otherwise
    :type to_be_updated: bool
    """
    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#

    #-------------__init__-------------#
    def __init__(self, rew_func, sim_costs, pred_costs, *ECs):
        """__init__ method's input

        :param rew_func: reward function (among `sign`, `linear`, `tanh`, `asympt`)
        :type rew_func: str
        :param sim_costs: simulation costs of the last batch of candidates (to initialize `threshold`)
        :type sim_costs: np.ndarray
        :param pred_costs: prediction costs of the last batch of candidates (to initialize `threshold`)
        :type pred_costs: np.ndarray
        :param ECs: evolution controls
        :type ECs: list(Evolution_Control)
        """

        Ensemble_EC.__init__(self, *ECs)            
        assert type(rew_func)==str
        assert type(sim_costs)==np.ndarray and type(pred_costs)==np.ndarray
        assert sim_costs.size==pred_costs.size
    
        self.__ECs_reward = [0 for i in range(0,len(ECs))]
        self.__idx_active=0
        self.__threshold =  np.average(np.power((sim_costs-pred_costs),2))
        if rew_func=="sign":
            self.__rew_func=r_sign
        elif rew_func=="linear":
            self.__rew_func=r_linear
        elif rew_func=="tanh":
            self.__rew_func=r_tanh
        elif rew_func=="asympt":
            self.__rew_func=r_asympt
        else:
            assert False

        self.__saved_idx = np.empty((0,0), dtype=int)
        self.__to_be_updated=False
    
    #-------------__del__-------------#
    def __del__(self):
        Ensemble_EC.__del__(self)
        del self.__ECs_reward
        del self.__idx_active
        del self.__threshold
        del self.__rew_func
        del self.__saved_idx
        del self.__to_be_updated

    #-------------__str__-------------#
    def __str__(self):
        res = "Adaptive Ensemble Evolution Control\n  ECs: {"
        for i,ec in enumerate(self.ECs_list):
            res+=" "+ec.__class__.__name__
        res+="}\n  ECs reward: {"
        for r in self.__ECs_reward:
            res+=" "+str(r)
        res+="}\n  index active EC: "+str(self.__idx_active)+"\n  threshold: "+str(self.__threshold)
        res+="\n  Reward function: "+str(self.__rew_func.__name__)
        res+="\n  saved indexes shape: "+str(self.__saved_idx.shape)+"\n  to be updated: "+str(self.__to_be_updated)
        return res


    #---------------------------------------------#
    #-------------getters and setters-------------#
    #---------------------------------------------#
    
    #-------------_get_ECs_reward-------------#
    def _get_ECs_reward(self):
        print("[Adaptive_EC.py] Impossible to get the ECs reward")
        return None

    #-------------_set_ECs_reward-------------#
    def _set_ECs_reward(self, new_ECs_reward):
        print("[Adaptive_EC.py] Impossible to modify the ECs reward")

    #-------------_del_ECs_reward-------------#
    def _del_ECs_reward(self):
        print("[Adaptive_EC.py] Impossible to delete the ECs reward")

    #-------------_get_idx_active-------------#
    def _get_idx_active(self):
        print("[Adaptive_EC.py] Impossible to get the index of the active EC")
        return None

    #-------------_set_idx_active-------------#
    def _set_idx_active(self, new_idx_active):
        print("[Adaptive_EC.py] Impossible to modify the index of the active EC")

    #-------------_del_idx_active-------------#
    def _del_idx_active(self):
        print("[Adaptive_EC.py] Impossible to delete the index of the active EC")

    #-------------_get_threshold-------------#
    def _get_threshold(self):
        print("[Adaptive_EC.py] Impossible to get the threshold")
        return None

    #-------------_set_threshold-------------#
    def _set_threshold(self, new_threshold):
        print("[Adaptive_EC.py] Impossible to modify the threshold")

    #-------------_del_threshold-------------#
    def _del_threshold(self):
        print("[Adaptive_EC.py] Impossible to delete the threshold")

    #-------------_get_rew_func-------------#
    def _get_rew_func(self):
        print("[Adaptive_EC.py] Impossible to get the reward function")
        return None

    #-------------_set_rew_func-------------#
    def _set_rew_func(self, new_rew_func):
        print("[Adaptive_EC.py] Impossible to modify the reward function")

    #-------------_del_rew_func-------------#
    def _del_rew_func(self):
        print("[Adaptive_EC.py] Impossible to delete the reward function")

    #-------------_get_saved_idx-------------#
    def _get_saved_idx(self):
        print("[Adaptive_EC.py] Impossible to get the saved indexes")
        return None

    #-------------_set_saved_idx-------------#
    def _set_saved_idx(self, new_saved_idx):
        print("[Adaptive_EC.py] Impossible to modify the saved indexes")

    #-------------_del_saved_idx-------------#
    def _del_saved_idx(self):
        print("[Adaptive_EC.py] Impossible to delete the saved indexes")

    #-------------_get_to_be_updated-------------#
    def _get_to_be_updated(self):
        print("[Adaptive_EC.py] Impossible to get the update control variable")
        return None

    #-------------_set_to_be_updated-------------#
    def _set_to_be_updated(self, new_to_be_updated):
        self.__to_be_updated=new_to_be_updated

    #-------------_del_to_be_updated-------------#
    def _del_to_be_updated(self):
        print("[Adaptive_EC.py] Impossible to delete the update control variable")

    #-------------property-------------#
    ECs_reward=property(_get_ECs_reward, _set_ECs_reward, _del_ECs_reward)
    idx_active=property(_get_idx_active, _set_idx_active, _del_idx_active)
    threshold=property(_get_threshold, _set_threshold, _del_threshold)
    rew_func=property(_get_rew_func, _set_rew_func, _del_rew_func)
    saved_idx=property(_get_saved_idx, _set_saved_idx, _del_saved_idx)
    to_be_updated=property(_get_to_be_updated, _set_to_be_updated, _del_to_be_updated)

    
    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#

    #-------------get_sorted_indexes-------------#
    def get_sorted_indexes(self, pop):
        Ensemble_EC.get_sorted_indexes(self, pop)
        
        self.__saved_idx = np.empty((len(self.ECs_list), pop.dvec.shape[0]), dtype=int)
        for i,ec in enumerate(self.ECs_list):
            self.__saved_idx[i] = ec.get_sorted_indexes(pop)
        self.__to_be_updated = True
        
        return self.__saved_idx[self.__idx_active]

    #-------------update_active-------------#
    def update_active(self, sim_costs, pred_costs):
        """Update the active EC and the threshold.

        The error, computed as the difference between simulated and predicted costs over the last batch of candidates, is compared to the threshold. A small error produces a reward for ECs that should have decided to simulate (including the active EC) and produces a penalty for other ECs. A large error produces a penalty for ECs that should have decided to simulate (including the active EC) and produces a reward for other ECs.

        :param sim_costs: simulation costs of the last batch of candidates (to compute the error)
        :type sim_costs: np.ndarray
        :param pred_costs: prediction costs of the last batch of candidates (to compute the error)
        :type pred_costs: np.ndarray
        """
        
        assert self.__to_be_updated
        
        error = np.power((sim_costs-pred_costs), 2)    
        rewards = self.__rew_func(self.__threshold, error)

        # Reward for the active EC
        self.__ECs_reward[self.__idx_active] = np.sum(rewards)

        # Reward for the deactivated ECs

        # According to simulations
        for i in range(self.__idx_active):
            intersect = np.intersect1d(self.__saved_idx[self.__idx_active,0:sim_costs.shape[0]], self.__saved_idx[i,0:sim_costs.shape[0]])
            idx_saved_idx = np.in1d(self.__saved_idx[self.__idx_active], intersect).nonzero()[0]
            self.__ECs_reward[i] = np.sum(rewards[idx_saved_idx])
            
        for i in range(self.__idx_active+1, len(self.ECs_list)):
            intersect = np.intersect1d(self.__saved_idx[self.__idx_active,0:sim_costs.shape[0]], self.__saved_idx[i,0:sim_costs.shape[0]])
            idx_saved_idx = np.in1d(self.__saved_idx[self.__idx_active], intersect).nonzero()[0]
            self.__ECs_reward[i] = np.sum(rewards[idx_saved_idx])

        # According to predictions or rejections
        for i in range(self.__idx_active):
            intersect = np.intersect1d(self.__saved_idx[self.__idx_active,0:sim_costs.shape[0]], self.__saved_idx[i,sim_costs.shape[0]:])
            idx_saved_idx = np.in1d(self.__saved_idx[self.__idx_active], intersect).nonzero()[0]
            self.__ECs_reward[i] -= np.sum(rewards[idx_saved_idx])
            
        for i in range(self.__idx_active+1, len(self.ECs_list)):
            intersect = np.intersect1d(self.__saved_idx[self.__idx_active,0:sim_costs.shape[0]], self.__saved_idx[i,sim_costs.shape[0]:])
            idx_saved_idx = np.in1d(self.__saved_idx[self.__idx_active], intersect).nonzero()[0]
            self.__ECs_reward[i] -= np.sum(rewards[idx_saved_idx])

        self.__threshold =  np.average(np.power((sim_costs-pred_costs),2))
        self.__idx_active = np.argmax(self.__ECs_reward)
        self.__to_be_updated = False
