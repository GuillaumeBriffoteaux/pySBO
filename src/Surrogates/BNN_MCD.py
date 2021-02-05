import os
import time
import pickle
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

from Surrogates.Surrogate import Surrogate
from Problems.Problem import Problem

    
#---------------------------------------#
#-------------class BNN_MCD-------------#
#---------------------------------------#
class BNN_MCD(Surrogate):
    """Class for Bayesian Neural Network with Monte Carlo Dropout (mono and multi dimensional targets).

    :param f_sim_archive: filename where are stored the past simulated individuals
    :type f_sim_archive: str
    :param pb: problem the surrogate is associated with
    :type pb: Problem
    :param n_train_samples: number of training samples to extract from the end of `f_sim_archive`, if float('inf') all the samples from `f_sim_archive` are considered
    :type n_train_samples: positive int or inf, not zero
    :param f_train_log: filename where will be recorded training log
    :type f_train_log: str
    :param f_trained_model: filename where will be recorded the trained surrogate model
    :type f_trained_model: str
    :param y_bounds: lower and upper bounds of the costs found in the training set.
    :type y_bounds: np.ndarray
    :param model: Keras model for the network
    :type model: tf.keras.Model
    """
    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#
 
    #-------------__init__-------------#
    def __init__(self, f_sim_archive, pb, n_train_samples, f_train_log, f_trained_model):
        Surrogate.__init__(self, f_sim_archive, pb, n_train_samples, f_train_log, f_trained_model)

        self.__y_bounds=np.empty((2,self.pb.n_obj))
        
        # Network hyperparameters
        n_hidden_layers = 2
        n_units = 256
        p_drop = 0.05
        act_func = 'tanh'
        assert n_hidden_layers>0
        assert n_units>0
        assert p_drop>=0.0 and p_drop<=1.0
        # Network building
        input_layer = tf.keras.Input(shape=(self.pb.n_dvar,))
        inter_layer = tf.keras.layers.Dropout(p_drop)(input_layer, training=True)
        inter_layer = tf.keras.layers.Dense(n_units, activation=act_func)(inter_layer)
        for i in range(n_hidden_layers-1):
            inter_layer = tf.keras.layers.Dropout(p_drop)(inter_layer, training=True)
            inter_layer = tf.keras.layers.Dense(n_units, activation=act_func)(inter_layer)
        inter_layer = tf.keras.layers.Dropout(p_drop)(inter_layer, training=True)
        output_layer = tf.keras.layers.Dense(self.pb.n_obj)(inter_layer)
        self.__model = tf.keras.Model(input_layer, output_layer)
        self.__model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False), loss='mse')
    
    #-------------__del__-------------#
    def __del__(self):
        Surrogate.__del__(self)
        del self.__model
        del self.__y_bounds
        
    #-------------__str__-------------#
    def __str__(self):
        return "MCDropout-based Bayesian Neural Network\n  training set filename: "+self.f_sim_archive+"\n  associated problem: {"+self.pb.__str__()+"}\n  training set size = "+str(self.n_train_samples)+"\n  training log filename: "+self.f_train_log+"\n  trained model saved in "+self.f_trained_model


    #---------------------------------------------#
    #-------------getters and setters-------------#
    #---------------------------------------------#

    #-------------_get_model-------------#
    def _get_model(self):
        print("[BNN_MCD.py] Impossible to get the model")
        return None

    #-------------_set_model-------------#
    def _set_model(self,new_model):
        print("[BNN_MCD.py] Impossible to modify the model")

    #-------------_del_model-------------#
    def _del_model(self):
        print("[BNN_MCD.py] Impossible to delete the model")

    #-------------_get_y_bounds-------------#
    def _get_y_bounds(self):
        print("[BNN_MCD.py] Impossible to get the y bounds")
        return None

    #-------------_set_y_bounds-------------#
    def _set_y_bounds(self,new_y_bounds):
        print("[BNN_MCD.py] Impossible to set the y bounds")

    #-------------_del_y_bounds-------------#
    def _del_y_bounds(self):
        print("[BNN_MCD.py] Impossible to delete the y bounds")

        
    #-------------property-------------#
    model=property(_get_model, _set_model, _del_model)
    y_bounds=property(_get_y_bounds, _set_y_bounds, _del_y_bounds)

    
    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#
    
    #-------------perform_prediction-------------#
    def perform_prediction(self, candidates):
        assert self.pb.is_feasible(candidates)
        if candidates.ndim==1:
            candidates = np.array([candidates])
            
        # Normalizing candidates
        copy_candidates = np.copy(candidates)
        copy_candidates = (2/(self.pb.get_bounds()[1]-self.pb.get_bounds()[0]))*copy_candidates+(-self.pb.get_bounds()[1]-self.pb.get_bounds()[0])/(self.pb.get_bounds()[1]-self.pb.get_bounds()[0]) # lies in [-1,1]

        # Predictions
        mean_preds=np.zeros((copy_candidates.shape[0], self.pb.n_obj))
        var_preds=np.zeros((copy_candidates.shape[0], self.pb.n_obj))
        n_subnets=2
        for i in range(0,n_subnets):
            preds = self.__model.predict(copy_candidates) # lies in [-1,1]
            preds = ((self.__y_bounds[1]-self.__y_bounds[0])*preds + (self.__y_bounds[1]+self.__y_bounds[0]))/2 # denormalization
            mean_preds += preds
            var_preds += pow(preds, 2)
        mean_preds = mean_preds/n_subnets
        if mean_preds.shape[1]==1:
            mean_preds = np.ndarray.flatten(mean_preds)
        if var_preds.shape[1]==1:
            var_preds = np.ndarray.flatten(var_preds)
        var_preds = (var_preds/n_subnets) - pow(mean_preds, 2)
        
        return (mean_preds, var_preds)
    
    #-------------perform_training-------------#
    # incremental training
    def perform_training(self):
        Surrogate.perform_training(self)

        # Loading training data
        (x_train, y_train) = self.load_sim_archive()
        x_train = x_train[max(x_train.shape[0]-self.n_train_samples,0):x_train.shape[0]]
        y_train = y_train[max(y_train.shape[0]-self.n_train_samples,0):y_train.shape[0]]
        if y_train.ndim==1:
            y_train = y_train.reshape(-1, 1)

        # Normalizing training data
        x_train_scaled = (2/(self.pb.get_bounds()[1]-self.pb.get_bounds()[0]))*x_train+(-self.pb.get_bounds()[1]-self.pb.get_bounds()[0])/(self.pb.get_bounds()[1]-self.pb.get_bounds()[0]) # lies in [-1,1]
        self.__y_bounds[0]=np.amin(y_train,0)
        self.__y_bounds[1]=np.amax(y_train,0)
        y_train_scaled = (2/(self.__y_bounds[1]-self.__y_bounds[0]))*y_train+(-self.__y_bounds[1]-self.__y_bounds[0])/(self.__y_bounds[1]-self.__y_bounds[0]) # lies in [-1,1]

        # Training
        t_start = time.time()
        my_histo = self.__model.fit( x=x_train_scaled, y=y_train_scaled, batch_size=y_train_scaled.shape[0], epochs=10, verbose=0, callbacks=[tf.keras.callbacks.ModelCheckpoint(filepath=self.f_trained_model, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='min', save_freq='epoch')], validation_data=(x_train_scaled, y_train_scaled), shuffle=True )
        self.__model.load_weights(self.f_trained_model)
        t_end = time.time()
        mse = self.__model.evaluate(x_train_scaled, y_train_scaled, verbose=0)

        # Saving the trained model
        self.__model.save(self.f_trained_model)

        # Log about training
        with open(self.f_train_log, 'a') as my_file:
            my_file.write(str(x_train_scaled.shape[0])+" "+str(mse)+" "+str((t_end-t_start))+"\n")

    #-------------load_trained_model-------------#
    def load_trained_model(self):
        self.__model = tf.keras.models.load_model(self.f_trained_model)
