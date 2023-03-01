import time
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import sklearn as sk
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

from Surrogates.Surrogate import Surrogate
from Problems.Problem import Problem

   
#---------------------------------------#
#-------------class BNN_MCD-------------#
#---------------------------------------#
class BNN_MCD(Surrogate):
    """Class for Bayesian Neural Network approximated via Monte Carlo Dropout.

    :param f_sim_archive: filename where are stored the simulated candidates
    :type f_sim_archive: str
    :param pb: problem the surrogate is associated with
    :type pb: Problem
    :param n_train_samples: number of training samples to extract from the end of `f_sim_archive`, if float('inf') all the samples from `f_sim_archive` are considered
    :type n_train_samples: positive int or inf, not zero
    :param f_train_log: filename where will be recorded training log
    :type f_train_log: str
    :param f_trained_model: filename where will be recorded the trained surrogate model
    :type f_trained_model: str
    :param y_bounds: lower and upper bounds of the objective values found in the training set
    :type y_bounds: np.ndarray
    :param outputs_scaler: objective values normalizer
    :type outputs_scaler: sklearn.preprocessing
    :param n_pred_subnets: number of sub-networks
    :type n_pred_subnets: positive int, not zero
    :param model: Keras model for the network
    :type model: tf.keras.Model
    """
    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#
 
    #-------------__init__-------------#
    def __init__(self, f_sim_archive, pb, n_train_samples, f_train_log, f_trained_model, n_pred_subnets):
        Surrogate.__init__(self, f_sim_archive, pb, n_train_samples, f_train_log, f_trained_model)

        # BNN_MCD specific attributes
        self.__y_bounds=np.empty((2,))
        self.__outputs_scaler = None
        self.__n_pred_subnets = n_pred_subnets
                
        # Network hyperparameters
        n_hidden_layers = 1
        n_units = 1024
        weight_decay = 1e-1
        weight_init_stdev = 1e-2
        p_drop = 0.1
        act_func = 'relu'

        # Network building
        input_layer = tf.keras.Input(shape=(self.pb.n_dvar,))
        inter_layer = tf.keras.layers.Dense(n_units, activation=act_func, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=weight_init_stdev))(input_layer)
        for i in range(n_hidden_layers-1):
            inter_layer = tf.keras.layers.Dropout(p_drop)(inter_layer, training=True)
            inter_layer = tf.keras.layers.Dense(n_units, activation=act_func, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=weight_init_stdev))(inter_layer)
        inter_layer = tf.keras.layers.Dropout(p_drop)(inter_layer, training=True)
        output_layer = tf.keras.layers.Dense(self.pb.n_obj, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=weight_init_stdev))(inter_layer)
        
        self.__model = tf.keras.Model(input_layer, output_layer)
        self.__model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.MeanSquaredError(), run_eagerly=False)
        self.__model.save_weights(self.f_trained_model)

        
    #-------------__del__-------------#
    def __del__(self):
        Surrogate.__del__(self)
        del self.__model
        del self.__y_bounds

        
    #-------------__str__-------------#
    def __str__(self):
        return "MCDropout-based Bayesian Neural Network\n  training set filename: "+self.f_sim_archive+"\n  associated problem: {"+self.pb.__str__()+"}\n  training set size = "+str(self.n_train_samples)+"\n  training log filename: "+self.f_train_log+"\n  trained model saved in "+self.f_trained_model

    
    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#

    
    #-------------perform_training-------------#
    # training from scratch
    def perform_training(self):
        Surrogate.perform_training(self)

        # Loading training data
        (x_train, y_train) = self.load_sim_archive()
        x_train = x_train[max(x_train.shape[0]-self.n_train_samples,0):x_train.shape[0]]
        y_train = y_train[max(y_train.shape[0]-self.n_train_samples,0):y_train.shape[0]]
        if self.pb.n_obj==1:
            y_train = y_train.reshape(-1, 1)

        # Normalize training data in [0,1]
        x_train = (x_train - self.pb.get_bounds()[0]) / (self.pb.get_bounds()[1] - self.pb.get_bounds()[0])
        self.__outputs_scaler = MinMaxScaler(copy=False)
        self.__outputs_scaler.fit(y_train)
        self.__outputs_scaler.transform(y_train)
        
        # Training
        t_start = time.time()
        self.__model.load_weights(self.f_trained_model)        
        kFold = KFold(n_splits=2)
        for train, test in kFold.split(x_train, y_train):
            my_histo = self.__model.fit( x=x_train[train], y=y_train[train], batch_size=32, epochs=10000, verbose=0, callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-8, patience=32, verbose=0, mode='min', baseline=None, restore_best_weights=True)], validation_data=(x_train[test], y_train[test]), shuffle=True )
        t_end = time.time()
        training_time = t_end-t_start

        # Compute training MSE and R-square in real-world units
        y_train = self.denormalize_predictions(y_train)
        preds = self.__model.predict(x_train, verbose=0)
        preds = self.denormalize_predictions(preds)
        training_mse = sk.metrics.mean_squared_error(y_train, preds, squared=True)
        training_r2 = sk.metrics.r2_score(y_train, preds)

        # Log about training
        with open(self.f_train_log, 'a') as my_file:
            my_file.write(str(y_train.shape[0])+" "+str(training_mse)+" "+str(training_r2)+" "+str(training_time)+"\n")


    #-------------perform_prediction-------------#
    def perform_prediction(self, candidates):
        assert self.pb.is_feasible(candidates)
        if candidates.ndim==1:
            candidates = np.array([candidates])

        # Normalizing candidates in [0,1]
        copy_candidates = (candidates - self.pb.get_bounds()[0]) / (self.pb.get_bounds()[1] - self.pb.get_bounds()[0])

        # Predictions
        preds=np.zeros((self.__n_pred_subnets, copy_candidates.shape[0], self.pb.n_obj))
        for i in range(0,self.__n_pred_subnets):
            preds[i,:,:] = self.__model.predict(copy_candidates, batch_size=copy_candidates.shape[0], verbose=0) # lies in [0,1]
            
        # Mean predictions and std predictions
        mean = np.mean(preds, 0)
        std = np.std(preds, 0)

        if self.pb.n_obj==1:
            mean = np.ndarray.flatten(mean)
            std = np.ndarray.flatten(std)
            
        # normalized results
        return (mean, std)

    
    #-------------denormalize_predictions-------------#
    def denormalize_predictions(self, preds):
        if self.pb.n_obj==1:
            preds = preds.reshape(-1, 1)
            
        denorm_preds = self.__outputs_scaler.inverse_transform(preds)
        
        if self.pb.n_obj==1:
            denorm_preds = denorm_preds.flatten()
            
        return denorm_preds

    
    #-------------normalize_obj_vals-------------#
    def normalize_obj_vals(self, obj_vals):
        if self.pb.n_obj==1:
            obj_vals = obj_vals.reshape(-1, 1)

        norm_obj_vals = self.__outputs_scaler.transform(obj_vals)

        if self.pb.n_obj==1:
            norm_obj_vals = norm_obj_vals.flatten()
        
        return norm_obj_vals
    

    #-------------load_trained_model-------------#
    def load_trained_model(self):
        self.__model = tf.keras.models.load_model(self.f_trained_model)
