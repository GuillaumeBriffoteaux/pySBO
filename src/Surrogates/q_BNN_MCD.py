import os
import time
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
import sklearn as sk
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold

from Surrogates.Dropout_Fixed_Pred_Mask import Dropout_Fixed_Pred_Mask
from Surrogates.Surrogate import Surrogate
from Problems.Problem import Problem

   
#-----------------------------------------#
#-------------class q_BNN_MCD-------------#
#-----------------------------------------#
class q_BNN_MCD(Surrogate):
    # """Class for Bayesian Neural Network with Monte Carlo Dropout.

    # For a "master" proc, regular dropout is used (dropout at training, whole net at prediction)
    # For a "worker" proc, the net is never trained and a fixed dropout mask is used at prediction.

    # :param f_sim_archive: filename where are stored the past simulated individuals
    # :type f_sim_archive: str
    # :param pb: problem the surrogate is associated with
    # :type pb: Problem
    # :param n_train_samples: number of training samples to extract from the end of `f_sim_archive`, if float('inf') all the samples from `f_sim_archive` are considered
    # :type n_train_samples: positive int or inf, not zero
    # :param f_train_log: filename where will be recorded training log
    # :type f_train_log: str
    # :param f_trained_model: filename where will be recorded the trained surrogate model
    # :type f_trained_model: str
    # :param y_bounds: lower and upper bounds of the costs found in the training set.
    # :type y_bounds: np.ndarray
    # :param model: Keras model for the network
    # :type model: tf.keras.Model
    # """
    
    #-----------------------------------------#
    #-------------special methods-------------#
    #-----------------------------------------#
 
    #-------------__init__-------------#
    def __init__(self, f_sim_archive, pb, n_train_samples, f_train_log, f_trained_model, proc, nhl, nunits, f_init_weights=None):
        Surrogate.__init__(self, f_sim_archive, pb, n_train_samples, f_train_log, f_trained_model)

        # BNN_MCD specific attributes
        self.__y_bounds=np.empty((2,))
        self.__outputs_scaler = None

        if f_init_weights==None and proc=="master":
            print("[q_BNN_MCD.py] you must provide a file for init weights storage")
            assert False

        # Network hyperparameters
        n_hidden_layers = nhl
        n_units = nunits
        weight_decay = 1e-1
        weight_init_stdev = 1e-2
        p_drop = 0.05
        act_func = 'tanh'
            
        self.__f_init_weights=f_init_weights
        self.__proc = proc
        assert n_hidden_layers>0
        assert n_units>0
        assert p_drop>=0.0 and p_drop<=1.0

        # Network building
        input_layer = tf.keras.Input(shape=(self.pb.n_dvar,))
        inter_layer = tf.keras.layers.Dense(n_units, activation=act_func, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=weight_init_stdev))(input_layer)
        self.__list_dropout_layers = []
        for i in range(n_hidden_layers-1):

            if self.__proc=="master":
                inter_layer = tf.keras.layers.Dropout(p_drop)(inter_layer)
            else:
                self.__list_dropout_layers.append(Dropout_Fixed_Pred_Mask(p_drop, "origin"))
                inter_layer = self.__list_dropout_layers[-1](inter_layer, training=True)                
                
            inter_layer = tf.keras.layers.Dense(n_units, activation=act_func, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=weight_init_stdev))(inter_layer)

        if self.__proc=="master":
            inter_layer = tf.keras.layers.Dropout(p_drop)(inter_layer)
        else:
            self.__list_dropout_layers.append(Dropout_Fixed_Pred_Mask(p_drop, "origin"))
            inter_layer = self.__list_dropout_layers[-1](inter_layer, training=True)
            
        output_layer = tf.keras.layers.Dense(1, kernel_regularizer=tf.keras.regularizers.l2(weight_decay), kernel_initializer=tf.keras.initializers.RandomNormal(mean=0., stddev=weight_init_stdev))(inter_layer)
        
        self.__model = tf.keras.Model(input_layer, output_layer)
        self.__model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=tf.keras.losses.MeanSquaredError(), run_eagerly=False)
        if self.__proc=="master":
            self.__model.save_weights(self.__f_init_weights)
        else:
            # Set Dropout_Fixed_Pred_Mask layers in prediction mode
            for dpl in self.__list_dropout_layers:
                dpl._set_mode("pred")


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
    

    #----------------------------------------#
    #-------------object methods-------------#
    #----------------------------------------#



    
    #-------------perform_training-------------#
    def perform_training(self):
        Surrogate.perform_training(self)

        if self.__proc!="master":
            print("[q_BNN_MCD.py] you can not train a non-master surrogate")
            assert False

        # Loading training data
        (x_train, y_train) = self.load_sim_archive()
        x_train = x_train[max(x_train.shape[0]-self.n_train_samples,0):x_train.shape[0]]
        y_train = y_train[max(y_train.shape[0]-self.n_train_samples,0):y_train.shape[0]]
        y_train = y_train.reshape(-1, 1)
            
        # Normalize training data in [0,1]
        x_train = (x_train - self.pb.get_bounds()[0]) / (self.pb.get_bounds()[1] - self.pb.get_bounds()[0])
        self.__outputs_scaler = MinMaxScaler(copy=False)
        self.__outputs_scaler.fit(y_train)
        self.__outputs_scaler.transform(y_train)

        # Training
        t_start = time.time()
        self.__model.load_weights(self.__f_init_weights)
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
        
        # saving the model
        self.__model.save_weights(self.f_trained_model)



        
    #-------------perform_prediction-------------#
    def perform_prediction(self, candidates):
        assert self.pb.is_feasible(candidates)
        if candidates.ndim==1:
            candidates = np.array([candidates])

        # Normalizing candidates in [0,1]
        copy_candidates = (candidates - self.pb.get_bounds()[0]) / (self.pb.get_bounds()[1] - self.pb.get_bounds()[0])
        
        preds=np.zeros((copy_candidates.shape[0], self.pb.n_obj))
        # for the master proc, the whole set is used for prediction (no dropout)
        # for the worker procs, the fixed dropout mask is used for prediction (Dropout_Fixed_Pred_Mask)
        preds=self.__model(copy_candidates) # lies in [0,1]
        mean=np.array(preds)
        std=np.zeros(shape=mean.shape)

        if self.pb.n_obj==1:
            mean = np.ndarray.flatten(mean)
            std = np.ndarray.flatten(std)

        # results normalized
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
    def normalize_obj_vals(self, costs):
        if self.pb.n_obj==1:
            costs = costs.reshape(-1, 1)

        norm_costs = self.__outputs_scaler.transform(costs)

        if self.pb.n_obj==1:
            norm_costs = norm_costs.flatten()
        
        return norm_costs



    
    #-------------load_trained_model-------------#
    def load_trained_model(self):
        self.__model.load_weights(self.f_trained_model)
