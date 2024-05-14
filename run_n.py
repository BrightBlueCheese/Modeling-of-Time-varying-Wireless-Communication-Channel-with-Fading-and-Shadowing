import numpy as np
import warnings
warnings.filterwarnings("ignore")

import data, model, utils, evaluations, train, model_hp, data_hp, automations

import random
import os
# # This is for only if you have multiple GPUs
# # If you have multiple GPUs, then pick ONE for this code.
# os.environ["CUDA_VISIBLE_DEVICES"]= "0" 
import tensorflow as tf

dict_MDN_hp = model_hp.dict_MDN_hp
dict_train_hp = model_hp.dict_train_hp
data_size_train = data_hp.data_size_train
data_size_test = data_hp.data_size_test

dict_n_train = data_hp.dict_n1_train
dict_n_test = data_hp.dict_n1_test

dir_main = './saved_models/Nakagami'
starting_from = False
nakagami_or_lognormal = 'nakagami'


num_versions = 10
for version in range(num_versions):
    # reset the random seed for every version since it's a sthochastic model
    np.random.seed(random.randint(1, 10000))

    con_n_train = data.Nakagami(
        m_array=dict_n_train["m_array"],
        eta=dict_n_train["eta"],
        Pt=dict_n_train["Pt"],
        alpha=dict_n_train["alpha"],
        d_0=dict_n_train["d_0"],
        d_array=dict_n_train["d_array"],
        noise=dict_n_train["noise"],
        low=dict_n_train["low"],
        high=dict_n_train["high"],
    ) 
    con_n_test = data.Nakagami(
        m_array=dict_n_test["m_array"],
        eta=dict_n_test["eta"],
        Pt=dict_n_test["Pt"],
        alpha=dict_n_test["alpha"],
        d_0=dict_n_test["d_0"],
        d_array=dict_n_test["d_array"],
        noise=dict_n_test["noise"],
        low=dict_n_test["low"],
        high=dict_n_test["high"],
    ) 
                        
    automations.automation_X1(
        con=con_n_train,
        con2=con_n_test,
        dir_to_save=f"{dir_main}/{version}",
        dict_MDN_hp=dict_MDN_hp,
        dict_train_hp=dict_train_hp,
        nakagami_or_lognormal=nakagami_or_lognormal,
        data_size_train=data_size_train,
        data_size_test=data_size_test,
        starting_from=starting_from,
    )