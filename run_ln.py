import numpy as np
import warnings
warnings.filterwarnings("ignore")

import importlib
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

dict_ln = data_hp.dict_ln1
starting_from = False 
dir_main = f'./saved_models/LogNormal_from_N_{starting_from}' ###
nakagami_or_lognormal = 'lognormal'

num_versions = 10
for version in range(num_versions):
    # reset the random seed for every version since it's a sthochastic model
    np.random.seed(random.randint(1, 10000))

    # ln1 : urban
    con_ln1 = data.LogNormal(
        eta=dict_ln["eta"],
        Pt=dict_ln["Pt"],
        alpha_1=dict_ln["alpha_1"],
        alpha_2=dict_ln["alpha_2"],
        delta_1=dict_ln["delta_1"],
        delta_2=dict_ln["delta_2"],
        d_0=dict_ln["d_0"],
        d_c=dict_ln["d_c"],
        d_array=dict_ln["d_array"],
    ) 

    automations.automation_X1(
        con=con_ln1,
        dir_to_save=f"{dir_main}/{version}",
        dict_MDN_hp=dict_MDN_hp,
        dict_train_hp=dict_train_hp,
        nakagami_or_lognormal=nakagami_or_lognormal,
        data_size_train=data_size_train,
        data_size_test=data_size_test,
        starting_from=starting_from,
    )