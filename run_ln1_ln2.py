import numpy as np
import warnings
warnings.filterwarnings("ignore")

import importlib
import data, model, utils, evaluations, train, model_hp, data_hp, automations

import random
import os
# # This is for only if you have multiple GPUs
# # If you have multiple GPUs, then pick ONE for this code.
# os.environ["CUDA_VISIBLE_DEVICES"]= "1"
import tensorflow as tf

dict_MDN_hp = model_hp.dict_MDN_hp
dict_train_hp = model_hp.dict_train_hp
data_size_train = data_hp.data_size_train
data_size_test = data_hp.data_size_test

dict_ln2 = data_hp.dict_ln2

# Choose either 'best' (Global Best) or 'median' (Global Median)
best_or_median = 'best'
# best_or_median = 'median'
dir_main = f'./saved_models/LogNormal1_LogNormal2_{best_or_median}' ###
nakagami_or_lognormal = 'lognormal'

num_versions = 10
for version in range(num_versions):
    # reset the random seed for every version since it's a sthochastic model
    np.random.seed(random.randint(1, 10000))

    # ln2 : rural
    con_ln2 = data.LogNormal(
        eta=dict_ln2["eta"],
        Pt=dict_ln2["Pt"],
        alpha_1=dict_ln2["alpha_1"],
        alpha_2=dict_ln2["alpha_2"],
        delta_1=dict_ln2["delta_1"],
        delta_2=dict_ln2["delta_2"],
        d_0=dict_ln2["d_0"],
        d_c=dict_ln2["d_c"],
        d_array=dict_ln2["d_array"],
    ) 

    automations.automation_X1_X2(
        con=con_ln2,
        dir_to_save=f"{dir_main}/{version}",
        dict_MDN_hp=dict_MDN_hp,
        dict_train_hp=dict_train_hp,
        nakagami_or_lognormal=nakagami_or_lognormal,
        data_size_train=data_size_train,
        data_size_test=data_size_test,
        best_or_median=best_or_median,
    )