# https://stackoverflow.com/questions/62375034/find-non-overlapping-area-between-two-kde-plots
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf

from scipy.stats import gaussian_kde
from datetime import datetime

import utils
import model
import os

def evaluation_OA(
    df_to_evaluate:pd.DataFrame,
    name_model:str=None,
    dir_main_to_save:str=f"./evaluations",
    show_plot:bool=True,
    to_save:bool=False, # not implemented
    is_lognormal:bool=False,
):

    
    list_for_df = list()
    for i in sorted(df_to_evaluate['d'].unique()):
        
        x0 = df_to_evaluate[df_to_evaluate['d'] == i]['genuine']
        x1 = df_to_evaluate[df_to_evaluate['d'] == i]['generated']
    
        kde0 = gaussian_kde(x0, bw_method=0.3)
        kde1 = gaussian_kde(x1, bw_method=0.3)
        
        xmin = min(x0.min(), x1.min())
        xmax = max(x0.max(), x1.max())
        
        dx = 0.2 * (xmax - xmin) # add a 20% margin, as the kde is wider than the data
        xmin -= dx
        xmax += dx
        
        # x_for_plot = np.linspace(xmin, xmax, len(x0))
        x_for_plot = np.linspace(xmin, xmax, int(len(x0)//10)) # or == 10000
        kde0_x = kde0(x_for_plot)
        kde1_x = kde1(x_for_plot)
        inters_x = np.minimum(kde0_x, kde1_x)
        
        area_inters_x = np.trapz(inters_x, x_for_plot)
        list_for_df.append([i, area_inters_x])

        if not show_plot:
            plt.ioff()
            plt.clf()
            plt.close()
            
        else :
            fig, ax = plt.subplots(figsize=(8, 6))
            
            if is_lognormal:
                mean0_x = np.sum(x_for_plot * kde0_x) / np.sum(kde0_x)
                plt.xlim(mean0_x-30, mean0_x+20)
                
            plt.fill_between(x_for_plot, kde0_x, 0, color='b', alpha=0.2, label='Genuine')
            plt.plot(x_for_plot, kde1_x, color='orange')
            plt.fill_between(x_for_plot, kde1_x, 0, color='orange', label='Generated', alpha=0.2)
            plt.plot(x_for_plot, inters_x, color='r')
            plt.fill_between(x_for_plot, inters_x, 0, facecolor='none', edgecolor='r', hatch='xx', label='Intersection')
            ax.set_title(f"{name_model} with Noise | d = {(i+1) * 10} | local Overlapped Area = {area_inters_x:.3f}")
            plt.xlabel('Receiving Power Scale')
            plt.ylabel('Probability Density Function')
            legend = plt.legend()
            plt.show()

    df_oa = pd.DataFrame(list_for_df, columns=['d', 'area_overlapped'])
    
    return df_oa

def evaluation_epoch(
    dir_ckpt:str,
    df_to_evaluate:pd.DataFrame,
    channel_type:str,
    show_plot:bool=False,
    to_save:bool=False,
):
    assert channel_type in ['lognormal', 'nakagami'], "'channel_type' must be either 'lognormal' or 'nakagami'."
    
    model_loaded = tf.keras.models.load_model(dir_ckpt, custom_objects={"mdn_loss":model.model_MDN().mdn_loss})

    time_start = float(datetime.now().strftime('%H%M%S.%f'))
    data_predicted = model_loaded.predict(df_to_evaluate['d'])
    
    time_end= float(datetime.now().strftime('%H%M%S.%f'))
    print(f"Prediction time {time_end - time_start}")

    # backscale
    if channel_type == 'nakagami':
        df_to_evaluate['generated'] = utils.inverse_log_transform_n(np.squeeze(data_predicted))
    elif channel_type == 'lognormal':
        df_to_evaluate['generated'] = utils.inverse_log_transform_ln(np.squeeze(data_predicted))
    else:
        raise ValueError("Invalid data detected - 'channel_type'")
    
    df_oa = evaluation_OA(
        df_to_evaluate=df_to_evaluate,
        show_plot=show_plot,
        to_save=to_save, 
    )
    
    return [df_oa['area_overlapped'].mean(), df_oa['area_overlapped'].std()]

def evaluation_all_epochs(
    df_to_evaluate:pd.DataFrame,
    dir_main:str,
    channel_type:str,
    name_log:str='log.csv',
    update_csv:bool=True,
    max_epoch:int=10,
):
    assert channel_type in ['lognormal', 'nakagami'], "'channel_type' must be either 'lognormal' or 'nakagami'."
    
    list_checkpoints = os.listdir(dir_main)
    list_checkpoints = [file for file in list_checkpoints if ".h5" in file]
    list_checkpoints = sorted(list_checkpoints, key=lambda x: float(x.split('_')[-1].split('.')[0]))

    list_avg_oa_and_std = list()
    for local_ckpt in list_checkpoints[:max_epoch]:
        try:
            local_list_avg_oa_and_std = evaluation_epoch(
                dir_ckpt=f"{dir_main}/{local_ckpt}",
                df_to_evaluate=df_to_evaluate,
                channel_type=channel_type,
            )
        except:
            print("Got Nan. Probably, its loss and val_loss are to large. Will store None instead.")
            local_list_avg_oa_and_std = [None, None]
            
        list_avg_oa_and_std.append(local_list_avg_oa_and_std)

    df_log = pd.read_csv(f'{dir_main}/log.csv')[['epoch' ,'loss', 'val_loss']]
    if df_log['epoch'][0] == 0:
        df_log['epoch'] = df_log['epoch'] + 1# csv epoch starts from 0. but from 1 for the actual .h5
    df_oa_std = pd.DataFrame(list_avg_oa_and_std, columns=['avg_oa', 'std'])
    df_new_log = pd.concat([df_log, df_oa_std], axis=1)
    df_new_log.to_csv(f'{dir_main}/log.csv', index=False)
    
    return df_new_log

def calculate_PE(
    arr_ideal:np.array, 
    arr_predicted:np.array,
):
    arr_PE = (((arr_predicted - arr_ideal) / arr_ideal) * 100).round(2)

    return arr_PE

def calculate_ScaledPE_Nakagami(
    df:pd.DataFrame,
    nakagami_con,
):
    arr_ideal_mean = nakagami_con.validate_mean()
    arr_ideal_var = nakagami_con.validate_var()

    arr_predicted_mean = df.groupby(by='d')['generated'].mean()
    arr_predicted_var = df.groupby(by='d')['generated'].var()

    arr_mean_PE = calculate_PE(arr_ideal=arr_ideal_mean, arr_predicted=arr_predicted_mean)
    arr_var_PE = calculate_PE(arr_ideal=arr_ideal_var, arr_predicted=arr_predicted_var)

    arr_mean_PE_avg = arr_mean_PE.abs().mean()
    arr_var_PE_avg = arr_var_PE.abs().mean()

    ScaledPE = (arr_mean_PE_avg*0.3 + arr_var_PE_avg*0.7) / 2

    return ScaledPE

def evaluation_ScaledPE_whole(
    list_of_list_dir_ckpt:list,
    df_to_evaluate:pd.DataFrame,
    channel_type:str,
    con_test,
):
    assert channel_type in ['lognormal', 'nakagami'], "'channel_type' must be either 'lognormal' or 'nakagami'."

    df_to_evaluate_cp = df_to_evaluate.copy()
    list_elements_for_df = list()
    for ep, list_ckpt_small in enumerate(list_of_list_dir_ckpt):
        print(f"==============={ep}==============")
        for ep_small, dir_ckpt in enumerate(list_ckpt_small):
            print(dir_ckpt)
            model_loaded = tf.keras.models.load_model(dir_ckpt, custom_objects={"mdn_loss":model.model_MDN().mdn_loss})
            data_predicted = model_loaded.predict(df_to_evaluate_cp['d'])
    
            # backscale
            if channel_type == 'nakagami':
                df_to_evaluate_cp['generated'] = utils.inverse_log_transform_n(np.squeeze(data_predicted))
            elif channel_type == 'lognormal':
                df_to_evaluate_cp['generated'] = utils.inverse_log_transform_ln(np.squeeze(data_predicted))
            else:
                raise ValueError("Invalid data detected - 'channel_type'")

            local_scaled_pe = calculate_ScaledPE_Nakagami(df=df_to_evaluate_cp, nakagami_con=con_test)
            list_elements_for_df.append([ep, ep_small+1, local_scaled_pe])

    df_scaled_pe = pd.DataFrame(list_elements_for_df, columns=['version', 'epoch', 'ScaledPE'])
    
    return df_scaled_pe