import pandas as pd
import numpy as np
import tensorflow as tf
import utils, model, train, evaluations

def automation_X1_X2(
    con,
    dir_to_save:str, # './model/LogNormal1_LogNormal2/0'
    dict_MDN_hp:dict,
    dict_train_hp:dict,
    nakagami_or_lognormal:str,
    best_or_median:str,
    con2=None,
    data_size_train:int=1080000,
    data_size_test:int=100000,
):
    assert nakagami_or_lognormal in ['nakagami', 'lognormal'], "'nakagami_or_lognormal' can only be those following : ['nakagami', 'lognormal']"
    assert not (nakagami_or_lognormal == 'nakagami' and con2 is None), "nakagami_or_lognormal == 'nakagami' requires con2 !!"
    assert best_or_median in ['best', 'median'], "'best_or_median' can only be those following : ['best', 'median']"
    
    
        # Rough Train & Test Data Generation and Scalining
    if nakagami_or_lognormal == 'nakagami':
        # Nakagami 1
        train_data, _, train_indices = con.generate(data_size_train)
        test_data, _, test_indices = con2.generate(data_size_test)
        train_data_log_scaled = utils.log_transform_n(train_data)
        name_model = 'Nakagami'
        
    elif nakagami_or_lognormal == 'lognormal':
        # LogNormal 1
        train_data, train_indices = con.generate(number=data_size_train)
        test_data, test_indices = con.generate(number=data_size_test)
        train_data_log_scaled = utils.log_transform_ln(train_data)
        name_model = 'LogNormal'
        
    utils.nan_value_detector(train_data)
    utils.nan_value_detector(train_data_log_scaled)
    
        # Scaled DF
    df_genuine_log_scaled = pd.DataFrame(np.hstack((train_data_log_scaled, train_indices)), columns=['data', 'd'])
    
        # MDN Modeling
    if best_or_median == 'best': 
        model_mdn = tf.keras.models.load_model(f"./saved_models/{name_model}_best.h5", custom_objects={"mdn_loss":model.model_MDN().mdn_loss})
    elif best_or_median == 'median':
        model_mdn = tf.keras.models.load_model(f"./saved_models/{name_model}_median.h5", custom_objects={"mdn_loss":model.model_MDN().mdn_loss})

        # Model 2
        # Train Data Preparation
    x = df_genuine_log_scaled['d'].astype(int)
    y = df_genuine_log_scaled['data']

        # Test Data Preparation
    df_test_back_scaled = pd.DataFrame(np.hstack((test_data, test_indices,)), columns=['genuine', 'd'])

        # Config & Callbacks
    dir_main = f"{dir_to_save}/{name_model}2"
    save_format = name_model + '_ep_{epoch:02d}.h5'
    dir_checkpoint = f'{dir_main}/{save_format}'
    dir_log = f'{dir_main}/log.csv'
    model_ckpt = tf.keras.callbacks.ModelCheckpoint(filepath=dir_checkpoint, monitor='val_loss')
    model_log = tf.keras.callbacks.CSVLogger(dir_log, separator=',', append=False)
    callbacks = [model_ckpt, model_log]

        # Train
    go_next_stage = train.train_model(
        model_to_train=model_mdn,
        x=x,
        y=y,
        callbacks=callbacks,
        batch_size=dict_train_hp['batch_size'],
        epochs=dict_train_hp['epochs'],
        shuffle=dict_train_hp['shuffle'],
        validation_split=dict_train_hp['validation_split'],
        nan_patience=dict_train_hp['nan_patience'],
    )

    if go_next_stage == False:
        return
        
        #Evaluation
    df_avg_oa_and_std = evaluations.evaluation_all_epochs(
        df_to_evaluate=df_test_back_scaled,
        dir_main=dir_main,
        channel_type=nakagami_or_lognormal,
        max_epoch=dict_train_hp['epochs'],
    )

def automation_X1(
    con,
    dir_to_save:str, # './model/LogNormal1_LogNormal2/0'
    dict_MDN_hp:dict,
    dict_train_hp:dict,
    nakagami_or_lognormal:str,
    starting_from:bool,
    con2=None,
    data_size_train:int=1080000,
    data_size_test:int=100000,
):
    assert nakagami_or_lognormal in ['nakagami', 'lognormal'], "'nakagami_or_lognormal' can only be those following : ['nakagami', 'lognormal']"
    assert not (nakagami_or_lognormal == 'nakagami' and con2 is None), "nakagami_or_lognormal == 'nakagami' requires con2 !!"
    assert starting_from in [False], "Only `False` is available for this research."
    
    
    
        # Rough Train & Test Data Generation and Scalining
    if nakagami_or_lognormal == 'nakagami':
        # Nakagami 1
        train_data, _, train_indices = con.generate(data_size_train)
        test_data, _, test_indices = con2.generate(data_size_test)
        train_data_log_scaled = utils.log_transform_n(train_data)
        name_model = 'Nakagami'
        
    elif nakagami_or_lognormal == 'lognormal':
        # LogNormal 1
        train_data, train_indices = con.generate(number=data_size_train)
        test_data, test_indices = con.generate(number=data_size_test)
        train_data_log_scaled = utils.log_transform_ln(train_data)
        name_model = 'LogNormal'
        
    utils.nan_value_detector(train_data)
    utils.nan_value_detector(train_data_log_scaled)
    
        # Scaled DF
    # LogNormal 2
    df_genuine_log_scaled = pd.DataFrame(np.hstack((train_data_log_scaled, train_indices)), columns=['data', 'd'])
    
        # MDN Modeling
    if starting_from == False:
        model_mdn = model.model_MDN(
            event_shape=dict_MDN_hp['event_shape'],
            num_components=dict_MDN_hp['num_components'],
            dense_param=dict_MDN_hp['dense_param'],
            learning_rate=dict_MDN_hp['learning_rate'],
        ).build_MDN()
        
        # Train Data Preparation
    x = df_genuine_log_scaled['d'].astype(int)
    y = df_genuine_log_scaled['data']

        # Test Data Preparation
    df_test_back_scaled = pd.DataFrame(np.hstack((test_data, test_indices,)), columns=['genuine', 'd'])

        # Config & Callbacks
    dir_main = f"{dir_to_save}/{name_model}"
    save_format = name_model + '_ep_{epoch:02d}.h5'
    dir_checkpoint = f'{dir_main}/{save_format}'
    dir_log = f'{dir_main}/log.csv'
    model_ckpt = tf.keras.callbacks.ModelCheckpoint(filepath=dir_checkpoint, monitor='val_loss')
    model_log = tf.keras.callbacks.CSVLogger(dir_log, separator=',', append=False)
    callbacks = [model_ckpt, model_log]

        # Lognormal 2
    # Train
    go_next_stage = train.train_model(
        model_to_train=model_mdn,
        x=x,
        y=y,
        callbacks=callbacks,
        batch_size=dict_train_hp['batch_size'],
        epochs=dict_train_hp['epochs'],
        shuffle=dict_train_hp['shuffle'],
        validation_split=dict_train_hp['validation_split'],
        nan_patience=dict_train_hp['nan_patience'],
    )

    if go_next_stage == False:
        return
        
    #Evaluation
    df_avg_oa_and_std = evaluations.evaluation_all_epochs(
        df_to_evaluate=df_test_back_scaled,
        dir_main=dir_main,
        channel_type=nakagami_or_lognormal,
        max_epoch=dict_train_hp['epochs'],
    )