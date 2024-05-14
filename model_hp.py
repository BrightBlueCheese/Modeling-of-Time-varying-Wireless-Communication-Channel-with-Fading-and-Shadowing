dict_MDN_hp = {
    "event_shape": [1],
    "num_components": 8,
    # "num_components": 12,
    # "dense_param": 256,
    "dense_param": 128,
    "num_dense_with_activation": 4,
    "learning_rate": 0.005, #0.005
}

dict_train_hp = {
    "batch_size": 10000,
    # "epochs": 10,
    "epochs": 15,
    "shuffle": True,
    "validation_split": 0.2,
    "nan_patience": 5,
}

dict_MDN_hp_larger = {
    "event_shape": [1],
    # "num_components": 8,
    "num_components": 12,
    "dense_param": 128,
    "num_dense_with_activation": 4,
    "learning_rate": 0.005, #0.005
}

