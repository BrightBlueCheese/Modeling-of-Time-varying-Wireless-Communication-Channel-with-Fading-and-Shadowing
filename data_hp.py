import numpy as np

data_size_train = 1080000
data_size_test = 100000

dict_n1_train = {
    "m_array": np.array(np.hstack([np.tile(2, 4), np.tile(2, 10), np.tile(1, 16)])),
    "eta": 7.29,
    "Pt": 0.28183815,
    "alpha": 2,
    "d_0":100,
    "d_array": np.arange(10, 301, 10, dtype=np.int32),
    # control 'low', and 'high' will prevent the train dataset to have some extreme values.
    # Thus, it will help the model to perform better. + Same for n2 train.
    "noise": 0.1,
    "low": 0.05,
    "high": 0.999,
}

dict_n1_test = {
    "m_array": dict_n1_train["m_array"],
    "eta": dict_n1_train["eta"],
    "Pt": dict_n1_train["Pt"],
    "alpha": dict_n1_train["alpha"],
    "d_0": dict_n1_train["d_0"],
    "d_array": dict_n1_train["d_array"],
    "noise": dict_n1_train["noise"],
    "low": 0,
    "high": 1.0,
}

dict_n2_train = {
    "m_array": np.array(np.hstack([np.tile(1.5, 14), np.tile(1, 16)])),
    "eta": 7.29,
    "Pt": 0.28183815,
    "alpha": 2.5,
    "d_0":100,
    "d_array": np.arange(10, 301, 10, dtype=np.int32),
    "noise": 0.1,
    "low": 0.05,
    "high": 0.999,
}

dict_n2_test = {
    "m_array": dict_n2_train["m_array"],
    "eta": dict_n2_train["eta"],
    "Pt": dict_n2_train["Pt"],
    "alpha": dict_n2_train["alpha"],
    "d_0": dict_n2_train["d_0"],
    "d_array": dict_n2_train["d_array"],
    "noise": dict_n2_train["noise"],
    "low": 0,
    "high": 1.0,
}

dict_ln1 = {
    "eta": 7.29e-10,
    "Pt": 0.28183815,
    "alpha_1": 2.56,
    "alpha_2": 6.34,
    "delta_1": 3.9,
    "delta_2": 5.2,
    "d_0": 1,
    "d_c": 102,
    "d_array": np.arange(10, 301, 10, dtype=np.int32),
}

dict_ln2 = {
    "eta": 7.29e-10,
    "Pt": 0.28183815,
    "alpha_1": 1.89,
    "alpha_2": 5.86,
    "delta_1": 3.1,
    "delta_2": 3.6,
    "d_0": 1,
    "d_c": 182,
    "d_array": np.arange(10, 301, 10, dtype=np.int32),
}