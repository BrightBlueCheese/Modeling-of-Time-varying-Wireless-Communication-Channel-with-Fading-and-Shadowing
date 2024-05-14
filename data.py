import numpy as np
from scipy.special import gamma, gammaincinv

##
# For Nakagami..
# The original eta is 7.29e-14. 
# But we sat eta as 7.29 due to convinience (i.g., scale, data type limitation).
# What you need to do is just multipling e-14 to the final result.
##
class Nakagami:
    def __init__(
        self, 
        m_array:np.ndarray=np.array(np.hstack([np.tile(2, 4), np.tile(2, 10), np.tile(1, 16)])), 
        eta:float=7.29, 
        Pt:float=0.28183815, 
        alpha:int=2, 
        d_0:int=100, 
        d_array:np.ndarray=np.arange(10, 301, 10, dtype=np.int32), 
        # if with noise, then noise=noise_var else None
        noise:float=None, 
        low:float=0, 
        high:float=1.0
    ):
        self.m_array = m_array
        self.eta = eta
        self.Pt = Pt
        self.alpha = alpha
        self.d_0 = d_0
        self.d_array = d_array
        self.d_len = len(d_array)
        self.noise = noise
        self.low = low
        self.high = high
        
        if self.noise != None:
            self.noise_mean = 1.259e-1
            self.noise_var = self.noise # 1e-2
    
    def generate(self, number:int=10000):
        
        labels_index = np.random.choice(self.d_len, number)
        
        d = self.d_array[labels_index]
        m = self.m_array[labels_index]
        Pr_d = self.Pt * self.eta * np.power((self.d_0/d), self.alpha)
        r = np.random.uniform(size=number, low=self.low, high=self.high)

        if self.noise != None:
            noise = np.random.normal(loc=self.noise_mean, scale=np.sqrt(self.noise_var), size=number)
            nakagami_data = (Pr_d/m) * gammaincinv(m, gamma(m)*r) + noise
        else:
            nakagami_data = (Pr_d/m) * gammaincinv(m, gamma(m)*r)

        rough_condition = np.hstack((np.expand_dims(labels_index, axis=1), np.expand_dims(r, axis=1)))
        
        return np.expand_dims(nakagami_data, axis=1), rough_condition, np.expand_dims(labels_index, axis=1)

    def validate_mean(self):
        
        list_ideal_mean = list()
        
        for idx in range(len(self.m_array)):
            Pr_d = self.Pt * self.eta * np.power((self.d_0/self.d_array[idx]), self.alpha)
            
            ideal_mean = np.array(Pr_d).mean()
            list_ideal_mean.append(ideal_mean)
            
        return np.asarray(list_ideal_mean)

    def validate_var(self):
        
        list_ideal_var = list()
        
        for idx in range(len(self.m_array)):
            Pr_d = self.Pt * self.eta * np.power((self.d_0/self.d_array[idx]), self.alpha)

            ideal_var = np.power(Pr_d, 2) / self.m_array[idx]
            list_ideal_var.append(ideal_var)
        
        return np.asarray(list_ideal_var)

class LogNormal:
    # d_max = 300
    def __init__(
        self, 
        eta:float=7.29e-10, 
        Pt:float=0.28183815, 
        alpha_1:float=2.56,
        alpha_2:float=6.34,
        delta_1:float=3.9,
        delta_2:float=5.2,
        d_0:int=1,
        d_c:int=102,
        d_array:np.ndarray=np.arange(10, 301, 10, dtype=np.int32)
    ):
        
        self.eta = eta
        self.Pt = Pt
        self.alpha_1 = alpha_1
        self.alpha_2 = alpha_2
        self.delta_1 = delta_1
        self.delta_2 = delta_2
        self.d_0 = d_0
        self.d_c = d_c
        self.d_array = d_array
        self.d_len = len(d_array)

    def generate(self, number:int=10000):
        labels_index = np.random.choice(self.d_len, number)
        d = self.d_array[labels_index]
        
        P_d0 = 10 * np.log((self.Pt * self.eta) / np.power(self.d_0, 2))
        X_delta_1 = np.random.normal(loc=0, scale=self.delta_1, size=number)
        X_delta_2 = np.random.normal(loc=0, scale=self.delta_2, size=number)

        lognormal_calculator = np.vectorize(self.calculate_Pr_d)
        lognormal_data = lognormal_calculator(P_d0, d, X_delta_1, X_delta_2)
        
        return np.expand_dims(lognormal_data, axis=1), np.expand_dims(labels_index, axis=1)

    def calculate_Pr_d(self, P_d0, d, X_delta_1, X_delta_2):
        if d > self.d_c:
            return (P_d0 - 
                    10 * self.alpha_1 * np.log(self.d_c / self.d_0) - 
                    10 * self.alpha_2 * np.log(d / self.d_c) +
                    X_delta_2)
        else:
            return (P_d0 -
                    10 * self.alpha_1 * np.log(d / self.d_0) +
                    X_delta_1)