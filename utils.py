import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Log Normal
    # coeef (435) MUST be larger than the minimum value of the original data
def log_transform_ln(X):
    return np.log(X + 435) 

def inverse_log_transform_ln(X_log_transformed):
    return np.exp(X_log_transformed) - 435

# Nakagami
def log_transform_n(X):
    return np.log(X + 2)
        
def inverse_log_transform_n(X_log_transformed):
    return np.exp(X_log_transformed) - 2


def nan_value_detector(arr=np.array):
    status = (np.isinf(arr) | np.isnan(arr)).any()
    assert not status, "The data contains np.nan or +- np.inf. Check your data's hyper-parameters. If nan value was detected at the scaled value side, then check whether you have changed the 'scaling function' or not."


def plot_original_and_scaled_data(df:pd.DataFrame):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
            
    sns.kdeplot(data=df[['original_data']], x='original_data', 
                fill=True, ax=ax1, color='red')
    ax1.set_title(f"Genuine - original")
    
    sns.kdeplot(data=df[['scaled_data']], 
                x='scaled_data', fill=True, ax=ax2, color='green')
    ax2.set_title(f"Genuine - scaled")
    
    fig.suptitle(f"Original vs Scaled as a Whole", y=1.05)
    plt.show()
    
def plot_original_and_scaled_data_by_d(df:pd.DataFrame):
    for i in range(len(df['d'].unique())):
        # Melt the data to long-form
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
        
        sns.kdeplot(data=df[df['d'] == i], x='original_data', 
                    fill=True, ax=ax1, color='red')
        ax1.set_title(f"Genuine - Original | d_idx : {i}")
        
        sns.kdeplot(data=df[df['d'] == i], 
                    x='scaled_data', fill=True, ax=ax2, color='green')
        ax2.set_title(f"Genuine - Scaled | d_idx : {i}")
        
        fig.suptitle(f"Original vs Scaled by distance 'd' | d = {int((i+1) * 10)}", y=1.05)
        plt.show()

def plot_data_distribution_by_d(df:pd.DataFrame):
    fig, ax = plt.subplots(figsize=(4, 3))
            
    sns.scatterplot(x=df['d'], y=df['original_data'], s=1)
    ax.set_title(f"Scatter Plot hue by d")
    
    plt.show()

