from datetime import datetime
import tensorflow as tf
import numpy as np
import random

class NanDetector(tf.keras.callbacks.Callback):
    def __init__(
        self, 
        nan_checker,
    ):
        super().__init__()
        self.nan_checker = nan_checker
        
    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get('loss')
        if loss is None or tf.math.is_nan(loss):
            print("Loss is NaN. Pause the training...")
            self.model.stop_training = True  # Stop the current training
            
            random_seed = random.randint(1, 10000)
            np.random.seed(random_seed)
            tf.random.set_seed(random_seed)
            
            self.model.reset_states()  # Reset the model states
            self.nan_checker.is_nan = True

class NanChecker:
    def __init__(
        self, 
        is_nan:bool=False,
    ):
        self.is_nan = is_nan
        
def train_model(
    model_to_train,
    x,
    y,
    callbacks,
    batch_size:int=10000,
    epochs:int=10,
    shuffle:bool=True,
    validation_split:float=0.2,
    nan_patience:int=5,
):
    
    nan_checker = NanChecker()
    nan_detector = NanDetector(nan_checker)
    
    while nan_patience > 0:
        
        time_start = float(datetime.now().strftime('%H%M%S.%f'))
        model_to_train.fit(
            x, 
            y, 
            batch_size=batch_size,
            epochs=epochs,
            shuffle=shuffle,
            validation_split=validation_split,
            callbacks=callbacks + [nan_detector],
            # callbacks=callbacks,
        )
        time_end= float(datetime.now().strftime('%H%M%S.%f'))
        
        if nan_checker.is_nan:
            nan_patience -= 1
            nan_checker.is_nan = False
            
        else:
            print(f"Training time {time_end - time_start}")
            nan_patience = -1
            
    if nan_patience == 0:
        print("The loss keeps getting nan. Please check either the data or model's hyper parameters.")
        print("If keep getting nan while the data is okay, try to increase the model hyper-parameters instead of changing 'nan_patience'.")
        print("------------------------- Or ------------------------------")
        print("Or maybe this iteration (version) data contains one or more outliers (i.g. negative values pass through log scaling..)")
        print("In this case, try to increase the coefficients of both `log_transform_XX` and `inverse_log_transform_XX` functions from `utils.py`")
        return False # if False, stop the next process
        
    else:
        return True # if True, keep going the next process