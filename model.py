import tensorflow as tf
from keras.layers import Input, Dense
from keras.models import Model
from tensorflow_probability import layers

class model_MDN:
    def __init__(
        self,
        event_shape:list=[1],
        num_components:int=8,
        dense_param:int=256,
        num_dense_with_activation:int=4,
        learning_rate:float=0.005,
    ):

        self.event_shape = event_shape
        self.num_components = num_components
        self.dense_param = dense_param
        self.num_dense_with_activation = num_dense_with_activation
        self.learning_rate = learning_rate
        self.params_size = layers.MixtureNormal.params_size(self.num_components, self.event_shape)

    def mdn_loss(
        self,
        y_true,
        model
    ):
        return -model.log_prob(y_true)

    def build_MDN(self):
        mdn_input = Input(shape=(1,), name="mdn_input") # shape: the shape of the data "distance"

        x = Dense(self.dense_param, activation='relu', name="mdn_dense_w_activation_1")(mdn_input)
        for i in range(self.num_dense_with_activation-1):
            x = Dense(self.dense_param, activation='relu', name=f"mdn_dense_w_activation_{i+2}")(x)

        x = Dense(self.params_size, activation=None, name="mdn_last_dense_wo_activation")(x)
        mdn_output = layers.MixtureNormal(self.num_components, self.event_shape, name="mdn_mixture_normal")(x)

        mdn_model = Model(mdn_input, mdn_output, name="mdn")
        
        mdn_model.compile(
            optimizer=tf.optimizers.Adam(learning_rate=self.learning_rate),
            loss=self.mdn_loss,
        )
        mdn_model.summary()

        return mdn_model