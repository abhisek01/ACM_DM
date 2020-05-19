import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam


class Critic:
    def __init__(self, sess, observation_dim):
        self.observation_dim = observation_dim
        # setting our created session as default session 
        K.set_session(sess)
        self.model = self.create_model()

    def create_model(self):
        state_input = Input(shape=self.observation_dim)
        state_h1 = Dense(70, activation='sigmoid')(state_input)
        output = Dense(1, activation='linear')(state_h1)
        model = Model(inputs=state_input, outputs=output)
        model.compile(loss="mse", optimizer=Adam(lr=0.002))
        return model
