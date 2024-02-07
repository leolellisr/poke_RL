# Base Keras classes for training and testing with REINFORCE 

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Sequential, load_model

# REINFORCE agent
class REINFORCEAgent(object):
    def __init__(self, nb_states, nb_actions, nb_hidden=128, alpha=0.0005, gamma=0.99):
        self.nb_actions = nb_actions
        self.action_space = [i for i in range(nb_actions)]
        self.gamma = gamma
        self.list_of_s = []
        self.list_of_a = []
        self.list_of_r = []
        # model construction
        self.model = Sequential()
        # 1st hidden layer
        self.model.add(Dense(nb_hidden, activation="relu", input_shape=(nb_states,)))
        # fix
        self.model.add(Flatten())
        # 2nd hidden layer
        self.model.add(Dense(nb_hidden, activation="relu"))
        # output layer
        self.model.add(Dense(nb_actions, activation="softmax"))
        # set the learning rate
        optimizer = tf.keras.optimizers.Adam(alpha)
        # loss function
        self.model.compile(optimizer=optimizer, loss=self.log_likelihood)

    @staticmethod
    def log_likelihood(y_true, y_pred):
        pi = K.clip(y_pred, 1e-8, 1 - 1e-8)
        return K.sum(-K.log(pi) * y_true)

    def get_action(self, s):
        p = self.model.predict(s[np.newaxis, :])[0]
        return np.random.choice(self.action_space, p=p)

    def add_sar(self, s, a, r):
        self.list_of_s.append(s)
        self.list_of_a.append(a)
        self.list_of_r.append(r)

    def fit(self):
        s_array = np.array(self.list_of_s)
        list_of_v = []
        for r in reversed(self.list_of_r):
            if not list_of_v:
                list_of_v.append(r)
            else:
                list_of_v = [r + self.gamma * list_of_v[0]] + list_of_v
        # adjust and normalize the list of returns
        list_of_v = list_of_v[1:] + [0]
        mean = np.mean(list_of_v)
        std = np.std(list_of_v)
        std = 1 if std == 0 else std
        list_of_norm_v = (list_of_v - mean) / std
        v_array = np.zeros([len(list_of_norm_v), self.nb_actions])
        v_array[np.arange(len(list_of_norm_v)), self.list_of_a] = np.array(list_of_norm_v)
        self.model.fit(x=s_array, y=v_array)
        # reset
        self.list_of_s = []
        self.list_of_a = []
        self.list_of_r = []

    def save_model(self, file_path):
        self.model.save(file_path)

    def load_model(self, file_path):
        self.model = load_model(file_path)
