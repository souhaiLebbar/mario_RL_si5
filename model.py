import tensorflow as tf
import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense, Conv2D, Flatten
from tensorflow.python.keras.models import clone_model
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.compat.v1.losses import huber_loss



class DDQN:
    def __init__(self, input_dim, output_dim, learning_rate, learning_rate_decay):

        self.online = Sequential()
        self.online.add(Conv2D(16, (4, 4), strides=(1, 1), activation='relu', input_shape=input_dim))
        self.online.add(Conv2D(32, (4, 4), strides=(1, 1), activation='relu'))
        self.online.add(Conv2D(32, (2, 2), strides=(1, 1), activation='relu'))
        self.online.add(Flatten())
        self.online.add(Dense(512, activation='relu'))
        self.online.add(Dense(output_dim, activation='relu'))

        self.loss_fn = huber_loss
        self.optimizer = Adam(learning_rate=learning_rate)
        self.learning_rate_decay = learning_rate_decay

        self.online.compile(optimizer=self.optimizer, metrics=['accuracy'])

        self.target = clone_model(self.online)

        # Q_target parameters are frozen
        for layer in self.target.layers:
            layer.trainable = False

    def __call__(self, input, model):
        if model == 'online':
            return self.online.predict(np.array([input]))[0]
        elif model == 'target':
            return self.target.predict(np.array([input]))[0]

    def backward(self, model, inputs, labels):
        model = self.online if model == "online" else self.target
        model.fit(inputs, labels, epochs=1, verbose=0)
        self.optimizer.lr.assign(self.optimizer.lr.value() * self.learning_rate_decay)





