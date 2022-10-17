from collections import deque
from lib2to3.pytree import convert
import random
import sys
from tkinter import Variable
import numpy as np
from AISettings.AISettingsInterface import Config
from model import DDQN
import json

import tensorflow as tf
from tensorflow import (
    Variable,
    convert_to_tensor,
    expand_dims,
    stack,
    squeeze
)
from tensorflow.keras.models import load_model
from tensorflow.compat.v1.train import exponential_decay
from tensorflow.math import argmax


def state_preprocess(state):
    return np.swapaxes(
            np.swapaxes(
                np.swapaxes(
                    state, 0, 3
                ), 1, 2
            ), 0, 1
        )

#based on pytorch RL tutorial by yfeng997: https://github.com/yfeng997/MadMario/blob/master/agent.py
class AIPlayer:
    def __init__(self, state_dim, action_space_dim, save_dir, date, config: Config):
        self.config = config
        self.state_dim = state_dim
        self.action_space_dim = action_space_dim
        self.save_dir = save_dir
        self.date = date

        self.net = DDQN(
            self.state_dim,
            self.action_space_dim,
            self.config.learning_rate,
            self.config.learning_rate_decay
        )

        self.config = config

        self.exploration_rate = self.config.exploration_rate 
        self.exploration_rate_decay = self.config.exploration_rate_decay
        self.exploration_rate_min = self.config.exploration_rate_min
        self.curr_step = 0

        """
            Memory
        """
        self.memory = deque(maxlen=self.config.deque_size)
        self.batch_size = self.config.batch_size
        self.save_every = self.config.save_every  # no. of experiences between saving Mario Net

        """
            Q learning
        """
        self.gamma = self.config.gamma
        self.burnin = self.config.burnin  # min. experiences before training
        self.learn_every = self.config.learn_every  # no. of experiences between updates to Q_online
        self.sync_every = self.config.sync_every  # no. of experiences between Q_target & Q_online sync

    def act(self, state):
        """
            Given a state, choose an epsilon-greedy action and update value of step.

            Inputs:
            state(LazyFrame): A single observation of the current state, dimension is (state_dim)
            Outputs:
            action_idx (int): An integer representing which action Mario will perform
        """
        # EXPLORE
        if (random.random() < self.exploration_rate):
            actionIdx = random.randint(0, self.action_space_dim-1)
        # EXPLOIT
        else:
            state = np.array(state)
            state = convert_to_tensor(state, dtype=tf.float32)
            state = expand_dims(state, 0)

            # road from shape (1, 4, 20, 16) to (20, 16, 4)
            state = state_preprocess(np.array(state))
            
            neuralNetOutput = self.net(state, model="online")
            neuralNetOutput = np.array(neuralNetOutput).squeeze()

            actionIdx = np.argmax(neuralNetOutput)

        # decrease exploration_rate
        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)

        # increment step
        self.curr_step += 1

        return actionIdx

    def cache(self, state, next_state, action, reward, done):
        """
        Store the experience to self.memory (replay buffer)

        Inputs:
        state (LazyFrame),
        next_state (LazyFrame),
        action (int),
        reward (float),
        done(bool))
        """
        state = np.array(state)
        next_state = np.array(next_state)

        state = convert_to_tensor(state, dtype=tf.float32)
        next_state = convert_to_tensor(next_state, dtype=tf.float32)
        action = convert_to_tensor([action])
        reward = convert_to_tensor([reward])
        done = convert_to_tensor([done])

        self.memory.append((state, next_state, action, reward, done))

    def recall(self):
        """
        Retrieve a batch of experiences from memory
        """
        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(stack, zip(*batch))
        return state, next_state, squeeze(action), squeeze(reward), squeeze(done)

    def learn(self):
        """Update online action value (Q) function with a batch of experiences"""
        if self.curr_step % self.sync_every == 0:
            self.sync_Q_target()

        if self.curr_step % self.save_every == 0:
            self.save()

        if self.curr_step < self.burnin:
            return None, None

        if self.curr_step % self.learn_every != 0:
            return None, None

        # Sample from memory get self.batch_size number of memories
        state, next_state, action, reward, done = self.recall()

        # Get TD Estimate, make predictions for the each memory
        td_est = self.td_estimate(state, action)

        # Get TD Target make predictions for next state of each memory
        td_tgt = self.td_target(reward, next_state, done)

        # Backpropagate loss through Q_online
        loss = self.update_Q_online(state, td_est, td_tgt)

        return (td_est.mean().numpy()[0], loss)

    def update_Q_online(self, state, td_estimate, td_target):
        self.net.backward('online', state, td_target)
        loss = self.net.loss_fn(td_estimate, td_target)

        return loss.item().numpy()[0]

    def sync_Q_target(self):
        for paramTarget, paramOnline in zip(self.net.target.trainable_variables, self.net.online.trainable_variables):
            paramTarget.assign(paramOnline.value())

    def td_estimate(self, state, action):
        """
            Output is batch_size number of rewards = Q_online(s,a) * 32
        """
        modelOutPut = self.net(state, model="online")
        current_Q = modelOutPut[np.arange(0, self.batch_size), action]  # Q_online(s,a)
        return current_Q

    def td_target(self, reward, next_state, done):
        """
            Output is batch_size number of Q*(s,a) = r + (1-done) * gamma * Q_target(s', argmax_a'( Q_online(s',a') ) )
        """
        next_state_Q = self.net(next_state, model="online") 
        best_action = argmax(next_state_Q, axis=1) # argmax_a'( Q_online(s',a') ) 
        next_Q = self.net(next_state, model="target")[np.arange(0, self.batch_size), best_action] # Q_target(s', argmax_a'( Q_online(s',a') ) )
        return (reward + (1 - done.float()) * self.gamma * next_Q).float() # Q*(s,a)

    def loadModel(self, path):
        dt = load_model(path)
        for paramOnline, paramDt in zip(self.net.online.trainable_variables, dt.trainable_variables):
            paramOnline.assign(paramDt.value())
        self.net.load_state_dict(dt["model"])
        self.exploration_rate = dt["exploration_rate"]
        print(f"Loading model at {path} with exploration rate {self.exploration_rate}")

    def saveHyperParameters(self):
        save_HyperParameters = self.save_dir / "hyperparameters"
        with open(save_HyperParameters, "w") as f:
            f.write(f"exploration_rate = {self.config.exploration_rate}\n")
            f.write(f"exploration_rate_decay = {self.config.exploration_rate_decay}\n")
            f.write(f"exploration_rate_min = {self.config.exploration_rate_min}\n")
            f.write(f"deque_size = {self.config.deque_size}\n")
            f.write(f"batch_size = {self.config.batch_size}\n")
            f.write(f"gamma (discount parameter) = {self.config.gamma}\n")
            f.write(f"learning_rate = {self.config.learning_rate}\n")
            f.write(f"learning_rate_decay = {self.config.learning_rate_decay}\n")
            f.write(f"burnin = {self.config.burnin}\n")
            f.write(f"learn_every = {self.config.learn_every}\n")
            f.write(f"sync_every = {self.config.sync_every}")

    def save(self):
        """
            Save the state to directory
        """
        save_path = (self.save_dir / f"mario_net_0{int(self.curr_step // self.save_every)}.chkpt")
        self.net.online.save(save_path)
        #torch.save(
            #dict(model=self.net.state_dict(), exploration_rate=self.exploration_rate),
            #save_path,
        #)
        print(f"MarioNet saved to {save_path} at step {self.curr_step}")
