import random
from asyncio import sleep
import itertools

from keras import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, AveragePooling1D, Dropout
from pyboy import WindowEvent
from AISettingsInterface import AISettingsInterface, Config

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from collections import deque
import numpy as np

# experiences replay buffer size
REPLAY_SIZE = 10000
# discount factor for target Q to caculate the TD aim value
GAMMA = 0.9


class GameState():
    def __init__(self, pyboy):
        game_wrapper = pyboy.game_wrapper()
        "Find the real level progress x"
        level_block = pyboy.get_memory_value(0xC0AB)
        # C202 Mario's X position relative to the screen
        mario_x = pyboy.get_memory_value(0xC202)
        scx = pyboy.botsupport_manager().screen(
        ).tilemap_position_list()[16][0]
        real = (scx - 7) % 16 if (scx - 7) % 16 != 0 else 16

        self.real_x_pos = level_block * 16 + real + mario_x
        self.time_left = game_wrapper.time_left
        self.lives_left = game_wrapper.lives_left
        self.score = game_wrapper.score
        self._level_progress_max = max(game_wrapper._level_progress_max, self.real_x_pos)
        self.world = game_wrapper.world


class DQN():
    def __init__(self, observation_space, action_space,_env):
        self.model = None
        self.memory = []  # [[1,1, 2500], [1,1, 200]]
        self.env=_env
        # for DQN

        self.state_dim1 = observation_space.shape[0]

        self.state_dim2 = observation_space.shape[1]

        self.state_dim3 = observation_space.shape[2]
        # the action is the output vector and it has two dimensions
        self.action_dim = action_space
        # init experience replay, the deque is a list that first-in & first-out
        self.replay_buffer = deque()

        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())
        self.cnn_model()

        self.epsilon = 1
        self.epsilon_decay = .995
        self.epsilon_min = 0.1

    def cnn_model(self):
        input_shape=self.env.observation_space.shape
        model = Sequential()
        # model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
        # model.add(Dropout(0.3))
        # model.add(Flatten())
        model.add(Dense(32, activation="relu",input_shape=input_shape[1:]))
        model.add(Dense(16, activation='relu'))
        model.add(Dense(5, activation='linear'))
        model.add(Flatten())
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

        return model

    def save_exp(self, _state, _action, _reward, _next_state, _done):
        # 将各种状态 存进memory中
        self.memory.append((_state, _action, _reward, _next_state, _done))

    def train_exp(self, batch_size):
        self.model = self.cnn_model()
        batches = min(batch_size, len(self.memory))
        batches = np.random.choice(len(self.memory), batches)

        for i in batches:
            # pick out parameters from memory
            _state, _action, _reward, _next_state, _done = self.memory[i]

            # return reward
            y_reward = _reward
            # if not done, get a discounted reward
            if not _done:
                y_reward = _reward + GAMMA * np.amax(self.model.predict(_next_state)[0])

            _y = self.model.predict(_state)

            _y[0][_action] = y_reward

            self.model.fit(_state, _y, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def act(self, _state):  # return a random number or the best reward
        # return random
        if np.random.rand() <= self.epsilon:
            # print('000000000',self.env.action_space.sample())
            return self.env.action_space.sample()
        else:
            # return best
            act_values = self.model.predict(_state)
            return np.argmax(act_values[0])  # returns action




    # old codes
    def GetReward(self, prevGameState: GameState, pyboy):
        """
        previousMario = mario before step is taken
        current_mario = mario after step is taken
        """
        timeRespawn = pyboy.get_memory_value(
            0xFFA6)  # Time until respawn from death (set when Mario has fell to the bottom of the screen)
        if (timeRespawn > 0):  # if we cant move return 0 reward otherwise we could be punished for crossing a level
            return 0

        "Get current game state"
        current_mario = self.GetGameState(pyboy)

        if max((current_mario.world[0] - prevGameState.world[0]),
               (current_mario.world[1] - prevGameState.world[1])):  # reset level progress max
            # reset level progress max on new level
            for _ in range(0, 5):
                pyboy.tick()  # skip frames to get updated x pos on next level

            current_mario = self.GetGameState(pyboy)

            pyboy.game_wrapper()._level_progress_max = current_mario.real_x_pos
            current_mario._level_progress_max = current_mario.real_x_pos

        if len(self.realMax) == 0:
            self.realMax.append([current_mario.world[0], current_mario.world[1], current_mario._level_progress_max])
        else:
            r = False
            for elem in self.realMax:  # fix max length
                if elem[0] == current_mario.world[0] and elem[1] == current_mario.world[1]:
                    elem[2] = current_mario._level_progress_max
                    r = True
                    break  # leave loop

            if r == False:  # this means this level does not exist
                self.realMax.append([current_mario.world[0], current_mario.world[1], current_mario._level_progress_max])

        # reward function simple
        clock = current_mario.time_left - prevGameState.time_left
        movement = current_mario.real_x_pos - prevGameState.real_x_pos
        death = -15 * (current_mario.lives_left - prevGameState.lives_left)
        levelReward = 15 * max((current_mario.world[0] - prevGameState.world[0]), (
                current_mario.world[1] - prevGameState.world[1]))  # +15 if either new level or new world

        reward = clock + death + movement + levelReward

        return reward

    def GetActions(self):
        baseActions = [WindowEvent.PRESS_ARROW_RIGHT,
                       WindowEvent.PRESS_BUTTON_A, WindowEvent.PRESS_ARROW_LEFT]

        totalActionsWithRepeats = list(itertools.permutations(baseActions, 2))
        withoutRepeats = []

        for combination in totalActionsWithRepeats:
            reversedCombination = combination[::-1]
            if (reversedCombination not in withoutRepeats):
                withoutRepeats.append(combination)

        filteredActions = [[action] for action in baseActions] + withoutRepeats

        # remove  ['PRESS_ARROW_RIGHT', 'PRESS_ARROW_LEFT']
        del filteredActions[4]

        return filteredActions

    def PrintGameState(self, pyboy):
        gameState = GameState(pyboy)
        game_wrapper = pyboy.game_wrapper()

        print("'Fake', level_progress: ", game_wrapper.level_progress)
        print("'Real', level_progress: ", gameState.real_x_pos)
        print("_level_progress_max: ", gameState._level_progress_max)
        print("World: ", gameState.world)
        print("Time respawn", pyboy.get_memory_value(0xFFA6))

    def GetGameState(self, pyboy):
        return GameState(pyboy)

    def GetHyperParameters(self) -> Config:
        config = Config()
        config.exploration_rate_decay = 0.999
        return config

    def GetLength(self, pyboy):
        result = sum([x[2] for x in self.realMax])

        pyboy.game_wrapper()._level_progress_max = 0  # reset max level progress because game hasnt implemented it
        self.realMax = []

        return result
