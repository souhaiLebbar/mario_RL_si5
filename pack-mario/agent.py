import random
from asyncio import sleep
import itertools
from pyboy import WindowEvent
from AISettingsInterface import AISettingsInterface, Config

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from collections import deque
import numpy as np

#reference : https://github.com/samurasun/RL/blob/master/DQN.py


# experiences replay buffer size
REPLAY_SIZE = 10000
# discount factor for target Q to caculate the TD aim value
GAMMA = 0.9
# the start value of epsilon
INITIAL_EPSILON = 0.5
# the final value of epsilon
FINAL_EPSILON = 0.01

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




class DQN(AISettingsInterface):
    def __init__(self, observation_space, action_space):
        self.realMax = []  # [[1,1, 2500], [1,1, 200]]


        # for DQN

        self.state_dim1 = observation_space.shape[0]

        self.state_dim2 = observation_space.shape[1]

        self.state_dim3 = observation_space.shape[2]
        # the action is the output vector and it has two dimensions
        self.action_dim = action_space
        # init experience replay, the deque is a list that first-in & first-out
        self.replay_buffer = deque()
        # you can create the network by the two parameters
        self.create_Q_network()
        # after create the network, we can define the training methods
        self.create_updating_method()
        # set the value in choose_action
        self.epsilon = INITIAL_EPSILON
        # Init session
        self.session = tf.InteractiveSession()
        self.session.run(tf.global_variables_initializer())



    def create_Q_network(self):


        #问题在这个函数/error here


        # first, set the input of networks
        self.state_input = tf.placeholder("float", [None, self.state_dim1, self.state_dim2, self.state_dim3])
        # second, create the current_net
        with tf.variable_scope('current_net'):
            # first, set the network's weights
            W1 = self.weight_variable([self.state_dim3, self.state_dim2,self.state_dim1, 50])
            b1 = self.bias_variable([50])
            W2 = self.weight_variable([50, 20])
            b2 = self.bias_variable([20])
            W3 = self.weight_variable([20, self.action_dim])
            b3 = self.bias_variable([self.action_dim])
            # second, set the layers
            # hidden layer one
            h_layer_one = tf.nn.relu(tf.matmul(self.state_input, W1) + b1)
            # hidden layer two
            h_layer_two = tf.nn.relu(tf.matmul(h_layer_one, W2) + b2)
            # the output of current_net
            self.Q_value = tf.matmul(h_layer_two, W3) + b3
        # third, create the current_net
        with tf.variable_scope('target_net'):
            # first, set the network's weights
            t_W1 = self.weight_variable([self.state_dim3, self.state_dim2,self.state_dim1, 50])
            t_b1 = self.bias_variable([50])
            t_W2 = self.weight_variable([50, 20])
            t_b2 = self.bias_variable([20])
            t_W3 = self.weight_variable([20, self.action_dim])
            t_b3 = self.bias_variable([self.action_dim])
            # second, set the layers
            # hidden layer one
            t_h_layer_one = tf.nn.relu(tf.matmul(self.state_input, t_W1) + t_b1)
            # hidden layer two
            t_h_layer_two = tf.nn.relu(tf.matmul(t_h_layer_one, t_W2) + t_b2)
            # the output of current_net
            self.target_Q_value = tf.matmul(t_h_layer_two, t_W3) + t_b3
        # at last, solve the parameters replace problem
        # the parameters of current_net
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='current_net')
        # the parameters of target_net
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        # define the operation that replace the target_net's parameters by current_net's parameters
        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

    # the function that give the weight initial value
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape)
        return tf.Variable(initial)

    # the function that give the bias initial value
    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial)

    # this the function that define the method to update the current_net's parameters
    def create_updating_method(self):
        # this the input action, use one hot presentation
        self.action_input = tf.placeholder("float", [None, self.action_dim])
        # this the TD aim value
        self.y_input = tf.placeholder("float", [None])
        # this the action's Q_value
        Q_action = tf.reduce_sum(tf.multiply(self.Q_value, self.action_input), reduction_indices=1)
        # this is the lost
        self.cost = tf.reduce_mean(tf.square(self.y_input - Q_action))
        # use the loss to optimize the network
        self.optimizer = tf.train.AdamOptimizer(0.0001).minimize(self.cost)

    # this is the function that use the network output the action
    def Choose_Action(self, state):
        # the output is a tensor, so the [0] is to get the output as a list

        #如果更改create_Q_network（），添加input——state维度，此处代码出现问题
        Q_value = self.Q_value.eval(feed_dict={
            self.state_input: [state]
        })[0]
        # use epsilon greedy to get the action
        if random.random() <= self.epsilon:
            # if lower than epsilon, give a random value
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            return random.randint(0, self.action_dim - 1)
        else:
            # if bigger than epsilon, give the argmax value
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            return np.argmax(Q_value)

    # this the function that store the data in replay memory
    def Store_Data(self, state, all, action, reward, next_state, done):
        # generate a list with all 0,and set the action is 1
        one_hot_action = np.zeros(self.action_dim)
        # place=all.index(action)
        one_hot_action[action] = 1
        # store all the elements
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
        # if the length of replay_buffer is bigger than REPLAY_SIZE
        # delete the left value, make the len is stable
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()

    # train the network, update the parameters of Q_value
    def Train_Network(self, BATCH_SIZE):
        # Step 1: obtain random minibatch from replay memory
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]

        # Step 2: calculate TD aim value
        y_batch = []
        # give the next_state_batch flow to target_Q_value and caculate the next state's Q_value
        Q_value_batch = self.target_Q_value.eval(feed_dict={self.state_input: next_state_batch})
        # caculate the TD aim value by the formulate
        for i in range(0, BATCH_SIZE):
            done = minibatch[i][4]
            if done:
                y_batch.append(reward_batch[i])
            else:
                # the Q value caculate use the max directly
                y_batch.append(reward_batch[i] + GAMMA * np.max(Q_value_batch[i]))

        # step 3: update the network
        self.optimizer.run(feed_dict={
            self.y_input: y_batch,
            self.action_input: action_batch,
            self.state_input: state_batch
        })

    def Update_Target_Network(self):
        # update target Q netowrk
        self.session.run(self.target_replace_op)


#old codes
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