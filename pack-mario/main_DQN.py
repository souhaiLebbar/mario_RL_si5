from pyboy.pyboy import *
from MarioAISettings import MarioAI
from CustomPyBoyGym import CustomPyBoyGym
from wrappers import SkipFrame, ResizeObservation
from gym.wrappers import FrameStack, NormalizeObservation
from agent import DQN
import numpy as np
import tensorflow.compat.v1 as tf
import copy




tf.disable_v2_behavior()
gameDimensions = (20, 16)
frameStack = 4

# pyboy = PyBoy("mario.gb", window_type="SDL2", window_scale=3, debug=False, game_wrapper=True)
pyboy = PyBoy("mario.gb", window_type="headless", window_scale=3, debug=False, game_wrapper=True)
env = CustomPyBoyGym(pyboy, observation_type="tiles")
aiSettings = MarioAI()
env.setAISettings(aiSettings)  # use this settings
env = SkipFrame(env, skip=4)
env = ResizeObservation(env, gameDimensions)  # transform MultiDiscreate to Box for framestack
env = NormalizeObservation(env)  # normalize the values
env = FrameStack(env, num_stack=frameStack)

pyboy.set_emulation_speed(0)

# get possible actions
filteredActions = aiSettings.GetActions()
# filteredActions= [(3,0), (5,0), (4,0), (3, 5), (5, 4)]

EPISODES=1000
BATCH_SIZE = 128
# print(env.observation_space.shape)
observation=env.reset()
print(env.action_space)
agent = DQN(env.observation_space, len(filteredActions),env)

# # while True:
# #     actions = filteredActions[0]
# #     # Agent performs action and moves 1 frame
# #     next_observation, reward, done, info = env.step(actions)


#save model and check for existing models
folder = os.getcwd()
imageList = os.listdir(folder)
for item in imageList:
    if os.path.isfile(os.path.join(folder, item)):
        if item == 'dqn.h5':
            tf.keras.models.load_model('dqn.h5')

    for e in range(EPISODES):
        state = env.reset()

        for time_t in range(5000):

            actions = agent.act(state)

            action=filteredActions[actions]
            _next_state, _reward, _done, _ = env.step(action)

            _reward = -100 if _done else _reward

            agent.save_exp(state, actions, _reward, _next_state, _done)

            # train
            state = copy.deepcopy(_next_state)

            # if done is true
            if _done:

                print("episode: {}/{}, score: {}"
                      .format(e, EPISODES, time_t))
                break

        agent.train_exp(32)

        if (e + 1) % 10 == 0:
            print("saving model")
            agent.model.save('dqn.h5')
