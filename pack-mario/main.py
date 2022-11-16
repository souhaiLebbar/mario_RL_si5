from pyboy.pyboy import *
from MarioAISettings import MarioAI
from CustomPyBoyGym import CustomPyBoyGym
from wrappers import SkipFrame, ResizeObservation
from gym.wrappers import FrameStack, NormalizeObservation
from agent import DQN
gameDimensions = (20, 16)
frameStack = 4

pyboy = PyBoy("mario.gb", window_type="SDL2", window_scale=3, debug=False, game_wrapper=True)
aiSettings = MarioAI()
env = CustomPyBoyGym(pyboy, observation_type="tiles")
env.setAISettings(aiSettings)  # use this settings
env = SkipFrame(env, skip=4)
env = ResizeObservation(env, gameDimensions)  # transform MultiDiscreate to Box for framestack
env = NormalizeObservation(env)  # normalize the values
env = FrameStack(env, num_stack=frameStack)

pyboy.set_emulation_speed(0)

observation = env.reset()
filteredActions = aiSettings.GetActions()  # get possible actions


while True:
    actions = filteredActions[0]
    # Agent performs action and moves 1 frame
    next_observation, reward, done, info = env.step(actions)

