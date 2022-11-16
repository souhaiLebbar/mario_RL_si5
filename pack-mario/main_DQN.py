from pyboy.pyboy import *
from MarioAISettings import MarioAI
from CustomPyBoyGym import CustomPyBoyGym
from wrappers import SkipFrame, ResizeObservation
from gym.wrappers import FrameStack, NormalizeObservation
from agent import DQN
gameDimensions = (20, 16)
frameStack = 4

pyboy = PyBoy("mario.gb", window_type="SDL2", window_scale=3, debug=False, game_wrapper=True)

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
STEPS = 300
# steps that copy the current_net's parameters to target_net
UPDATE_STEP = 50
BATCH_SIZE = 128

observation=env.reset()
print([observation])
agent = DQN(env.observation_space, len(filteredActions))

# # while True:
# #     actions = filteredActions[0]
# #     # Agent performs action and moves 1 frame
# #     next_observation, reward, done, info = env.step(actions)
#
for episode in range(EPISODES):
    # get the initial state
    observation = env.reset()
    for step in range(STEPS):
        # get the action by state
        action = agent.Choose_Action(observation)
        # step the env forward and get the new state
        next_state, reward, done, info = env.step(action)
        # store the data in order to update the network in future
        agent.Store_Data(observation, filteredActions, action, reward, next_state, done)
        if len(agent.replay_buffer) > BATCH_SIZE:
            agent.Train_Network(BATCH_SIZE)
        if step % UPDATE_STEP == 0:
            agent.Update_Target_Network()
        state = next_state
        if done:
            break
