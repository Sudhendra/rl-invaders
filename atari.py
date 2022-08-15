from imports import *

# Testing a random env with open AI gym
env = gym.make("ALE/SpaceInvaders-v5")
height, width, channels = env.observation_space.shape
actions = env.action_space.n

# getting the action names. Understanding better
print(env.unwrapped.get_action_meanings())