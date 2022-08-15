from imports import *

# Testing a random env with open AI gym
env = gym.make("ALE/SpaceInvaders-v5", render_mode="rgb_array") # use render_mode="human" to visualize the gameplay
height, width, channels = env.observation_space.shape
actions = env.action_space.n

# getting the action names. Understanding better
# print(env.unwrapped.get_action_meanings())

# running an env with random steps
episodes = 5
for episode in range(1, episodes+1):
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render("rgb_array")
        action = random.choice([0,1,2,3,4,5]) # total 6 actions ['NOOP', 'FIRE', 'RIGHT', 'LEFT', 'RIGHTFIRE', 'LEFTFIRE']
        n_state, reward, done, info = env.step(action) # taking the random action
        score += reward
    print("Episode: {} Score: {}".format(episode, score))
env.close()