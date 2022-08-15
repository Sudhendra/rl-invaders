# The deep learning model
# func: learns from the env.observation_space pixels
# goal: create a filter which can detect objects in the game. ex: where the enemy pixel is, where the mther ship is etc.

from imports import *
from random_play import height, width, channels, actions

def build_model(height, width, channels, actions):
    model = Sequential() # creating a Sequential structure
    model.add(Convolution2D(32, (8,8), strides=(4,4), activation='relu', input_shape=(3, height, width, channels)))
    model.add(Convolution2D(64, (4,4), strides=(2,2), activation='relu'))
    model.add(Convolution2D(64, (3,3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(actions, activation='linear')) # output layer
    return model

# using height, width, channels and actions from env created in random_play.py
model = build_model(height, width, channels, actions)
print(model.summary())

