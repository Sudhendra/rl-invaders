# Building RL agent with keras

from imports import *
from dl_model import model
from random_play import actions, env

def build_agent(model, actions):
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.2, nb_steps=10000)
    memory = SequentialMemory(limit=1000, window_length=3) # to increase or dec windowed period, need to change input_shape in your DL model to match it.
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                    enable_dueling_network=True, dueling_type='avg',
                    nb_actions=actions, nb_steps_warmup=10000
                    )
    return dqn

dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-4))
dqn.fit(env, nb_steps=10000, visualize=False, verbose=2)

