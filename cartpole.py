# coding=utf-8

import dqn
import gym
import numpy as np
from keras.models import save_model

env = gym.make('CartPole-v0')
agent = dqn.DQN(env)
flag = 0
#agent.load('2.h5')

for i in range(10000):
    state = env.reset()
    for s in range(400):
        state = np.reshape(state, (1, agent.state_dim))
        action = agent.e_action(state)
        next_state, reward, done, _ = env.step(action)
        reward = -1 if done else 0.1
        agent.perceive(state, action, reward, next_state, done)
        state = next_state
        if done:
            break

    if i % 100 == 0:
        t_reward = 0
        for j in range(10):
            state = env.reset()
            while True:
                state = np.reshape(state, (1, agent.state_dim))
                action = agent.action(state)
                env.render()
                state, reward, done, _ = env.step(action)
                t_reward += reward
                if done:
                    break
        print(i, t_reward / 10, len(agent.replay))
        if t_reward / 10 >= 250:
            break
    if t_reward / 10 > flag:
        flag = t_reward / 10
        save_model(agent.model, 'cartpole.h5')
