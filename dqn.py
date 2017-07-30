# coding=utf-8
from keras.models import Sequential, load_model
from keras.layers import Dense
import numpy as np
import random
from collections import deque

BATCH_SIZE = 64
GAMMA = 0.9
EPSILON = 0.5


class DQN:
    def __init__(self, env):
        self.replay = deque(maxlen=10000)
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n
        self.epsilon = EPSILON

        self.create_model()

    def create_model(self):
        self.model = Sequential()
        self.model.add(Dense(20, activation='relu', input_shape=(self.state_dim,)))
        self.model.add(Dense(20, activation='relu'))
        self.model.add(Dense(self.action_dim, activation='linear'))
        self.model.compile(optimizer='Adam', loss='mse')

    def perceive(self, state, action, reward, next_state, done):
        self.replay.append((state, action, reward, next_state, done))
        if len(self.replay) > BATCH_SIZE:
            self.train_Q_network()

    def train_Q_network(self):
        minbatch = random.sample(self.replay, BATCH_SIZE)
        state_batch = np.vstack([i[0] for i in minbatch])
        action_batch = [i[1] for i in minbatch]
        reward_batch = [i[2] for i in minbatch]
        next_state_batch = np.vstack([i[3] for i in minbatch])
        y_batch = []
        Q_value_batch = self.model.predict(next_state_batch)

        for i in range(BATCH_SIZE):
            if minbatch[i][4]:
                reward = reward_batch[i]
            else:
                reward = reward_batch[i] + GAMMA * np.max(Q_value_batch[i])
            target = [0 for _ in range(self.action_dim)]
            target[action_batch[i]] = reward
            y_batch.append(target)

        self.model.train_on_batch(state_batch, y_batch)

    def e_action(self, state):
        Q_value = self.model.predict(state)
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        else:
            return np.argmax(Q_value)
        self.epsilon -= (EPSILON - 0.01) / 10000

    def action(self, state):
        return np.argmax(self.model.predict(state))

    def load(self, model):
        self.model = load_model(model)
