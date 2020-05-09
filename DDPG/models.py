import random
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


class Memory:
    def __init__(self, max_len, batch_size):
        self.buffer = deque(maxlen=max_len)
        self.batch_size = batch_size

    def push(self, state, action, reward, new_state):
        memory = (state, action, reward, new_state)
        self.buffer.append(memory)

    def get_batch(self):
        batch = random.sample(self.buffer, self.batch_size)
        states = []
        actions = []
        rewards = []
        new_states = []
        for (state, action, reward, new_state) in batch:
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            new_states.append(new_state)

        return states, actions, rewards, new_states


class Actor(nn.Module):
    def __init__(self, num_actions):
        super(Actor, self).__init__()
        self.num_actions = num_actions
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(3200, 256)
        self.linear2 = nn.Linear(256, 64)
        self.linear3 = nn.Linear(64, num_actions)

    def forward(self, state):
        x = self.pool(F.relu(self.conv1(state)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(state.shape[0], 3200)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))

        return x


class Critic(nn.Module):
    def __init__(self, num_actions):
        super(Critic, self).__init__()
        self.num_actions = num_actions
        self.conv1 = nn.Conv2d(1, 8, 3)
        self.conv2 = nn.Conv2d(8, 16, 3)
        self.conv3 = nn.Conv2d(16, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.linear1 = nn.Linear(3200, 256)
        self.linear2 = nn.Linear(256, 64)
        self.linear3 = nn.Linear(64, 3*num_actions)
        self.linear4 = nn.Linear(4*num_actions, num_actions)

    def forward(self, state, action):
        x = self.pool(F.relu(self.conv1(state)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(state.shape[0], 3200)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        x = torch.cat([x, action], 1)
        x = self.linear4(x)

        return x
