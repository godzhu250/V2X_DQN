import math
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    def __init__(self):
        self.state_dim = config.STATE_DIM
        self.action_dim = config.ACTION_DIM

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net = QNetwork(self.state_dim, self.action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=config.LEARNING_RATE)
        self.loss_fn = nn.MSELoss()

        self.memory = deque(maxlen=config.MEMORY_SIZE)

        # Step-based epsilon schedule state.
        self.steps_done = 0
        self.current_epsilon = float(config.EPS_START)
        self.epsilon = self.current_epsilon

    def _compute_epsilon(self):
        eps = config.EPS_END + (config.EPS_START - config.EPS_END) * math.exp(
            -1.0 * self.steps_done / max(1, config.EPS_DECAY)
        )
        return float(eps)

    def select_action(self, state, is_training=True):
        if is_training:
            epsilon = self._compute_epsilon()
            self.current_epsilon = epsilon
            self.epsilon = epsilon
            self.steps_done += 1

            if random.random() < epsilon:
                return random.randrange(self.action_dim)

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self):
        if len(self.memory) < config.BATCH_SIZE:
            return 0.0

        batch = random.sample(self.memory, config.BATCH_SIZE)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        state_batch = torch.FloatTensor(np.array(state_batch)).to(self.device)
        action_batch = torch.LongTensor(action_batch).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(next_state_batch)).to(self.device)
        done_batch = torch.FloatTensor(done_batch).unsqueeze(1).to(self.device)

        q_expected = self.policy_net(state_batch).gather(1, action_batch)

        with torch.no_grad():
            q_next = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
            q_target = reward_batch + (config.GAMMA * q_next * (1 - done_batch))

        loss = self.loss_fn(q_expected, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path))
