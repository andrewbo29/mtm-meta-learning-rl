import random
from collections import namedtuple, deque

import torch

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done', 'success'))


class ReplayMemory(object):

    def __init__(self, device="cpu", capacity=10000):
        self.device = device
        self.memory = deque([], maxlen=capacity)
        self.state = None
        self.action = None
        self.next_state = None
        self.reward = None
        self.done = None
        self.success = None

    def __del__(self):
        del self.device
        del self.state, self.action, self.next_state, self.reward, self.done, self.success

    def __len__(self):
        return len(self.success)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def get_states(self):
        states = []
        for t in self.memory:
            states.append(t.state)
        states = torch.stack(states).to(self.device)
        return states

    def get_actions(self):
        actions = []
        for t in self.memory:
            actions.append(t.action)
        actions = torch.stack(actions).to(self.device)
        return actions

    def get_next_states(self):
        next_states = []
        for t in self.memory:
            next_states.append(t.next_state)
        next_states = torch.stack(next_states).to(self.device)
        return next_states

    def get_rewards(self):
        rewards = []
        for t in self.memory:
            rewards.append(t.reward)
        rewards = torch.stack(rewards).to(self.device)
        return rewards

    def get_dones_success(self):
        dones = []
        success = []
        for t in self.memory:
            dones.append(t.done)
            if t.done or (not t.done and t.success):
                success.append(1 if t.success else 0)
        dones = torch.stack(dones).to(self.device)
        success = torch.tensor(success).to(self.device)
        return dones, success

    def to_tensor(self):
        self.state = self.get_states()
        self.action = self.get_actions()
        self.next_state = self.get_next_states()
        self.reward = self.get_rewards()
        self.done, self.success = self.get_dones_success()
        del self.memory
