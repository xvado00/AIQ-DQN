import random
import binascii
import os
from collections import namedtuple, deque

Transition = namedtuple(
    'Transition',
    ('state', 'action', 'next_state', 'reward')
)


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
        self.id = binascii.b2a_hex(os.urandom(8))
        self.count = 0

    def push(self, *args):
        transition = Transition(*args)
        self.memory.append(transition)
        self.count += 1

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def clear(self):
        self.memory.clear()

    def __len__(self):
        return len(self.memory)

    def __str__(self):
        return list(self.memory)
