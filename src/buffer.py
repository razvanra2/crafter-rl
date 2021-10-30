import random
from typing import List
from collections import deque, namedtuple

Transition = namedtuple('Transition', ('state', 'action',  'reward', 'next_state', 'done'))


class ReplayBuffer(object):
    def __init__(self, capacity: int):
        """
        Constructor

        Parameters
        ----------
        capacity
            Replay buffer capacity
        """
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size) -> List[Transition]:
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)