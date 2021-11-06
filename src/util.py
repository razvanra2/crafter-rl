import itertools
from copy import deepcopy
import torch.nn as nn
import torch.nn.functional as F

def get_epsilon_schedule(start=1.0, end=0.1, steps=500):
    eps_step = (start - end) / steps
    def frange(start, end, step):
        x = start
        while x > end:
            yield x
            x -= step
    return itertools.chain(frange(start, end, eps_step), itertools.repeat(end))

class NnModel(nn.Module):
    def __init__(self, action_num, input_ch=4, lin_size=32):
        super(NnModel, self).__init__()

        self.conv1 = nn.Conv2d(input_ch, 8, kernel_size=3)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3)
        self.fc1 = nn.Linear(8 * 80 * 80, 8 * 80)
        self.fc2 = nn.Linear(8 * 80, lin_size)
        self.fc3 = nn.Linear(lin_size, action_num)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DuelingNnModel(nn.Module):
    def __init__(self, action_num, input_ch=4, lin_size=32):
        super(DuelingNnModel, self).__init__()

        self.conv1 = nn.Conv2d(input_ch, 8, kernel_size=3)
        self.conv2 = nn.Conv2d(8, 8, kernel_size=3)

        self.value_stream = nn.Sequential(
            nn.Linear(8 * 80 * 80, 8 * 80),
            nn.ReLU(),
            nn.Linear(8 * 80, lin_size),
            nn.ReLU(),
            nn.Linear(lin_size, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(8 * 80 * 80, 8 * 80),
            nn.ReLU(),
            nn.Linear(8 * 80, lin_size),
            nn.ReLU(),
            nn.Linear(lin_size, action_num)
        )

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)

        values = self.value_stream(x)
        advantages = self.advantage_stream(x)
        qvals = values + (advantages - advantages.mean())

        return qvals