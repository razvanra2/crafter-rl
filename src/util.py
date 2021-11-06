import itertools
from copy import deepcopy
import torch.nn as nn
import torch.optim as O

def get_epsilon_schedule(start=1.0, end=0.1, steps=500):
    """ Returns either:
        - a generator of epsilon values
        - a function that receives the current step and returns an epsilon

        The epsilon values returned by the generator or function need
        to be degraded from the `start` value to the `end` within the number
        of `steps` and then continue returning the `end` value indefinetly.

        You can pick any schedule (exp, poly, etc.). I tested with linear decay.
    """
    eps_step = (start - end) / steps
    def frange(start, end, step):
        x = start
        while x > end:
            yield x
            x -= step
    return itertools.chain(frange(start, end, eps_step), itertools.repeat(end))

class View(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

def get_estimator(action_num, device, input_ch=4, lin_size=32):
    return nn.Sequential(
        nn.Conv2d(input_ch, 8, kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(8, 8, kernel_size=3),
        nn.ReLU(inplace=True),
        View(),
        nn.Linear(8 * 80 * 80, 8 * 80),
        nn.ReLU(inplace=True),
        nn.Linear(8 * 80, lin_size),
        nn.ReLU(inplace=True),
        nn.Linear(lin_size, action_num),
    ).to(device)