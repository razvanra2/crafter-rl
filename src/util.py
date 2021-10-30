import numpy as np
import random
import torch
import torch.nn as nn
import torch.tensor as Tensor
from typing import Union, Tuple


def eps_generator(start_eps: float = 1.0, end_eps: float = 0.1, plateau: int = 1e4):
    """
    Epsilon scheduler
    
    Parameters
    ----------
    start_eps: 
        Epsilon starting value
    end_eps: 
        Epsilon ending value
    plateau: 
        Iteration to plateau epsilon to end_eps

    Returns
    -------
    Linear decayed epsilon
    """
    crt_iter = -1

    while True:
        crt_iter += 1
        frac = min(crt_iter / plateau, 1)
        eps = (1 - frac) * start_eps + frac * end_eps
        yield eps


def select_epsilon_greedy_action(
        Q: nn.Module,
        s: Tensor,
        eps: float,
        with_val: bool = False) -> Union[int, Tuple[int, float]]:
    """
    Select epsilon greedy action

    Parameters
    ----------
    Q
        Q network
    s
        State tensor
    eps
        Epsilon value

    Returns
    -------
    Epsilon greedy action
    """
    rand = np.random.rand()

    # compute Q-vals
    with torch.no_grad():
        Q_vals = Q(s)

    qval, act = Q_vals.max(dim=1)
    qval, act = qval.item(), act.item()

    # with prob eps select a random action
    if rand < eps:
        act = np.random.choice(np.arange(Q.outputs))

    return (act, qval) if with_val else act


def set_seed(seed: int = 13):
    """
    Sets a seed to ensure reproducibility
    Parameters
    ----------
    seed
        seed to be set
    """

    # torch related
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # others
    np.random.seed(seed)
    random.seed(seed)
