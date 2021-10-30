import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from copy import deepcopy

from IPython.display import clear_output

import gym
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()

from itertools import count
from typing import Callable

from src.buffer import ReplayBuffer
from src.util import select_epsilon_greedy_action, eps_generator
import matplotlib.animation as animation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def eps_generator(max_eps: float=1.0, min_eps: float=0.02, max_iter: int = 1e6):
    crt_iter = -1

    while True:
        crt_iter += 1
        frac = min(crt_iter/max_iter, 1)
        eps = (1 - frac) * max_eps + frac * min_eps
        yield eps

def select_epilson_greedy_action(Q: nn.Module, s: Tensor, eps: float):
    rand = np.random.rand()
    if rand < eps:
        return np.random.choice(np.arange(Q.num_actions))

    with torch.no_grad():
        output = Q(s).argmax(dim=1).item()

    return output

class DdqnAgent:
    def __init__(self, action_num) -> None:
        self.action_num = action_num

        self.policy = torch.distributions.Categorical(
            torch.ones(action_num) / action_num
        )

    @torch.no_grad()
    def ddqn_target(
        self,
        Q: nn.Module,
        target_Q: nn.Module,
        r_batch: Tensor,
        s_next_batch: Tensor,
        done_batch: Tensor,
        gamma: float) -> Tensor:

        next_max_Q = target_Q(s_next_batch).gather(1, Q(s_next_batch).argmax(dim=1, keepdim=True))
        next_max_Q = next_max_Q.view(-1).detach()
        next_Q_values = (1 - done_batch) * next_max_Q

        return r_batch + (gamma * next_Q_values)

    def learn(self,
        env: gym.Env,
        Network: nn.Module,
        eval: Callable,
        eval_env: gym.Env,
        opt,
        batch_size: int = 128,
        gamma: float = 0.9,
        replay_buffer_size=10000,
        learning_starts: int = 1000,
        learning_freq: int = 5,
        target_update_freq: int = 100,
        log_every: int = 100,
        max_allowed_steps: float = 10000,
        ):

        target_function = self.ddqn_target
        input_arg = env.observation_space.shape[0]
        num_actions = env.action_space.n
        Q = Network(input_arg, num_actions).to(device)
        target_Q = Network(input_arg, num_actions).to(device)
        optimizer = optim.Adam(Q.parameters(), lr=1e-4)
        criterion = nn.MSELoss()
        replay_buffer = ReplayBuffer(replay_buffer_size)
        eps_scheduler = iter(eps_generator())
        total_steps = 0
        num_param_updates = 0

        episode_scores = []
        episode_stddev = []
        all_qs_mean = []
        frames = []

        s = env.reset()

        for _ in count():
            total_steps += 1
            frames.append(env.render())

            if total_steps > max_allowed_steps:
                break

            if total_steps > learning_starts:
                eps = next(eps_scheduler)
                s = torch.tensor(s, device=device).float()
                a = select_epsilon_greedy_action(Q, s.unsqueeze(dim=0), eps)
            else:
                a = np.random.choice(np.arange(num_actions))

            s_next, r, done, _ = env.step(a)

            replay_buffer.push(
                Tensor(s).to(device=device) if type(s) != Tensor else s,
                int(a),
                r,
                Tensor(s_next).to(device=device) if type(s_next) != Tensor else s_next,
                int(done))

            if done and total_steps > learning_starts:
                s = env.reset()


            s = s_next
            if (total_steps > learning_starts and total_steps % learning_freq == 0):
                batch = replay_buffer.sample(batch_size)

                s_batch = torch.stack([getattr(x, 'state') for x in batch]).float().to(device=device)
                a_batch = torch.tensor([getattr(x, 'action') for x in batch], dtype=torch.long, device=device)
                r_batch = torch.tensor([getattr(x, 'reward') for x in batch], dtype=torch.float32, device=device)
                s_next_batch = torch.stack([getattr(x, 'next_state') for x in batch]).float().to(device=device)
                done_batch = torch.tensor([getattr(x, 'done') for x in batch], dtype=torch.long, device=device)

                Q_values = Q(s_batch).gather(1, a_batch.unsqueeze(1)).view(-1)

                target_Q_values = target_function(Q, target_Q, r_batch, s_next_batch, done_batch, gamma)

                loss = criterion(target_Q_values, Q_values)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                num_param_updates += 1

                if num_param_updates % target_update_freq == 0:
                    target_Q.load_state_dict(Q.state_dict())


            if total_steps % log_every == 0:
                eval(self, eval_env, total_steps, opt)

        return episode_scores, episode_stddev, all_qs_mean, frames



    def act(self, observation):
        """ Since this is a random agent the observation is not used."""
        return self.policy.sample().item()