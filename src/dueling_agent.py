from src.dqn_agent import DQN

from copy import deepcopy
import torch

class DuelingDQN(DQN):
    def __init__(self, estimator, buffer, optimizer, epsilon_schedule, action_num, gamma=0.9, update_steps=1, update_target_steps=10, warmup_steps=10000):
        super().__init__(estimator, buffer, optimizer, epsilon_schedule, action_num, gamma=gamma, update_steps=update_steps, update_target_steps=update_target_steps, warmup_steps=warmup_steps)
        self.loss = torch.nn.MSELoss()

    def _update(self, states, actions, rewards, states_, done):
        
        curr_Q = self._estimator.forward(states).gather(1, actions)
        curr_Q = curr_Q.squeeze(1)
        next_Q = self._estimator.forward(states_)
        max_next_Q = torch.max(next_Q, 1)[0]
        expected_Q = rewards.squeeze(1) + self._gamma * max_next_Q

        loss = self.loss(curr_Q, expected_Q)

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()