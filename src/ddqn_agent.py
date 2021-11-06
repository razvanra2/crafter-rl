from src.dqn_agent import DQN

from copy import deepcopy
import torch

class DoubleDQN(DQN):
    def _update(self, states, actions, rewards, states_, done):
        with torch.no_grad():
            actions_ = self._estimator(states_).argmax(1, keepdim=True)
            q_values_ = self._target_estimator(states_)
        q_values = self._estimator(states)

        qsa = q_values.gather(1, actions)
        qsa_ = q_values_.gather(1, actions_)

        target_qsa = rewards + self._gamma * qsa_ * (1 - done.float())

        loss = (qsa - target_qsa).pow(2).mean()

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()