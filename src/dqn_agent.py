from copy import deepcopy
import torch

class DQN:
    def __init__(
            self,
            estimator,
            buffer,
            optimizer,
            epsilon_schedule,
            action_num,
            gamma=0.9,
            update_steps=1,
            update_target_steps=10,
            warmup_steps=10000,
    ):
        self._estimator = estimator
        self._target_estimator = deepcopy(estimator)
        self._buffer = buffer
        self._optimizer = optimizer
        self._epsilon = epsilon_schedule
        self._action_num = action_num
        self._gamma = gamma
        self._update_steps = update_steps
        self._update_target_steps = update_target_steps
        self._warmup_steps = warmup_steps
        self._step_cnt = 0
        assert warmup_steps > self._buffer._batch_size, (
            "You should have at least a batch in the ER.")

    def act(self, state):
        with torch.no_grad():
            return self._estimator(state).argmax()

    def step(self, state):
        if self._step_cnt < self._warmup_steps:
            return torch.randint(self._action_num, (1,)).item()

        if next(self._epsilon) < torch.rand(1).item():
            with torch.no_grad():
                qvals = self._estimator(state)
                return qvals.argmax()
        else:
            return torch.randint(self._action_num, (1,)).item()

    def learn(self, state, action, reward, state_, done):
        self._buffer.push((state, action, reward, state_, done))

        if self._step_cnt < self._warmup_steps:
            self._step_cnt += 1
            return

        if self._step_cnt % self._update_steps == 0:
            batch = self._buffer.sample()
            self._update(*batch)

        if self._step_cnt % self._update_target_steps == 0:
            self._target_estimator.load_state_dict(self._estimator.state_dict())

        self._step_cnt += 1

    def _update(self, states, actions, rewards, states_, done):
        q_values = self._estimator(states)
        with torch.no_grad():
            q_values_ = self._target_estimator(states_)

        qsa = q_values.gather(1, actions)
        qsa_ = q_values_.max(1, keepdim=True)[0]

        target_qsa = rewards + self._gamma * qsa_ * (1 - done.float())

        loss = (qsa - target_qsa).pow(2).mean()

        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()