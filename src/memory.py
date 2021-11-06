from collections import deque
import random
import torch

class ReplayMemory:
    def __init__(self, device, size=1000, batch_size=32):
        self._buffer = deque(maxlen=size)
        self._batch_size = batch_size
        self.device = device

    def push(self, transition):
        self._buffer.append(transition)

    def sample(self):
        """ Sample from self._buffer

            Should return a tuple of tensors of size:
            (
                states:     N * (C*K) * H * W,  (torch.uint8)
                actions:    N * 1, (torch.int64)
                rewards:    N * 1, (torch.float32)
                states_:    N * (C*K) * H * W,  (torch.uint8)
                done:       N * 1, (torch.uint8)
            )

            where N is the batch_size, C is the number of channels = 3 and
            K is the number of stacked states.
        """
        # sample
        s, a, r, s_, d = zip(*random.sample(self._buffer, self._batch_size))

        # reshape, convert if needed, put on device (use torch.to(DEVICE))
        return (
            torch.cat(s, 0).to(self.device),
            torch.tensor(a, dtype=torch.int64).unsqueeze(1).to(self.device),
            torch.tensor(r, dtype=torch.float32).unsqueeze(1).to(self.device),
            torch.cat(s_, 0).to(self.device),
            torch.tensor(d, dtype=torch.uint8).unsqueeze(1).to(self.device)
        )

    def __len__(self):
        return len(self._buffer)