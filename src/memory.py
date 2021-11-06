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
        s, a, r, s_, d = zip(*random.sample(self._buffer, self._batch_size))

        return (
            torch.cat(s, 0).to(self.device),
            torch.tensor(a, dtype=torch.int64).unsqueeze(1).to(self.device),
            torch.tensor(r, dtype=torch.float32).unsqueeze(1).to(self.device),
            torch.cat(s_, 0).to(self.device),
            torch.tensor(d, dtype=torch.uint8).unsqueeze(1).to(self.device)
        )

    def __len__(self):
        return len(self._buffer)