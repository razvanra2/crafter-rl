import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, in_channels, outputs):
        super(DQN, self).__init__()
        self.in_channels = in_channels
        self.outputs = outputs

        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(3136, 1024)
        self.fc2 = nn.Linear(1024, outputs)

    def forward(self, x):
        x = x.view((-1, 4, 84, 84))
        x = F.relu(self.conv1(x), inplace=True)
        x = F.relu(self.conv2(x), inplace=True)
        x = F.relu(self.conv3(x), inplace=True)
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class LinearDQN(nn.Module):
    def __init__(self, inputs, outputs):
        super(LinearDQN, self).__init__()
        self.inputs = inputs
        self.outputs = outputs

        self.fc1 = nn.Linear(inputs, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, outputs)

    def forward(self, x):
        x = F.relu(self.fc1(x), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        x = self.fc3(x)
        return x