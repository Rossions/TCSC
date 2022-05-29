import torch
from torch import nn
from torch.nn import functional as F


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, phase_action_dim, duration_action_dim):
        super(PolicyNetwork, self).__init__()
        self.phase_action_dim = phase_action_dim
        self.duration_action_dim = duration_action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, action_dim))

    def forward(self, state):
        phase_action = self.net(state)[:, :self.phase_action_dim]
        phase_action = F.softmax(phase_action, dim=1)
        duration_action = self.net(state)[:, self.phase_action_dim:]
        return phase_action, duration_action


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, phase_action_dim, duration_action_dim):
        super(ValueNetwork, self).__init__()
        input_dim = state_dim + phase_action_dim + duration_action_dim
        self.net_one = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1))
        self.net_two = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1))

    def forward(self, input_tensor):
        net_one = self.net_one(input_tensor)
        net_two = self.net_two(input_tensor)
        return net_one, net_two


if __name__ == '__main__':
    state_dim = 3
    action_dim = 6
    phase_action_dim = 5
    duration_action_dim = 1

    state = torch.randn(1, state_dim)
    policy_net = PolicyNetwork(state_dim, action_dim, phase_action_dim, duration_action_dim)
    value_net = ValueNetwork(state_dim, phase_action_dim, duration_action_dim)

    phase_action, duration_action = policy_net(state)
    print(phase_action)
    print(duration_action)

    value = value_net(torch.cat((state, phase_action, duration_action), dim=1))
    print(value)
