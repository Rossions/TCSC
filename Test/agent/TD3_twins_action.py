import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, phase_action_dim, duration_action_dim, max_action):
        super(PolicyNetwork, self).__init__()
        self.phase_action_dim = phase_action_dim
        self.duration_action_dim = duration_action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim))
        self.max_action = max_action

    def forward(self, state):
        phase_action = self.net(state)[:self.phase_action_dim]
        phase_action = torch.softmax(phase_action, dim=-1)
        duration_action = torch.tanh(self.net(state)[self.phase_action_dim:]) * self.max_action
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

    def forward_net_one(self, input_tensor):
        net_one = self.net_one(input_tensor)
        return net_one


class TD3_twins_action(object):
    def __init__(self, env_with_dead, state_dim, action_dim, phase_action_dim, duration_action_dim, max_action,
                 min_action,
                 gamma=0.99,
                 a_lr=1e-4, c_lr=1e-4, Q_batchsize=256):
        self.actor = PolicyNetwork(state_dim, action_dim, phase_action_dim, duration_action_dim, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=a_lr)
        self.actor_target = copy.deepcopy(self.actor)

        self.q_critic = ValueNetwork(state_dim, phase_action_dim, duration_action_dim).to(device)
        self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=c_lr)
        self.q_critic_target = copy.deepcopy(self.q_critic)

        self.env_with_Dead = env_with_dead
        self.action_dim = action_dim
        self.max_action = max_action
        self.min_action = min_action
        self.gamma = gamma
        self.policy_noise = 0.2 * max_action
        self.noise_clip_max = 0.5 * max_action
        self.noise_clip_min = 0.5 * min_action
        self.tau = 0.005
        self.Q_batchsize = Q_batchsize
        self.delay_counter = -1
        self.delay_freq = 1

    def select_action(self, state):  # only used when interact with the env
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            phase_action, duration_action = self.actor(state)
        return phase_action.cpu().numpy().flatten(), duration_action.cpu().numpy().flatten()

    def train(self, replay_buffer):
        self.delay_counter += 1
        with torch.no_grad():
            s, p_a, d_a, r, s_prime, dead_mask = replay_buffer.sample(self.Q_batchsize)
            noise = (torch.randn_like(d_a) * self.policy_noise).clamp(self.noise_clip_min, self.noise_clip_max)
            _, d_a_noise = self.actor_target(s_prime)
            smoothed_target_a = (d_a_noise + noise).clamp(self.min_action, self.max_action)
            # Noisy on target action

        # Compute the target Q value
        target_Q1, target_Q2 = self.q_critic_target(s_prime, p_a, smoothed_target_a)
        target_Q = torch.min(target_Q1, target_Q2)
        '''DEAD OR NOT'''
        if self.env_with_Dead:
            target_Q = r + (1 - dead_mask) * self.gamma * target_Q  # env with dead
        else:
            target_Q = r + self.gamma * target_Q  # env without dead

        # Get current Q estimates
        current_Q1, current_Q2 = self.q_critic(s, p_a, d_a)

        # Compute critic loss
        q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        # Optimize the q_critic
        self.q_critic_optimizer.zero_grad()
        q_loss.backward()
        self.q_critic_optimizer.step()

        if self.delay_counter == self.delay_freq:
            # Update Actor
            a_loss = -self.q_critic.Q1(s, self.actor(s)).mean()
            self.actor_optimizer.zero_grad()
            a_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.delay_counter = -1

    def save(self, episode):
        torch.save(self.actor.state_dict(), "ppo_actor{}.pth".format(episode))
        torch.save(self.q_critic.state_dict(), "ppo_q_critic{}.pth".format(episode))

    def load(self, episode):
        self.actor.load_state_dict(torch.load("ppo_actor{}.pth".format(episode), map_location='cpu'))

        self.q_critic.load_state_dict(torch.load("ppo_q_critic{}.pth".format(episode), map_location='cpu'))


if __name__ == '__main__':
    state_dim = 3
    action_dim = 6
    phase_action_dim = 5
    duration_action_dim = 1
    max_action = 1
    state = torch.randn(1, state_dim)
    policy_net = PolicyNetwork(state_dim, action_dim, phase_action_dim, duration_action_dim, max_action)
    value_net = ValueNetwork(state_dim, phase_action_dim, duration_action_dim)

    phase_action, duration_action = policy_net(state)
    print(phase_action)
    print(duration_action)

    value = value_net(torch.cat((state, phase_action, duration_action), dim=1))
    print(value)
