import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from collections import namedtuple
from torch.distributions import Categorical


class DQN(torch.nn.Module):
    def __init__(self, channels, height, width, outputs):
        super(DQN, self).__init__()
        self.pre_head_dim = 16  # 32
        k_size = 1

        self.cnv_net = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=16, kernel_size=k_size, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=k_size, stride=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=k_size, stride=1),
            nn.ReLU()
        )

        def cnv_size(size, kernel_size=k_size, stride=1):
            return (size - (kernel_size - 1) - 1) // stride + 1

        out_w = cnv_size(cnv_size(cnv_size(width)))
        out_h = cnv_size(cnv_size(cnv_size(height)))
        self.dense_layer_size = (out_w * out_h * 64)

        self.head = nn.Linear(self.dense_layer_size, outputs)


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class A2C(DQN):
    def __init__(self, *args, **kwargs):
        super(A2C, self).__init__(*args, **kwargs)
        self.value_head = nn.Linear(self.dense_layer_size, 1)

    def forward(self, x):
        x = self.cnv_net(x)
        x = x.view(x.size(0), -1)
        return f.softmax(self.head(x), dim=-1), self.value_head(x)  # added exploration factor (0.1)


class A2CAgent(object):
    def __init__(self,
                 observation_shape: [int],
                 number_of_actions: int,
                 gamma: float = 0.99,
                 mini_batch_size: int = 32,
                 seed: int = 42) -> None:
        torch.manual_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.gamma = gamma
        self.number_of_actions = number_of_actions
        self.mini_batch_size = mini_batch_size
        self.policy_net = A2C(observation_shape[0],
                              observation_shape[1],
                              observation_shape[2],
                              number_of_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=0.0001)  # decreased learning rate
        self.saved_actions = []
        self.rewards = []

    def save_probs(self, log_prob, value):
        self.saved_actions.append((log_prob, value))

    def save(self, state, action, next_state, reward, done):
        self.rewards.append(reward)

    def train(self):
        R = np.array(0)
        policy_losses = []
        value_losses = []
        returns = []
        eps = np.finfo(np.float32).eps.item()

        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, device=self.device, dtype=torch.float)
        returns = (returns - returns.mean()) / (returns.std() + eps)

        for (log_prob, value), R in zip(self.saved_actions, returns):
            advantage = (R - value.item())
            policy_losses.append(-log_prob * advantage)
            target = torch.tensor([R], device=self.device)
            value_losses.append(f.mse_loss(value, target))

        self.optimizer.zero_grad()
        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        loss.backward()
        self.optimizer.step()
        self.saved_actions = []
        self.rewards = []

    def policy(self, state):

        probs, state_value = self.policy_net(
            torch.tensor([state], device=self.device, dtype=torch.float32)
        )

        m = Categorical(probs)
        action = m.sample()

        self.save_probs(m.log_prob(action), state_value)

        return action.item()

    def save_weights(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load_weights(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location='cpu'))
        self.policy_net.eval()


def make_a2c_agent(observation_shape, number_of_actions, seed):

    agent = A2CAgent(observation_shape=observation_shape,
                     number_of_actions=number_of_actions,
                     gamma=1.0,
                     mini_batch_size=32,
                     seed=seed,
                     )

    return agent
