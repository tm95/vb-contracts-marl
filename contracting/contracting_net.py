import torch
import torch.nn as nn
from collections import namedtuple
import torch.nn.functional as f
import random
import numpy as np


class ConvNet(nn.Module):

    def __init__(self, channels, height, nb_agents):
        super().__init__()

        self.cnv_net = nn.Sequential(
            nn.Linear(height*height*channels, 64),  # 64
            nn.ELU(),
            nn.Linear(64, 32),  # 64
            nn.ELU(),
            nn.Linear(32, nb_agents),
            nn.Softmax()
        )

        self.head = nn.Sequential(
            nn.Linear(864, 128),
            nn.Linear(128, 128),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.cnv_net(x)
        return x


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'contract'))


class ReplayMemory(object):
    def __init__(self, capacity: int, seed: int = 42) -> None:
        self.rng = random
        self.rng.seed(seed)
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args) -> None:
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size) -> []:
        return self.rng.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Contracting_Net(nn.Module):
    def __init__(self,
                 observation_shape: [int],
                 sinkhorn_iter: int = 50,
                 mini_batch_size: int = 32,
                 gamma: float = 0.99,
                 target_update_period: int = 1000,
                 seed: int = 42,
                 nb_agents: int = 0,
                 epsilon_decay: float = 0.001,
                 buffer_capacity: int = 50000,
                 warm_up_duration: int = 2000,
                 epsilon_min: float = 0.001
                 ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.device = torch.device("cpu")
        self.memory = ReplayMemory(buffer_capacity, seed)
        self.warm_up_duration = warm_up_duration
        self.channels = observation_shape[0]
        self.height = observation_shape[1]
        self.width = observation_shape[2]
        self.mini_batch_size = mini_batch_size
        self.sinkhorn_iter = sinkhorn_iter
        self.gamma = gamma
        self.nb_agents = nb_agents
        self.saved_probs = []
        self.pre_head_dim = self.nb_agents * self.height * self.width
        self.epsilon = 1
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.training_count = 0
        self.target_update_period = target_update_period
        self.conv_net = ConvNet(self.channels,
                                self.height,
                                self.nb_agents).to(self.device)
        self.optimizer = torch.optim.Adagrad(list(self.conv_net.parameters()), lr=0.0005)

    def save(self, state, action, next_state, reward, done, num_contracts):
        state = torch.tensor(state, device=self.device, dtype=torch.float32)
        action = torch.tensor(action, device=self.device, dtype=torch.long)
        if all(agent_done is True for agent_done in done):
            next_state = None
        else:
            next_state = torch.tensor(next_state, device=self.device, dtype=torch.float32)
        reward = torch.tensor(reward, device=self.device, dtype=torch.float32)
        contracts = torch.tensor(num_contracts, device=self.device, dtype=torch.float32)
        self.memory.push(state, action, next_state, reward, contracts)

    def train(self):
        if len(self.memory) < self.warm_up_duration:
            return
        transitions = self.memory.sample(self.mini_batch_size)
        batch = Transition(*zip(*transitions))

        if all(batch is None for batch in batch.next_state):
            next_state_batch = torch.tensor([])
        else:
            next_state_batch = torch.cat([s for s in batch.next_state if s is not None])

        rewards = []

        for i in range(len(batch.reward)):
            rewards.append(batch.reward[i]*(1-(int(self.nb_agents/2) - batch.contract[i])))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(rewards)

        non_final_mask = torch.ones(self.mini_batch_size * self.nb_agents, device=self.device, dtype=torch.bool)

        state_action_values = self.conv_net(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze()
        next_state_values = torch.zeros(self.mini_batch_size * self.nb_agents, device=self.device)
        next_state_values[non_final_mask] = self.conv_net(next_state_batch).max(1)[0].detach()
        expected_state_action_values = reward_batch + next_state_values * self.gamma

        self.optimizer.zero_grad()
        loss = f.smooth_l1_loss(state_action_values, expected_state_action_values)  # huber loss
        loss.backward()
        self.optimizer.step()
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)
        self.training_count += 1
        if self.training_count % self.target_update_period is 0:
            print("loss before target network update: {:.5f}".format(loss))
            self.conv_net.load_state_dict(self.conv_net.state_dict())
            self.conv_net.eval()

    def policy(self, observation):
        if torch.rand(1) < self.epsilon:
            return abs(np.argmax(torch.randn(self.nb_agents, self.nb_agents).numpy(), axis=1))
        else:
            return np.argmax(self.compute_partner(observation), axis=1)

    def compute_partner(self, observation):
        # Process input
        agents = []
        for i in range(self.nb_agents):
            x = self.conv_net(torch.tensor([observation[i]], device=self.device, dtype=torch.float32))
            agents.append(x)

        x = torch.cat(agents, dim=1)

        partner = x.view(self.nb_agents, self.nb_agents)

        return partner.cpu().detach().numpy()

    def save_weights(self, path):
        torch.save(self.conv_net.state_dict(), path)

    def load_weights(self, path):
        self.conv_net.load_state_dict(torch.load(path, map_location='cpu'))
        self.conv_net.eval()


def make_contracting_net(params, observation_shape):

    contracting_net = Contracting_Net(observation_shape=observation_shape,
                                      nb_agents=params.nb_agents,
                                      mini_batch_size=128,
                                      gamma=params.gamma,
                                      target_update_period=params.target_update_period,
                                      epsilon_decay=params.epsilon_decay_contract,
                                      warm_up_duration=params.warm_up_duration,
                                      buffer_capacity=params.buffer_capacity,
                                      epsilon_min=0.01)
    return contracting_net
