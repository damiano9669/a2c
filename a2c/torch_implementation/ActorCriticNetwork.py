import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ActorCriticNetwork(nn.Module):
    def __init__(self, input_size, number_actions):
        super(ActorCriticNetwork, self).__init__()

        hidden_size = 256

        self.critic_linear1 = nn.Linear(input_size, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, 1)

        self.actor_linear1 = nn.Linear(input_size, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, number_actions)

    def forward(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))

        value = F.relu(self.critic_linear1(state))
        value = self.critic_linear2(value)

        policy_dist = F.relu(self.actor_linear1(state))
        policy_dist = F.softmax(self.actor_linear2(policy_dist), dim=1)

        return value, policy_dist
