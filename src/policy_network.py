import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal

from gym.spaces import Box, Discrete


class PolicyNetwork(nn.Module):
    def __init__(self, in_dim, action_space, hiddens=[], scales=None, activation=nn.Tanh()):
        super().__init__()

        layer_dims = [in_dim] + hiddens
        self.action_space = action_space
        self.num_layers = len(layer_dims)
        self.enc_ = nn.Sequential(*[fc(in_dim, out_dim, activation=activation)
            for (in_dim, out_dim) in zip(layer_dims, layer_dims[1:])])

        if (isinstance(self.action_space, Discrete)):
            action_dim = self.action_space.n
            self.discrete_ = nn.Sequential(
                nn.Linear(layer_dims[-1], action_dim)
            )
        elif (isinstance(self.action_space, Box)):
            action_dim = self.action_space.shape[0]

            self.mu_ = nn.Sequential(
                fc(layer_dims[-1], action_dim)
            )
            self.std_ = nn.Sequential(
                fc(layer_dims[-1], action_dim)
            )
            self.softplus = nn.Softplus()
            # self.scales = [1] * action_dim if scales==None else scales
        else:
            print("Unsupported action_space type: ", self.action_space)
            exit(1)

    def forward(self, obs):
        if (isinstance(self.action_space, Discrete)):
            x = self.enc_(obs)
            x = self.discrete_(x)
            return Categorical(logits=F.log_softmax(x, dim=1))
        elif (isinstance(self.action_space, Box)):
            x = self.enc_(obs)
            mu  = 2 * self.mu_(x)# * self.scales
            std = self.softplus(self.std_(x)) + 1e-3
            return Normal(mu, std)
        else:
            print("Unsupported action_space type: ", self.action_space)
            exit(1)


def fc(in_dim, out_dim, activation=nn.Tanh()):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        activation
    )
