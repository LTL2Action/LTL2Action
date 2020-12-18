"""
This is the description of the deep NN currently being used.
It is a small CNN for the features with an GRU encoding of the LTL task.
The features and LTL are preprocessed by utils.format.get_obss_preprocessor(...) function:
    - In that function, I transformed the LTL tuple representation into a text representation:
    - Input:  ('until',('not','a'),('and', 'b', ('until',('not','c'),'d')))
    - output: ['until', 'not', 'a', 'and', 'b', 'until', 'not', 'c', 'd']
Each of those tokens get a one-hot embedding representation by the utils.format.Vocabulary class.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
import torch_ac

from gym.spaces import Box, Discrete

from gnns.graphs.GCN import *
from gnns.graphs.GNN import GNNMaker

from env_model import getEnvModel
from policy_network import PolicyNetwork

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class ACModel(nn.Module, torch_ac.ACModel):
    def __init__(self, env, obs_space, action_space, ignoreLTL, gnn_type, dumb_ac, freeze_ltl):
        super().__init__()

        # Decide which components are enabled
        self.use_progression_info = "progress_info" in obs_space
        self.use_text = not ignoreLTL and (gnn_type == "GRU" or gnn_type == "LSTM") and "text" in obs_space
        self.use_ast = not ignoreLTL and ("GCN" in gnn_type) and "text" in obs_space
        self.gnn_type = gnn_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_space = action_space
        self.dumb_ac = dumb_ac

        self.freeze_pretrained_params = freeze_ltl
        if self.freeze_pretrained_params:
            print("Freezing the LTL module.")

        self.env_model = getEnvModel(env, obs_space)

        # Define text embedding
        if self.use_progression_info:
            self.text_embedding_size = 32
            self.simple_encoder = nn.Sequential(
                nn.Linear(obs_space["progress_info"], 64),
                nn.Tanh(),
                nn.Linear(64, self.text_embedding_size),
                nn.Tanh()
            ).to(self.device)
            print("Linear encoder Number of parameters:", sum(p.numel() for p in self.simple_encoder.parameters() if p.requires_grad))

        elif self.use_text:
            self.word_embedding_size = 32
            self.text_embedding_size = 32
            if self.gnn_type == "GRU":
                self.text_rnn = GRUModel(obs_space["text"], self.word_embedding_size, 16, self.text_embedding_size).to(self.device)
            else:
                assert(self.gnn_type == "LSTM")
                self.text_rnn = LSTMModel(obs_space["text"], self.word_embedding_size, 16, self.text_embedding_size).to(self.device)
            print("RNN Number of parameters:", sum(p.numel() for p in self.text_rnn.parameters() if p.requires_grad))
        
        elif self.use_ast:
            hidden_dim = 32
            self.text_embedding_size = 32
            self.gnn = GNNMaker(self.gnn_type, obs_space["text"], self.text_embedding_size).to(self.device)
            print("GNN Number of parameters:", sum(p.numel() for p in self.gnn.parameters() if p.requires_grad))

       # Resize image embedding
        self.embedding_size = self.env_model.size()
        print("embedding size:", self.embedding_size)
        if self.use_text or self.use_ast or self.use_progression_info:
            self.embedding_size += self.text_embedding_size

        if self.dumb_ac:
            # Define actor's model
            self.actor = PolicyNetwork(self.embedding_size, self.action_space)

            # Define critic's model
            self.critic = nn.Sequential(
                nn.Linear(self.embedding_size, 1)
            )
        else:
            # Define actor's model
            self.actor = PolicyNetwork(self.embedding_size, self.action_space, hiddens=[64, 64, 64], activation=nn.ReLU())

            # Define critic's model
            self.critic = nn.Sequential(
                nn.Linear(self.embedding_size, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )

        # Initialize parameters correctly
        self.apply(init_params)

    def forward(self, obs):
        embedding = self.env_model(obs)

        if self.use_progression_info:
            embed_ltl = self.simple_encoder(obs.progress_info)
            embedding = torch.cat((embedding, embed_ltl), dim=1) if embedding is not None else embed_ltl

        # Adding Text
        elif self.use_text:
            embed_text = self.text_rnn(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1) if embedding is not None else embed_text

        # Adding GNN
        elif self.use_ast:
            embed_gnn = self.gnn(obs.text)
            embedding = torch.cat((embedding, embed_gnn), dim=1) if embedding is not None else embed_gnn

        # Actor
        dist = self.actor(embedding)

        # Critic
        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value

    def load_pretrained_gnn(self, model_state):
        # We delete all keys relating to the actor/critic.
        new_model_state = model_state.copy()

        for key in model_state.keys():
            if key.find("actor") != -1 or key.find("critic") != -1:
                del new_model_state[key]

        self.load_state_dict(new_model_state, strict=False)

        if self.freeze_pretrained_params:
            target = self.text_rnn if self.gnn_type == "GRU" or self.gnn_type == "LSTM" else self.gnn

            for param in target.parameters():
                param.requires_grad = False


class LSTMModel(nn.Module):
    def __init__(self, obs_size, word_embedding_size=32, hidden_dim=32, text_embedding_size=32):
        super().__init__()
        # For all our experiments we want the embedding to be a fixed size so we can "transfer". 
        self.word_embedding = nn.Embedding(obs_size, word_embedding_size)
        self.lstm = nn.LSTM(word_embedding_size, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.output_layer = nn.Linear(2*hidden_dim, text_embedding_size)

    def forward(self, text):
        hidden, _ = self.lstm(self.word_embedding(text))
        return self.output_layer(hidden[:, -1, :])


class GRUModel(nn.Module):
    def __init__(self, obs_size, word_embedding_size=32, hidden_dim=32, text_embedding_size=32):
        super().__init__()
        self.word_embedding = nn.Embedding(obs_size, word_embedding_size)
        self.gru = nn.GRU(word_embedding_size, hidden_dim, num_layers=2, batch_first=True, bidirectional=True)
        self.output_layer = nn.Linear(2*hidden_dim, text_embedding_size)

    def forward(self, text):
        hidden, _ = self.gru(self.word_embedding(text))
        return self.output_layer(hidden[:, -1, :])



