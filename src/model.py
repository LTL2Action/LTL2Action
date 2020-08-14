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
from torch.distributions.categorical import Categorical
import torch_ac

from gnns.graphs.GCN import *
from gnns.graphs.GNN import GNNMaker

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)


class ACModel(nn.Module, torch_ac.RecurrentACModel):
    def __init__(self, obs_space, action_space, ignoreLTL, use_memory, gnn_type, append_h0):
        super().__init__()

        # Decide which components are enabled
        self.use_text = not ignoreLTL and not gnn_type
        self.use_memory = False
        self.gnn_type = gnn_type
        self.append_h0 = append_h0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define image embedding
        n = obs_space["image"][0]
        m = obs_space["image"][1]
        k = obs_space["image"][2]
        self.image_conv = nn.Sequential(
            nn.Conv2d(k, 16, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(16, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU()
        )
        #self.image_embedding_size = ((n-1)//2-2)*((m-1)//2-2)*64
        self.image_embedding_size = (n-3)*(m-3)*64

        # Define memory
        if self.use_memory:
            self.memory_rnn = nn.LSTMCell(self.image_embedding_size, self.semi_memory_size)

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 128
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        if self.gnn_type:
            hidden_dim = 32
            self.text_embedding_size = 128
            self.gnn = GNNMaker(self.gnn_type, obs_space["text"], self.text_embedding_size, self.append_h0).to(self.device)

        # Resize image embedding
        self.embedding_size = self.semi_memory_size
        if self.use_text or self.gnn_type:
            self.embedding_size += self.text_embedding_size

        # Define actor's model
        self.actor = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, action_space.n)
        )

        # Define critic's model
        self.critic = nn.Sequential(
            nn.Linear(self.embedding_size, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

        # Initialize parameters correctly
        self.apply(init_params)

    @property
    def memory_size(self):
        return 2*self.semi_memory_size

    @property
    def semi_memory_size(self):
        return self.image_embedding_size

    def forward(self, obs, memory):
        x = obs.image.transpose(1, 3).transpose(2, 3)
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)

        # Image
        if self.use_memory:
            hidden = (memory[:, :self.semi_memory_size], memory[:, self.semi_memory_size:])
            hidden = self.memory_rnn(x, hidden)
            embedding = hidden[0]
            memory = torch.cat(hidden, dim=1)
        else:
            embedding = x

        # Adding Text
        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1)

        # Adding GNN
        if self.gnn_type:
            embed_gnn = self.gnn(obs.text)
            embedding = torch.cat((embedding, embed_gnn), dim=1)

        # Actor
        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        # Critic
        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value, memory

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]
