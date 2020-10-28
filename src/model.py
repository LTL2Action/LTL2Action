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


class ACModel(nn.Module, torch_ac.ACModel):
    def __init__(self, obs_space, action_space, ignoreLTL, gnn_type, append_h0, dumb_ac, unfreeze_ltl):
        super().__init__()

        # Decide which components are enabled
        self.use_text = not ignoreLTL and not gnn_type
        self.gnn_type = gnn_type
        self.append_h0 = append_h0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_space = action_space
        self.dumb_ac = dumb_ac

        self.freeze_pretrained_params = not unfreeze_ltl
        print("Frozen LTL module" if self.freeze_pretrained_params else "Unfrozen LTL module")

        # Define image embedding
        if "image" in obs_space.keys():
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
        else:
            self.image_embedding_size = 0

        # Define text embedding
        if self.use_text:
            self.word_embedding_size = 32
            self.word_embedding = nn.Embedding(obs_space["text"], self.word_embedding_size)
            self.text_embedding_size = 32
            self.text_rnn = nn.GRU(self.word_embedding_size, self.text_embedding_size, batch_first=True)

        if self.gnn_type:
            hidden_dim = 32
            self.text_embedding_size = 128
            self.gnn = GNNMaker(self.gnn_type, obs_space["text"], self.text_embedding_size, self.append_h0).to(self.device)

        # Resize image embedding
        self.embedding_size = self.image_embedding_size
        if self.use_text or self.gnn_type:
            self.embedding_size += self.text_embedding_size

        if self.dumb_ac:
            # Define actor's model
            self.actor = nn.Sequential(
                nn.Linear(self.embedding_size, self.action_space.n)
            )

            # Define critic's model
            self.critic = nn.Sequential(
                nn.Linear(self.embedding_size, 1)
            )
        else:
            # Define actor's model
            self.actor = nn.Sequential(
                nn.Linear(self.embedding_size, 64),
                nn.Tanh(),
                nn.Linear(64, self.action_space.n)
            )

            # Define critic's model
            self.critic = nn.Sequential(
                nn.Linear(self.embedding_size, 64),
                nn.Tanh(),
                nn.Linear(64, 1)
            )

        # Initialize parameters correctly
        self.apply(init_params)

    def forward(self, obs):
        embedding = None 

        if "image" in obs.keys():
            x = obs.image.transpose(1, 3).transpose(2, 3)
            x = self.image_conv(x)
            x = x.reshape(x.shape[0], -1)
            embedding = x

        # Adding Text
        if self.use_text:
            embed_text = self._get_embed_text(obs.text)
            embedding = torch.cat((embedding, embed_text), dim=1) if embedding is not None else embed_text

        # Adding GNN
        if self.gnn_type:
            embed_gnn = self.gnn(obs.text)
            embedding = torch.cat((embedding, embed_gnn), dim=1) if embedding is not None else embed_gnn

        # Actor
        x = self.actor(embedding)
        dist = Categorical(logits=F.log_softmax(x, dim=1))

        # Critic
        x = self.critic(embedding)
        value = x.squeeze(1)

        return dist, value

    def _get_embed_text(self, text):
        _, hidden = self.text_rnn(self.word_embedding(text))
        return hidden[-1]

    def load_pretrained_rnn(self, model_state):

        # We delete all keys relating to the actor/critic.
        # We only wish to load the `word_embedding` and `text_rnn` parameters in new_model_state.
        new_model_state = model_state.copy() 

        for key in model_state.keys():
            if key.find("actor") != -1 or key.find("critic") != -1:
                del new_model_state[key]

        self.load_state_dict(new_model_state, strict=False)

        if self.freeze_pretrained_params:
            for param in self.text_rnn.parameters():
                param.requires_grad = False

