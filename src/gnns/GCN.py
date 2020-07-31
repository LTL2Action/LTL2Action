import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch.conv import GraphConv

class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims = [32]):
        super(GCN, self).__init__()
        self.num_layers = len(hidden_dims)
        self.convs = nn.ModuleList([GraphConv(in_dim, out_dim) for (in_dim, out_dim)
                      in zip([input_dim] + hidden_dims[:-1], hidden_dims)])

        self.g_embed = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, g):
        g = np.array(g).reshape((1, -1)).tolist()[0]
        g = dgl.batch(g)
        h = g.ndata["feat"].float()
        for i in range(self.num_layers):
            h = F.relu(self.convs[i](g, h))
        g.ndata['h'] = h

        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')
        return self.g_embed(hg).squeeze(1)
