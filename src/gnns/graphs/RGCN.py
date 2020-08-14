import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
from dgl.nn.pytorch.conv import RelGraphConv

from gnns.graphs.GNN import GNN

from utils.ast_builder import edge_types

class RGCN(GNN):
    def __init__(self, input_dim, output_dim, append_h0, **kwargs):
        super().__init__(input_dim, output_dim, append_h0)

        self.append_h0 = append_h0
        hidden_dims = kwargs.get('hidden_dims', [32])
        self.num_layers = len(hidden_dims)

        if self.append_h0:
            hidden_plus_input_dims = [hd + input_dim for hd in hidden_dims]
            self.convs = nn.ModuleList([RelGraphConv(in_dim, out_dim, len(edge_types), activation=F.relu)
                for (in_dim, out_dim) in zip([input_dim] + hidden_plus_input_dims[:-1], hidden_dims)])
        else:
            self.convs = nn.ModuleList([RelGraphConv(in_dim, out_dim, len(edge_types), activation=F.relu)
                for (in_dim, out_dim) in zip([input_dim] + hidden_dims[:-1], hidden_dims)])

        self.g_embed = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, g):
        g = np.array(g).reshape((1, -1)).tolist()[0]
        g = dgl.batch(g)
        h_0 = g.ndata["feat"].float()
        h = h_0
        etypes = g.edata["type"].float()
        for i in range(self.num_layers):
            if self.append_h0 and i != 0:
                h = self.convs[i](g, torch.cat([h, h_0], dim=1), etypes)
            else:
                h = self.convs[i](g, h, etypes)
        g.ndata['h'] = h

        # Calculate graph representation by averaging all the hidden node representations.
        hg = dgl.mean_nodes(g, 'h')
        return self.g_embed(hg).squeeze(1)


class RGCNRoot(RGCN):
    def __init__(self, input_dim, output_dim, append_h0, **kwargs):
        super().__init__(input_dim, output_dim, append_h0, **kwargs)

    def forward(self, g):
        g = np.array(g).reshape((1, -1)).tolist()[0]
        g = dgl.batch(g)

        h_0 = g.ndata["feat"].float().squeeze()
        h = h_0
        etypes = g.edata["type"]
        for i in range(self.num_layers):
            if self.append_h0 and i != 0:
                h = self.convs[i](g, torch.cat([h, h_0], dim=1), etypes)
            else:
                h = self.convs[i](g, h, etypes)
        g.ndata['h'] = h # TODO (Pashootan): Check if this is redundant
        hg = dgl.sum_nodes(g, 'h', weight='is_root')
        return self.g_embed(hg).squeeze(1)
