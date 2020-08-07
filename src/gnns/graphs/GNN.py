import torch
import torch.nn as nn

from gnns import *

class GNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()

    def forward(self, g):
        raise NotImplementedError

def GNNMaker(gnn_type, input_dim, output_dim):
    clazz = lookup(gnn_type)
    return clazz["class"](input_dim, output_dim, **clazz["kwargs"])
