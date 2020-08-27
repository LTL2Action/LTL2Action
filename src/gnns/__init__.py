from gnns.graph_registry import *
from gnns.graphs.GNN import *

register(id="GCN_2x32_MEAN", entry_point="gnns.graphs.GCN.GCN", hidden_dims=[32, 32])

register(id="GCN_4x32_MEAN", entry_point="gnns.graphs.GCN.GCN", hidden_dims=[32, 32, 32, 32])

register(id="GCN_32_MEAN", entry_point="gnns.graphs.GCN.GCN", hidden_dims=[32])

register(id="GCN_32_ROOT", entry_point="gnns.graphs.GCN.GCNRoot", hidden_dims=[32])

register(id="GCN_2x32_ROOT", entry_point="gnns.graphs.GCN.GCNRoot", hidden_dims=[32, 32])

register(id="GCN_4x32_ROOT", entry_point="gnns.graphs.GCN.GCNRoot", hidden_dims=[32, 32, 32, 32])

register(id="GCN_2x32_ROOT_SHARED", entry_point="gnns.graphs.GCN.GCNRootShared", hidden_dim=32, num_layers=2)

register(id="GCN_4x32_ROOT_SHARED", entry_point="gnns.graphs.GCN.GCNRootShared", hidden_dim=32, num_layers=4)

register(id="RGCN_2x32_ROOT", entry_point="gnns.graphs.RGCN.RGCNRoot", hidden_dims=[32, 32])

register(id="RGCN_4x32_ROOT", entry_point="gnns.graphs.RGCN.RGCNRoot", hidden_dims=[32] * 4)

register(id="RGCN_8x32_ROOT", entry_point="gnns.graphs.RGCN.RGCNRoot", hidden_dims=[32] * 8)
