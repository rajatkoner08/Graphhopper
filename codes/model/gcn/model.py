import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class GCN(torch.nn.Module):
    def __init__(self, num_features, dropout=0.):
        super(GCN, self).__init__()
        self.dropout = dropout
        self.conv1 = GCNConv(num_features, 16,
                             normalize=True)
        self.conv2 = GCNConv(16, num_features,
                             normalize=True)

    def forward(self, x, data):
        nb, nc, nd = x.shape
        x = x.view(nb*nc, nd)
        pad_seq = data.graph_seq.view(nb*nc,1)
        x *= pad_seq

        edge_index = data.graph_adj
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training, p=self.dropout)
        x = self.conv2(x, edge_index)
        return x.view(nb, nc, nd)