import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    def __init__(self, num_features, dropout=0.):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.conv1 = GATConv(num_features, 8, heads=8, dropout=dropout)
        self.conv2 = GATConv(
            8 * 8, num_features, heads=1, concat=True, dropout=dropout)

    def forward(self, x, data):
        nb, nc, nd = x.shape
        x = x.view(nb*nc, nd)
        pad_seq = data.seq.view(nb*nc,1)
        x *= pad_seq

        edge_index = data.edges
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x.view(nb, nc, nd)
