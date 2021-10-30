import torch
import torch.nn as nn
import numpy as np

__author__ = "Yu-Hsiang Huang"

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.softmax = nn.Softmax(dim=2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, q, k, v=None, attn4rel=False, mask=None):

        attn = torch.bmm(q, k.transpose(1, 2))  #todo proper mask matrix is need to give foucs on which combnination of object are,it wud be q*(mask*K)
        attn = attn / self.temperature

        if mask is not None:
            attn = attn.masked_fill(mask, -np.inf)

        if attn4rel:
            attn = self.sigmoid(attn)
            return None, attn
        else:
            attn = self.softmax(attn)
            attn = self.dropout(attn)
            output = torch.bmm(attn, v)

            return output, attn
