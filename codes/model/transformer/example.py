import torch
import torch.nn as nn

from codes.model.transformer.Models import Encoder, Decoder


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.encoder = Encoder(10, 100, 50, 1, 8, 50, 50, 50, 50)
        self.decoder = Decoder(10, 100, 50, 1, 8, 50, 50, 50, 50)

    def forward(self, x=None):
        img_feats= torch.randn(1,2,50)
        src_seq = torch.ones(1,2)
        obj_feats = torch.randn(1,3,50)
        tgt_seq = torch.ones(1,3)

        img_feats = self.encoder(img_feats, src_seq)
        x2 = self.decoder(obj_feats, tgt_seq, src_seq, img_feats)
        return x2

if __name__=='__main__':
    net = Net()
    out = net(None)