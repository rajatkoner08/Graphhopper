import torch
import torch.nn as nn

from codes.model.encoder import EmptyLayer
from codes.model import GCN, GAT, TDecoder


class BaseFusion(nn.Module):
    def __init__(self, types, config):
        super(BaseFusion, self).__init__()
        self.types = types
        if 'transformer' in types:
            self.decoder = TDecoder(config.transformer_nlayers, config.n_head,
                                    config.transformer_d_k, config.transformer_d_k,
                                    config.embedding_size, config.embedding_size,
                                    config.dropout)

    def __call__(self, feature_x, feature_y, data, nums):
        out = []
        if 'mean1' in self.types:
            out.append(self.mean_type_1(feature_x, feature_y, data))
        if 'dot1' in self.types:
            out.append(self.dot_type_1(feature_x, feature_y, data))
        if 'mean2' in self.types:
            out.append(self.mean_type_2(feature_x, feature_y, data))
        if 'dot2' in self.types:
            out.append(self.dot_type_2(feature_x, feature_y, data))
        if 'transformer' in self.types:
            out.append(self.transformer_function(feature_x, feature_y, data))
        out = torch.cat(out, dim=-1)
        out = out.repeat_interleave(nums,0)
        return out

    def mean_type_1(self, feature_x, feature_y, data):
        raise NotImplementedError

    def dot_type_1(self, feature_x, feature_y, data):
        raise NotImplementedError

    def mean_type_2(self, feature_x, feature_y, data):
        """
        inputs: feature_y: [B,N,D]
        """
        feature_y *= data.question_seq.unsqueeze(-1)
        extra_question = torch.sum(feature_y, dim=1)
        extra_question = extra_question / data.question_rowlength.unsqueeze(-1)
        return extra_question

    def dot_type_2(self, feature_x, feature_y, data):
        """
        feature: nb, nc, nd
        data.row: nb
        data.seq: nb, nc
        """
        question_mean = feature_y * data.question_seq.unsqueeze(-1)
        question_mean = question_mean.sum(1)/data.question_rowlength.unsqueeze(-1) #[B,D]
        question_expanded = question_mean.unsqueeze(-1) #[B,D,1]
        dot = torch.bmm(feature_x, question_expanded) #[B,N,1]
        dot_squeeze = dot.squeeze(2)
        coefs = torch.nn.functional.softmax(dot_squeeze, dim=-1) #[B,1,N]

        scores = torch.bmm(coefs.unsqueeze(1), feature_x) #[B,1,D]
        scores = scores.squeeze(1)
        return scores

    def transformer_function(self, feature_x, feature_y, data):
        # inputs: dec_inputs, tgt_seq, src_seq, enc_outputs
        fusion = self.decoder(feature_y, data.question_seq, data.seq, feature_x)
        return fusion.mean(1)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

    def setup(self, encoder_type, decoder_type, fusion_type, config):
        self.encoder_type=encoder_type
        self.decoder_type=decoder_type
        self.fusion_type=fusion_type
        # graph encoder
        if encoder_type=='gcn':
            self.graph_enc = GCN(config.embedding_size, config.dropout)
        elif encoder_type=='gat':
            self.graph_enc = GAT(config.embedding_size, config.dropout)
        elif encoder_type=='transformer':
            raise NotImplementedError
        else:
            self.graph_enc = EmptyLayer()

        # question encoder
        self.question_dec = EmptyLayer()

        # fusion (Q2G, G2Q, ...)
        self.fusion = BaseFusion(fusion_type, config)

    def query(self, data, feature_x, feature_y, nums):

        # graph encoder
        feature_x = self.graph_enc(feature_x, data)
        b,n,d = feature_x.shape
        feature_x_flatten = feature_x.view(b*n, d)

        # question encoder
        feature_y = self.question_dec(feature_y, data.question_seq)

        # fusion
        fusion = self.fusion(feature_x, feature_y, data, nums)

        return fusion, feature_x_flatten.clone()
