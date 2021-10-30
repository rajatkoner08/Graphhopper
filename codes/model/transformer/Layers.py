''' Define the Layers '''
import torch.nn as nn
from codes.model.transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward

__author__ = "Yu-Hsiang Huang"


class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)

        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, enc_slf_attn

class RelLayer(nn.Module):
    '''Apply attention on Node context to filter relation'''
    def __init__(self, d_model, d_k, d_v=None, n_head=1, dropout=0.1):
        super(RelLayer, self).__init__()
        self.slf_node_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, attn4rel=True, dropout=dropout)

    def forward(self, enc_input, slf_attn_mask=None):

        rel_slf_attn = self.slf_node_attn(enc_input, enc_input, v=None, mask=slf_attn_mask)

        return rel_slf_attn


class NewDecoder(nn.Module):
    '''Compose with three layers'''
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, comb_type=None):
        super(DecoderLayer, self).__init__()
        if comb_type == 'concat':
            self.slf_attn = MultiHeadAttention(2*n_head, 2*d_model, d_k, d_v, dropout=dropout)
        else:
            self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)

class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, comb_type=None):
        super(DecoderLayer, self).__init__()
        if comb_type == 'concat':      #todo is this relevent now?as multiply is done already
            self.slf_attn = MultiHeadAttention(2*n_head, 2*d_model, d_k, d_v, dropout=dropout)
        else:
            self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, dec_input, enc_output, non_pad_mask=None, slf_attn_mask=None, dec_enc_attn_mask=None, e2e_attn=True):
        dec_output = dec_input   #todo temp exp for n2e dn e2e, remove later

        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output *= non_pad_mask

        if e2e_attn:
            dec_output, dec_slf_attn = self.slf_attn(
                dec_input, dec_input, dec_input, mask=slf_attn_mask)
        else:
            dec_slf_attn = None

        dec_output *= non_pad_mask


        dec_output = self.pos_ffn(dec_output)
        dec_output *= non_pad_mask

        return dec_output, dec_slf_attn, dec_enc_attn
