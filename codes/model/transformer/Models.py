''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
import codes.model.transformer.Constants as Constants
from codes.model.transformer.Layers import EncoderLayer, DecoderLayer, RelLayer

__author__ = "Yu-Hsiang Huang"

def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(Constants.PAD).type(torch.float).unsqueeze(-1)

def get_sinusoid_encoding_table(seq_len, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return position / np.power(seq_len, 2 * (hid_idx // 2) / d_hid)     #todo check 1000 will be better for here?

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(seq_len)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(Constants.PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()

        # self.src_word_emb = nn.Embedding(
        #     n_src_vocab, d_word_vec, padding_idx=Constants.PAD)

        # self.pos_index_enc = nn.Embedding.from_pretrained(
        #     get_sinusoid_encoding_table(max_seq, d_word_vec//2, padding_idx=0),
        #     freeze=True)  # for objects will repeat twice the enc
        # self.position_enc = nn.Embedding.from_pretrained(
        #     get_sinusoid_encoding_table(img_dim, d_word_vec, padding_idx=0),
        #     freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])



    def forward(self, img_feats, src_seq,  src_pos = None, return_attns=False):

        enc_slf_attn_list = []

        # -- Prepare masks
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        non_pad_mask = get_non_pad_mask(src_seq)

        # -- Forward
        if src_pos is None:
            enc_output = img_feats
        else:
            enc_output = img_feats + self.pos_index_enc(src_pos).repeat(1,1,2)  #todo to be tested, repeat data 2 times if 1024 dn after till 2048

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
            if return_attns:
                enc_slf_attn_list += [enc_slf_attn]

        if return_attns:
            return enc_output, enc_slf_attn_list
        return enc_output


class Rel(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(self, d_model, dropout=0.1):

        super().__init__()

        self.rel_layer = RelLayer(d_model,  d_model, d_model, n_head=1, dropout=dropout)


    def forward(self, enc_out, src_seq,  src_pos = None, return_attns=False):

        # -- Prepare masks
        self_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        rel_self_attn = self.rel_layer(enc_out, self_attn_mask)

        return rel_self_attn


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):

        super().__init__()

        # self.tgt_word_emb = nn.Embedding(
        #     n_tgt_vocab, d_word_vec, padding_idx=Constants.PAD)
        #
        self.pos_index_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(30, d_model, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, obj_comb, tgt_seq, src_seq, enc_output, pos_enc=True, return_attns=False, e2e=True):

        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq)

        #pad after valid seq
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)

        #todo use some masking technique that network can understand which are two combination and which are not
        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)

        # -- Forward
        if pos_enc:
            # generate sinusoid position encoding for question tokens(list[int] -> embedding)
            combined_indices = obj_comb.new_zeros(obj_comb.size())
            pos = torch.arange(0, obj_comb.size()[1])
            pos = pos.expand(obj_comb.shape[0], obj_comb.shape[1])
            pos = pos.to(obj_comb.device)
            combined_indices[:,:,:] = self.pos_index_enc(pos)
            dec_output = obj_comb + combined_indices
        else:
            dec_output = obj_comb

        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask_keypad,
                dec_enc_attn_mask=dec_enc_attn_mask, e2e_attn=e2e)

            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]

        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output

class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''

    def __init__(self, n_src_vocab, n_tgt_vocab, img_dim = 592, d_word_vec=512, d_model=512, d_inner=2048, n_layers=6,
                 n_head=8, d_k=64, d_v=64, dropout=0.1, tgt_emb_prj_weight_sharing=True,
                 emb_src_tgt_weight_sharing=True, pos_enc_type='no_enc'):

        super().__init__()

        img_dim +=1

        self.encoder = Encoder(
            n_src_vocab=n_src_vocab, img_dim=img_dim,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout, pos_enc_type=pos_enc_type)

        self.decoder = Decoder(
            n_tgt_vocab=n_tgt_vocab, max_seq=img_dim,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)

        assert d_model == d_word_vec, \
        'To facilitate the residual connections, \
         the dimensions of all module outputs shall be the same.'

        if tgt_emb_prj_weight_sharing:
            # Share the weight matrix between target word embedding & the final logit dense layer
            self.tgt_word_prj.weight = self.decoder.tgt_word_emb.weight
            self.x_logit_scale = (d_model ** -0.5)
        else:
            self.x_logit_scale = 1.

        if emb_src_tgt_weight_sharing:
            # Share the weight matrix between source & target word embeddings
            assert n_src_vocab == n_tgt_vocab, \
            "To share word embedding table, the vocabulary size of src/tgt shall be the same."
            self.encoder.src_word_emb.weight = self.decoder.tgt_word_emb.weight

    def forward(self, src_seq, src_pos, tgt_seq, tgt_pos):

        tgt_seq, tgt_pos = tgt_seq[:, :-1], tgt_pos[:, :-1]

        enc_output, *_ = self.encoder(src_seq, src_pos)
        dec_output, *_ = self.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
        seq_logit = self.tgt_word_prj(dec_output) * self.x_logit_scale

        return seq_logit.view(-1, seq_logit.size(2))
