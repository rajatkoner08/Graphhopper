import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

from codes.model.basemodel import Model as FusionModel


class PolicyMLP(nn.Module):
    def __init__(self, m, config):
        super(PolicyMLP, self).__init__()
        in_1 = m*config.embedding_size+ m*config.hidden_size \
               + (len(config.fusion_type)-1)*config.embedding_size
        h_1 = 4*config.hidden_size
        out_1 = m*config.embedding_size
        self.dense = nn.Linear(in_1, h_1)
        self.dense_1 = nn.Linear(h_1, out_1)
    def init_weights(self):
        torch.nn.init.xavier_normal_(self.dense.weight)
        torch.nn.init.xavier_normal_(self.dense_1.weight)
        torch.nn.init.zeros_(self.dense.bias)
        torch.nn.init.zeros_(self.dense_1.bias)
    def forward(self, x):
        x = self.dense(x)
        x = F.relu(x)
        x = self.dense_1(x)
        x = F.relu(x)
        return x


class RNN(nn.Module):
    def __init__(self, m, config):
        super(RNN, self).__init__()
        input_dim = config.embedding_size*m
        h_dim = config.hidden_size*m
        self.cell_0 = nn.LSTM(input_dim, h_dim, config.LSTM_layers)
    def init_weights(self):
        for name, param in self.cell_0.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias_ih' in name:
                nn.init.zeros_(param)
            elif 'bias_hh' in name:
                nn.init.zeros_(param)
    def forward(self, x, h):
        x.unsqueeze_(0)
        out, new_state = self.cell_0(x, h)
        return out.squeeze(0), new_state


class EmbeddingLayer(nn.Module):
    def __init__(self, config, action_vocab_size, entity_vocab_size):
        super().__init__()
        self.embedding_size = config.embedding_size
        self.global_relation_lookup = nn.Embedding(action_vocab_size, config.embedding_size)
        self.global_entity_lookup = nn.Embedding(entity_vocab_size, config.embedding_size)
        self.global_relation_lookup.weight.requires_grad_(config.train_relation_embeddings)
        self.global_entity_lookup.weight.requires_grad_(config.train_entity_embeddings)

    def init_weights(self, entity_embeddings, relation_embeddings):
        if entity_embeddings is not None:
            train_entity = self.global_relation_lookup.weight.requires_grad
            self.global_entity_lookup = self.global_entity_lookup.from_pretrained(
                torch.tensor(entity_embeddings), freeze=not train_entity)
        else:
            torch.nn.init.xavier_uniform_(self.global_entity_lookup.weight)
        if relation_embeddings is not None:
            train_relation = self.global_relation_lookup.weight.requires_grad
            self.global_relation_lookup = self.global_relation_lookup.from_pretrained(
                torch.tensor(relation_embeddings), freeze= not train_relation)
        else:
            torch.nn.init.xavier_uniform_(self.global_relation_lookup.weight)

    def reset(self, soft_labels, soft_weight, soft_r, soft_r_weight):
        # nodes
        self.soft_labels = soft_labels
        self.soft_weight = soft_weight
        self.soft_embedding = self.global_entity_lookup(soft_labels)
        self.init_avg_embedding = (self.soft_embedding * soft_weight.unsqueeze(-1)).mean(-2)

        # edges
        self.soft_r = soft_r
        self.soft_rweight = soft_r_weight
        self.soft_rembedding = self.global_relation_lookup(soft_r)
        self.init_avg_rembedding = (self.soft_rembedding * soft_r_weight.unsqueeze(-1)).mean(-2)

    def get_gnn_embedding(self, data):
        return self.init_avg_embedding, self.question_enc(data)

    def update(self, updated):
        self.update_avg_embedding = updated

        # reshape
        d_b, d_p, d_soft = self.soft_labels.shape
        self.soft_labels = self.soft_labels.reshape(-1, d_soft)
        self.soft_weight = self.soft_weight.reshape(-1, d_soft)
        self.init_avg_embedding = self.init_avg_embedding.view(-1, self.embedding_size)
        self.soft_embedding = self.soft_embedding.view(d_b* d_p, d_soft, self.embedding_size)

        # reshape
        d_b, d_p, d_soft = self.soft_r.shape
        self.soft_r = self.soft_r.reshape(-1, d_soft)
        self.soft_rweight = self.soft_rweight.reshape(-1, d_soft)
        self.init_avg_rembedding = self.init_avg_rembedding.view(-1, self.embedding_size)
        self.soft_rembedding = self.soft_rembedding.view(d_b* d_p, d_soft, self.embedding_size)

    def question_enc(self, data):
        feature_y = self.global_entity_lookup(data.question_tokens)
        return feature_y

    def get_global_entity_embedding(self, entities):
        return self.global_entity_lookup(entities)

    def action_encoder(self, next_relations, next_entites):
        relation_embedding = self._helper(self.init_avg_rembedding, next_relations)
        entity_embedding = self._helper(self.update_avg_embedding, next_entites)
        action_embedding = torch.cat([relation_embedding, entity_embedding], dim=-1)
        return action_embedding

    def entity_chooser_encoder(self, next_relations, current_entities, next_entities):
        relation_embedding = self._helper(self.init_avg_rembedding, next_relations)
        entity_embedding = self.global_entity_lookup(next_entities) # assume PAD has same embeddingID and arrayID
        action_embedding = torch.cat([relation_embedding, entity_embedding], dim=-1)

        next_ent_weights = self.soft_weight[current_entities]
        weights_canvas = torch.zeros_like(next_relations, dtype=torch.float32)
        weights_canvas[:, :3] = next_ent_weights
        return action_embedding, weights_canvas

    def entity_encoding_new(self, indices):
        return self._helper(self.update_avg_embedding, indices)

    def _helper(self, lookup, indices):
        origin_dim = indices.shape
        embeddings = lookup[indices.flatten()]
        return embeddings.view(*origin_dim, self.embedding_size)

    # convert idx back
    def decode_entity(self, current_entities, islast):
        return self.soft_labels[:,0][current_entities].cpu().data.numpy()

    def decode_relation(self, current_relations, islast):
        return self.soft_r[:,0][current_relations].cpu().data.numpy()


class Agent(nn.Module):
    def __init__(self, config):
        super(Agent, self).__init__()
        self.action_vocab_size = len(config.relation_vocab)
        self.entity_vocab_size = len(config.embedding_vocab)
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.rPAD = torch.tensor(config.rPAD, dtype=torch.int64, device=self.device)
        self.dummy_score = torch.tensor(-99999.0, dtype=torch.float32, device=self.device)
        self.num_rollouts = config.num_rollouts
        self.LSTM_Layers = config.LSTM_layers
        self.batch_size = config.batch_size
        self.hidden_size = config.hidden_size
        self.dummy_start = config.DUMMY_START_RELATION
        self.dummy_start_label = torch.tensor(
            np.ones(self.batch_size*self.num_rollouts, dtype='int64') * config.DUMMY_START_RELATION)
        self.use_entity_embeddings = config.use_entity_embeddings
        if self.use_entity_embeddings:
            self.m = 2
        else:
            self.m =1
        self.MLP_for_policy = PolicyMLP(self.m, config)
        self.multi_rnn_cell = RNN(self.m, config)
        self.fusion_model = FusionModel()
        self.fusion_model.setup(config.encoder_type,config.decoder_type,config.fusion_type,config)
        self.encoder_type = config.encoder_type
        self.embedding_layer = EmbeddingLayer(config, self.action_vocab_size, self.entity_vocab_size)

        if config.agg_mode == "add":
            self._agg = lambda x: x[0]+x[1]
        elif config.agg_mode == "mul":
            self._agg = lambda x: x[0] * x[1]
        else:
            self._agg = lambda x: x[0]

    def init_weights(self, entity_embeddings, relation_embeddings):
        self.MLP_for_policy.init_weights()
        self.multi_rnn_cell.init_weights()
        self.embedding_layer.init_weights(entity_embeddings, relation_embeddings)

    def prepare_classifier_input(self, next_labels):
        h = torch.cat([self.prev_state[0], self.prev_state[1]], dim=-1)
        h = h.squeeze(0).clone().detach() #only works for lstm layer=1
        Q = self.query_embedding.clone().detach()

        if next_labels.ndim==1:
            next_labels = next_labels.unsqueeze(0)
        next_labels= self.embedding_layer.get_global_entity_embedding(next_labels)
        return h, Q, next_labels.clone().detach()


    def init(self, data, num_paths, eval=False):
        self.range_arr = torch.arange(num_paths).type(torch.long)
        base = data.nodes.shape[0]
        nums = num_paths//base

        self.embedding_layer.reset(data.labels, data.weights, data.edge_labels, data.edge_weights)
        use_cuda = torch.cuda.is_available()

        feature_x, feature_y = self.embedding_layer.get_gnn_embedding(data)
        self.query_embedding, updatex = self.fusion_model.query(data, feature_x, feature_y, nums)
        self.embedding_layer.update(updatex)

        self.prev_state = (
            torch.zeros(self.LSTM_Layers, num_paths, self.m*self.hidden_size),
            torch.zeros(self.LSTM_Layers, num_paths, self.m * self.hidden_size)
        )
        if not eval:
            self.prev_relation = self.dummy_start_label
        else:
            self.prev_relation = torch.tensor(
                np.ones(num_paths, dtype='int64') * self.dummy_start)

        if use_cuda:
            self.prev_state = (self.prev_state[0].cuda(), self.prev_state[1].cuda())
            self.prev_relation = self.prev_relation.cuda()
            self.range_arr = self.range_arr.cuda()
            #self.prev_relation_reset = self.prev_relation_reset.cuda()
        self.all_probs = []
        self.all_logits = []
        self.all_path_log = []

    def forward(self, next_relation, next_entity, current_entity, debug=True, laststep=False):
        human_read_e, human_read_r = None, None
        loss, new_state, logits, idx, chosen_relation = self.step(
            next_relations=next_relation,
            next_entities=next_entity,
            current_entities=current_entity,
            prev_state=self.prev_state,
            prev_relation=self.prev_relation,
            query_embedding=self.query_embedding,
            range_arr=self.range_arr,
            laststep = laststep
        )
        self.prev_state = new_state
        self.prev_relation = chosen_relation
        self.all_probs.append(loss)
        self.all_logits.append(logits)

        return self.all_probs, self.all_logits, chosen_relation, idx, (human_read_e, human_read_r)

    def step(self, next_relations, next_entities, prev_state, prev_relation,
                query_embedding, current_entities, range_arr, laststep=False):
        prev_action_embedding = self.embedding_layer.action_encoder(prev_relation, current_entities)

        output, new_state = self.multi_rnn_cell(prev_action_embedding, prev_state)

        prev_entity = self.embedding_layer.entity_encoding_new(current_entities)
        if self.use_entity_embeddings:
            state = torch.cat([output, prev_entity], dim=-1)
        else:
            state = output
        if not laststep:
            candidate_action_embeddings = self.embedding_layer.action_encoder(next_relations, next_entities)
        else:
            candidate_action_embeddings, next_ent_weights = self.embedding_layer.entity_chooser_encoder(next_relations, current_entities, next_entities)

        state_query_concat = torch.cat([state, query_embedding], dim=-1)
        output = self.MLP_for_policy(state_query_concat)
        output.unsqueeze_(dim=1)
        prelim_scores = torch.sum(candidate_action_embeddings*output, dim=2)

        scores = torch.where(next_relations==self.rPAD, self.dummy_score, prelim_scores)

        if not laststep:
            multinomial = Categorical(logits=scores)
        else:
            prior = next_ent_weights
            likelihood = F.softmax(scores, dim=1)
            post = self._agg((likelihood, prior))
            post = post / (post.sum(1, keepdim=True) + 1e-6)
            assert torch.isfinite(post).all().item()
            multinomial = Categorical(probs=post)
        action_idx = multinomial.sample()  # [B]

        loss = - multinomial.log_prob(action_idx) # [B]
        chosen_relation = next_relations[range_arr, action_idx]

        return loss, new_state, torch.log_softmax(scores, dim=-1), action_idx, chosen_relation
