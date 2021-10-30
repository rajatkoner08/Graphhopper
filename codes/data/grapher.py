from collections import defaultdict
import logging
import numpy as np
import json
import os
logger = logging.getLogger(__name__)
import torch
import networkx as nx
from networkx.readwrite import json_graph
from tqdm import tqdm


class Symbols(object):
    def __init__(self, emb_vocab, rel_vocab):
        self.__dict__ = {
            'ePAD':0, 'eHUB':1, 'EOS':2, 'TRUE': 3, 'FALSE': 4,
            'rPAD':0, 'DUMMY_START_RELATION':1, 'NO_OP':2, 'rHUB':3, 'JUDGE':4,
        }
        self.edge_index = {
            'PAD':0,
            'DUMMY_START_RELATION':1,
            'NO_OP':2,
            'HUB':3,
            'JUDGE':4,
        }
        self.node_index = {'ePAD':0, 'eHUB':1, 'EOS':2, 'TRUE': 3, 'FALSE': 4,}
        self.__validate__(emb_vocab, rel_vocab)
        self.num_e_symbols = 5
        self.num_r_symbols = 5

    def __validate__(self, emb_vocab, rel_vocab):
        assert self.rPAD == rel_vocab['PAD']
        assert self.ePAD == emb_vocab['PAD']
        assert self.rHUB == rel_vocab['UNK']


class Batch(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def to_tensor(self):
        for k,v in self.__dict__.items():
            self.__dict__[k] = torch.from_numpy(v)

    def to(self, device):
        for k,v in self.__dict__.items():
            self.__dict__[k] = v.to(device)

    def add(self, **kwargs):
        self.__dict__.update(kwargs)


class GrapherBatcher:
    def __init__(self, input_dir, mode, embedding_vocab, relation_vocab, max_num_actions, hub_node, streaming=False):
        self.graph_encoder = embedding_vocab
        self.rev_embedding_vocab = dict([(v, k) for k, v in embedding_vocab.items()])
        self.rev_relation_vocab = dict([(v, k) for k, v in relation_vocab.items()])
        self.symbols = Symbols(embedding_vocab, relation_vocab)

        graph_list = json.load(open(input_dir + '/split_index.json'))
        graph_list = [k for k,v in graph_list.items() if v==mode]
        assert len(graph_list) > 0, "Empty Graph Folder"
        logger.info(f"Total number of graphs in index {len(graph_list)}")
        graph_files = [os.path.join(input_dir, 'graphs', f'{x}.json') for x in graph_list if os.path.isfile(os.path.join(input_dir, 'graphs', f'{x}.json'))]
        logger.info(f"Total number of graphs found in folder {len(graph_files)}")

        self._stream_loading = streaming and mode!='train' # option for stream for validation
        if self._stream_loading:
            logger.info('load data in stream')
            self._input_dir = input_dir + '/graphs/'
            self._embedding_vocab = embedding_vocab
            self._relation_vocab = relation_vocab
            self._max_num_actions = max_num_actions
        else:
            self.graphs = dict()
            self.graphs_len = dict()
            for graph_file in tqdm(graph_files):
                idx = os.path.basename(graph_file).split('.')[0]
                g = RelationEntityGraph(graph_file, embedding_vocab, relation_vocab, max_num_actions, self.symbols)
                self.graphs[idx] = g
                self.graphs_len[idx] = len(g)

            logger.info(f'total graphs {len(self.graphs)}')

    def load_graph(self, idx):
        graph_file = self._input_dir + f'{idx}.json'
        g = RelationEntityGraph(graph_file, self._embedding_vocab, self._relation_vocab, self._max_num_actions, self.symbols)
        ret = [
            g.graph_array.copy(),
            g.nodes.copy(),
            g.edge_lists.copy(),
            g.node_labels.copy(),
            g.node_weights.copy(),
            g.edge_labels.copy(),
            g.edge_weights.copy(),
        ]
        return ret

    def get_graph_data(self, idx, num_rollouts, hub_node, mode):
        """minimal operation for data loading and processing"""
        # 1. collect graphs
        batch = defaultdict(list)
        graph_arrays = []
        if self._stream_loading:
            ret = [self.load_graph(x) for x in idx]

            for item in ret:
                graph_arrays.append(item[0])
                batch["nodes"].append(item[1])
                batch["edge_list"].append(item[2])

                batch["node_labels"].append(item[3])
                batch["node_weights"].append(item[4])
                batch["edge_labels"].append(item[5])
                batch["edge_weights"].append(item[6])
        else:
            for x in idx:
                g = self.graphs[x]
                graph_arrays.append(g.graph_array.copy())
                batch["nodes"].append(g.nodes.copy())
                batch["edge_list"].append(g.edge_lists.copy())

                batch["node_labels"].append(g.node_labels.copy())
                batch["node_weights"].append(g.node_weights.copy())
                batch["edge_labels"].append(g.edge_labels.copy())
                batch["edge_weights"].append(g.edge_weights.copy())

        # 2. batch
        pad_length = max([len(x) for x in batch["node_labels"]])
        pad_length_e = max([len(x) for x in batch["edge_labels"]])
        batch_size = len(graph_arrays)
        max_num_action = graph_arrays[0].shape[1]
        self._max_num_action = max_num_action

        nodes_padded = np.zeros((batch_size, pad_length), dtype=np.int64)
        seq_padded = np.zeros((batch_size, pad_length), dtype=np.int64)
        edges_padded = []

        graph_array_padded = np.zeros((batch_size*pad_length, max_num_action, 2), dtype=np.int64)

        n_padded = np.zeros((2, batch_size, pad_length, 3), dtype=np.float32)
        e_padded = np.zeros((2, batch_size, pad_length_e, 3), dtype=np.float32)

        offsets = np.array([i*pad_length for i in range(batch_size)], dtype=np.int64)
        offsets_e = np.array([i*pad_length_e for i in range(batch_size)], dtype=np.int64)

        for i in range(batch_size):
            valid = len(graph_arrays[i])
            valid_e = len(batch["edge_labels"][i])

            nodes_padded[i, :valid] = batch["nodes"][i]
            seq_padded[i, :valid] = 1
            graph_arrays[i][:,:,0] += i*pad_length
            r_mask = graph_arrays[i][:,:,1]==self.symbols.rPAD
            graph_arrays[i][:,:,1][~r_mask] += i*pad_length_e
            graph_array_padded[i*pad_length:(i*pad_length+valid), :, :]= graph_arrays[i]

            edges_padded.append(batch["edge_list"][i] + offsets[i])
            n3 = batch["node_labels"][i].shape[1]
            n_padded[0, i, :valid, :n3] = batch["node_labels"][i]
            n_padded[1, i, :valid, :n3] = batch["node_weights"][i]
            e_padded[0, i, :valid_e, :n3] = batch["edge_labels"][i]
            e_padded[1, i, :valid_e, :n3] = batch["edge_weights"][i]

        edges_padded = np.concatenate(edges_padded, axis=0)
        start_padded = np.ones((batch_size*num_rollouts), dtype=np.int64) * self.symbols.eHUB + offsets.repeat(num_rollouts)
        dummy_start_rel = np.ones((batch_size*num_rollouts), dtype=np.int64) * self.symbols.DUMMY_START_RELATION + offsets_e.repeat(num_rollouts)

        final_padded = np.ones((batch_size * num_rollouts), dtype=np.int64) * self.symbols.EOS + offsets.repeat(num_rollouts)

        # 3. tensor
        self.graph_array = torch.from_numpy(graph_array_padded)

        # optimize
        b = Batch(nodes = nodes_padded, edges = edges_padded.T, seq = seq_padded,
                  labels = n_padded[0,...].astype(np.int64), weights = n_padded[1,...],
                  edge_labels = e_padded[0,...].astype(np.int64), edge_weights = e_padded[1,...],
                  start_entities = start_padded, start_relations=dummy_start_rel, final_padded=final_padded)
        b.to_tensor()

        self._labels = b.labels.view(-1, 3)
        self._canvas_next = torch.zeros(start_padded.shape[0], self._max_num_action, 2, dtype=torch.int64)
        self._canvas_next[:, :3, 1] = b.start_relations.unsqueeze(1)

        assert b.labels.shape==(batch_size, pad_length,3)
        assert b.edge_labels.shape==(batch_size, pad_length_e,3)
        assert b.weights.shape==(batch_size, pad_length,3)
        assert b.edge_weights.shape==(batch_size, pad_length_e,3)
        assert b.seq.shape==(batch_size,pad_length)
        assert b.nodes.shape==(batch_size,pad_length)

        # 3. return
        if torch.cuda.is_available():
            self.graph_array = self.graph_array.cuda()
            self._canvas_next = self._canvas_next.cuda()

        return b
        # graph_array, node_list_labels, node_list_weight, edge_list
        # start_node
        # return

    def return_next_actions(self, current_entities):
        return self.graph_array[current_entities, :, :].clone()

    def return_entity_chooser_actions(self, current_entities):
        # current_entities #[B, id]
        # b.labels: [100, (w0, w1, w2)]
        self._canvas_next[:, :, 0] *= 0
        self._canvas_next[:, :3, 0] = self._labels[current_entities]
        return self._canvas_next.clone()

    def get_statistics(self):
        numbers = [x.get_statistics() for x in self.graphs.values()]
        entities = [x[0] for x in numbers]
        relations = [x[1] for x in numbers]
        logger.info("total number of graphs {} for {}".format(len(entities), self.mode))
        # logger.info("total number of entities {} relations {}".format(sum(entities),len(self.rev_relation_vocab)))
        logger.info("number of nodes in one graph max {} min {} mean {:.1f} median {:.1f}"
                    .format(max(entities), min(entities), np.mean(entities), np.median(entities)))

class RelationEntityGraph:

    def __init__(self, graph_path, embedding_vocab, relation_vocab, max_num_actions, symbols):
        self._emb_vocab = embedding_vocab
        self._rel_vocab = relation_vocab
        self._symbols = symbols
        self._pad_soft_k = -1
        g = self.load_json(graph_path)
        self.data = g
        self._number_of_edges = self.data.number_of_edges()
        self._number_of_nodes = self.data.number_of_nodes()
        assert self._number_of_edges > 0
        assert self._number_of_nodes > 0

        # properties/embeddings/dicts
        self.get_node_label_weight(g)
        self.get_edge_label_weight(g)
        self.nodes = np.arange(self.n_total_nodes)
        self.edge_lists = np.array(list(g.edges())) + self._symbols.num_r_symbols

        # graph array
        self.graph_array = self.create_graph_array(g, max_num_actions, symbols)

        assert self.node_labels.shape[1]==self.edge_labels.shape[1]
        assert self.node_labels.shape==self.node_weights.shape

        del self.data


    def create_graph_array(self, g, max_num_actions, symbols):
        array_store = np.ones((self.n_total_nodes, max_num_actions, 2), dtype=np.dtype('int64'))

        array_store[:, :, 0] *= self._symbols.ePAD
        array_store[:, :, 1] *= self._symbols.rPAD

        for e1, neighs in g.adjacency():
            e1 += self._symbols.num_e_symbols
            num_actions = 1
            array_store[e1, 0, 1] = self._symbols.NO_OP
            array_store[e1, 0, 0] = e1
            for e2 in neighs.keys():
                e2 += self._symbols.num_e_symbols
                if num_actions == array_store.shape[1]:
                    break
                array_store[e1,num_actions,0] = e2
                array_store[e1,num_actions,1] = self._edge_index[(e1,e2)]
                num_actions += 1

        array_store[self._symbols.eHUB, :self.n_nodes, 0] = self.nodes[self._symbols.num_e_symbols:][:max_num_actions]
        array_store[self._symbols.eHUB, :self.n_nodes, 1] = self._symbols.rHUB

        eos = self._symbols.EOS
        eT = self._symbols.TRUE
        eF = self._symbols.FALSE

        array_store[eos,:2,0] = np.array([eT, eF])
        array_store[eos,:2,1] = np.array([self._symbols.JUDGE])

        self.node_labels[self._symbols.eHUB, 0] = self._symbols.eHUB
        self.node_labels[eos, 0] = self._symbols.EOS
        self.node_labels[eT, 0] = self._symbols.TRUE
        self.node_labels[eF, 0] = self._symbols.FALSE
        self.node_weights[self._symbols.eHUB, 0] = 1
        self.node_weights[eos, 0] = 1
        self.node_weights[eT, 0] = 1
        self.node_weights[eF, 0] = 1

        # edge
        for k,v in self._symbols.edge_index.items():
            self.edge_labels[v, 0] = v
            self.edge_weights[v, 0] = 1

        return array_store

    def get_node_label_weight(self, g):
        n_nodes = g.number_of_nodes()
        assert n_nodes>1
        node_names_ = nx.get_node_attributes(g, 'name')

        node_weights_ = nx.get_node_attributes(g, 'weight')

        node_names, node_weights = [], []
        self._pad_soft_k = 3
        for i in range(n_nodes):
            name = node_names_[i]
            weight = node_weights_[i]
            n, w = self._pad(name,weight)
            node_names.append(n)
            node_weights.append(w)
        node_names = [self._emb_vocab.get(i, 0) for row in node_names for i in row]
        node_labels = np.array(node_names).reshape(-1, self._pad_soft_k)
        node_weights = np.array(node_weights)

        self.node_labels = self._shift_array(node_labels, self._symbols.ePAD, self._symbols.num_e_symbols)
        self.node_weights = self._shift_array(node_weights, 0, self._symbols.num_e_symbols)

    def get_edge_label_weight(self, g):
        edge_names = []
        edge_weights = []
        edge_index = self._symbols.edge_index.copy()
        for e in g.edges(data=True):
            edge_index[(e[0]+self._symbols.num_e_symbols, e[1]+self._symbols.num_e_symbols)] = len(edge_index)
            n, w = self._pad(e[2]['name'], e[2]['weight'])
            assert len(n)==len(w)==self._pad_soft_k
            edge_names.append(n)
            edge_weights.append(w)

        edge_names = [self._rel_vocab.get(i,0) for row in edge_names for i in row]

        edge_labels = np.array(edge_names).reshape(-1, self._pad_soft_k)

        edge_weights = np.array(edge_weights)
        edge_weights = edge_weights / np.sum(edge_weights, axis=1, keepdims=True)

        self.edge_labels = self._shift_array(edge_labels, self._symbols.rPAD, self._symbols.num_r_symbols)
        self.edge_weights = self._shift_array(edge_weights, 0, self._symbols.num_r_symbols)

        self._edge_index = edge_index

    def _shift_array(self, arr, value, d):
        new = np.ndarray((arr.shape[0]+d,arr.shape[1]), dtype=arr.dtype)
        new.fill(value)
        new[d:, :] = arr
        return new

    def _pad(self, name, weight):
        k = self._pad_soft_k
        if not name:
            return name, weight
        elif len(name) == len(weight) == k:
            return name, weight
        elif len(name) > k:
            raise
        name += ['PAD'] * (k - len(name))
        weight += [0] * (k - len(weight))
        return name, weight

    def __len__(self):
        return len(self.nodes)

    @property
    def n_edges(self):
        return self._number_of_edges

    @property
    def n_nodes(self):
        return self._number_of_nodes

    @property
    def n_total_nodes(self):
        return self.n_nodes + self._symbols.num_e_symbols

    @staticmethod
    def load_json(filename):
        if not os.path.isfile(filename):
            raise FileNotFoundError(filename)
        data = json.load(open(filename))
        return json_graph.node_link_graph(data)

