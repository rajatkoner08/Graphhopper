#!/usr/bin/env python
# coding: utf-8
import os
import glob
from collections import defaultdict
import json
import pickle
import h5py
import numpy as np
import networkx as nx
from networkx.readwrite import json_graph
from tqdm import tqdm


class DataIO():
    def __init__(self, data):
        self.data = data

    @staticmethod
    def is_fileexits(filename):
        if not os.path.isfile(filename):
            raise FileNotFoundError

    @classmethod
    def from_pickle(cls, filename):
        cls.is_fileexits(filename)
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        return data

    @classmethod
    def from_json(cls, filename):
        cls.is_fileexits(filename)
        data = json.load(open(filename, 'r'))
        return data

    @classmethod
    def from_hdf5(cls, filename, copy=False):
        cls.is_fileexits(filename)
        f = h5py.File(filename, 'r')
        if copy:
            data = {k:f[k][:] for k in f.keys()}
            f.close()
            f = data
        return f

    @classmethod
    def sample_dict(cls, data, nums=100):
        return {k:data[k] for k in list(data.keys())[:nums]}

    @classmethod
    def dump_hdf5(cls, data, filename):
        f = h5py.File(filename, 'w')
        for k,v in data.items():
            f.create_dataset(k, data=v)
        f.close()

    @classmethod
    def dump_json(cls, data, filename):
        json.dump(data, open(filename,'w'))

    @classmethod
    def dump_pickle(cls, data, filename):
        with open(filename, 'wb') as f:
            pickle.dump(data, f)


class ObjID():
    def add(self, key):
        """key -> continuous id"""
        if key not in self.__dict__:
            self.__dict__[key] = len(self.__dict__)
    def __getitem__(self, key):
        return self.__dict__[key]
    def __repr__(self):
        return str(self.__dict__)
    def __len__(self):
        return len(self.__dict__)


class SGConstructor:
    def __init__(self, sg_list):
        self.sg_list = sg_list

    @classmethod
    def from_dict(cls, graph_file):
        dict_dir = '../../data/dicts/'
        attribute_category = DataIO.from_json(os.path.join(dict_dir, 'attributeCategoryExtend.json'))
        ind_to_attr = DataIO.from_pickle(os.path.join(dict_dir, 'ind_to_attr.pkl'))
        ind_to_cls = DataIO.from_pickle(os.path.join(dict_dir, 'ind_to_class.pkl'))
        ind_to_rel = DataIO.from_pickle(os.path.join(dict_dir, 'ind_to_rels.pkl'))
        image_shapes = DataIO.from_json(os.path.join(dict_dir, 'image_shape.json'))
        graphs=DataIO.from_pickle(graph_file)

        sg_list = {}
        for gid, graph_dict in tqdm(graphs.items()):
            sg_list[gid] = cls.instance_from_dict(image_shapes[gid], graph_dict, attribute_category, 3, True, ind_to_cls, ind_to_rel, ind_to_attr)
        return cls(sg_list)

    @staticmethod
    def instance_from_dict(img_shape, graph_dict, attribute_category=None, topk=2, attr=True, ind_to_cls=None, ind_to_rel=None, ind_to_attr=None):
        if not attribute_category: attribute_category = dict()

        graph_dict.update(img_shape)

        assert len(graph_dict['obj_score'])>graph_dict['pred_rels'].max()
        assert len(graph_dict['pred_rels']) == len(graph_dict['pred_rels'])
        assert len(graph_dict['obj_score']) == len(graph_dict['gt_boxes'])

        obj_score = (-graph_dict['obj_score']).argsort(axis=1)[:,:topk]
        rel_score = (-graph_dict['rel_scores']).argsort(axis=1)[:,:topk]

        attr_score_pruned = graph_dict['pred_attr'][:,1:]
        attr_idx_pruned = graph_dict['pred_attr'][:,0].astype(np.int).tolist()
        attr_score_idn_pruned = ((-attr_score_pruned).argsort(axis=1)[:,:topk])
        attr_score_value_pruned = np.take_along_axis(attr_score_pruned, attr_score_idn_pruned, axis=1).tolist()
        assert len(attr_idx_pruned)==len(attr_score_idn_pruned)==len(attr_score_value_pruned)

        vocab = ObjID()
        for i in range(len(graph_dict['obj_score'])):
            vocab.add(str(f"x{i}"))
        num_objects = len(vocab)

        nodes = []
        adjacency = []
        for idx in range(len(obj_score)):
            k = f"x{idx}"
            nodes.append({'id': vocab[k],
                          'object': k,
                          'name': [ind_to_cls[x] for x in obj_score[idx]],
                          'weight': (graph_dict['obj_score'][idx][obj_score[idx]]).tolist(),
                         })
            adjacency.append([])
        # relation
        for idx, rel in enumerate(graph_dict['pred_rels']):
            src = f'x{rel[0]}'
            tgt = f'x{rel[1]}'

            link = {'id': vocab[tgt],
                   'name': [ind_to_rel[x] for x in rel_score[idx]],
                   'weight': (graph_dict['rel_scores'][idx][rel_score[idx]]).tolist(),
                   }
            adjacency[vocab[src]].append(link)

        # attribute
        if attr:
            for idx, attr_score_i_idx, weights in zip(attr_idx_pruned, attr_score_idn_pruned, attr_score_value_pruned):
                k = f"x{idx}"
                names = [ind_to_attr[x] for x in attr_score_i_idx]
                link_dict = defaultdict(list)
                for i, (n, w) in enumerate(zip(names, weights)):
                    hyper = 'has ' + attribute_category.get(n, 'state')
                    link_dict[hyper].append((n,w))

                # add new link(s)
                for i, (hyper, values) in enumerate(link_dict.items()):
                    vocab.add(k+'attr'+str(i))
                    nodes.append({'id':vocab[k + 'attr'+str(i)],
                                  'name': [value[0] for value in values],
                                  'weight': [value[1] for value in values],
                                 })
                    adjacency.append([])

                    link = {
                        'id': vocab[k + 'attr'+str(i)],
                        'name': [hyper],
                        'weight': [1.0],
                    }
                    adjacency[vocab[k]].append(link)

        # position
        for idx in range(len(graph_dict['gt_boxes'])):
            v = graph_dict['gt_boxes'][idx]

            hpos = 'right' if (v[0] + v[2]) > graph_dict['width'] else 'left'
            vpos = 'bottom' if (v[1] + v[3]) > graph_dict['height'] else 'top'

            k = str(f'x{idx}')
            vocab.add(k + 'vpos')
            nodes.append({'id': vocab[k + 'vpos'], 'name': [vpos], 'weight': [1.0]})
            link = {'id': vocab[k + 'vpos'],
                   'name': ['has vposition'],
                   'weight': [1.0]}
            adjacency[vocab[k]].append(link)
            adjacency.append([])

            vocab.add(k + 'hpos')
            nodes.append({'id': vocab[k + 'hpos'], 'name': [hpos], 'weight': [1.0]})
            link = {'id': vocab[k + 'hpos'],
                   'name': ['has hposition'],
                   'weight': [1.0]}
            adjacency[vocab[k]].append(link)
            adjacency.append([])

        data = {
            'nodes': nodes,
            'adjacency': adjacency,
        }
        g = json_graph.adjacency_graph(data, multigraph=False, directed=True)

        # node id is contiguous [0, len)
        assert all([x == i for i, (x, y) in enumerate(g.nodes(data=True))])
        g = SGConstructor.normalize_edge_weights(g)

        # add reverse links
        for r in g.edges():
            if r[::-1] not in g.edges():
                reverse_rel = ['reverse ' + x for x in g.edges[r]['name']]
                g.add_edge(r[1], r[0], name=reverse_rel, id=r[0], weight=g.edges[r]['weight'])

        return g

    @staticmethod
    def normalize_edge_weights(g):
        for e in g.edges():
            g.edges[e]['weight'] = SGConstructor.normalize_list(g.edges[e]['weight'])
        return g

    @staticmethod
    def normalize_list(x):
        y = np.array(x)
        return list(np.array(y)/np.sum(y))

    @classmethod
    def load_pickle(cls, root_dir):
        files = glob.glob(os.path.join(root_dir, '*.pkl'))
        sg_list = {}
        for i, file in enumerate(files):
            gid = os.path.basename(file).split('.')[0]
            sg_list[gid] = nx.read_gpickle(file)
        return cls(sg_list)

    @classmethod
    def load_json(cls, root_dir):
        files = glob.glob(os.path.join(root_dir, '*.json'))
        sg_list = {}
        for i, file in enumerate(files):
            gid = os.path.basename(file).split('.')[0]
            js_graph = json.load(open(file,'r'))
            sg_list[gid] = json_graph.node_link_graph(js_graph)
        return cls(sg_list)

    def dump(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for gid, sg in self.sg_list.items():
            output_path = os.path.join(output_dir, f'{gid}.pkl')
            nx.write_gpickle(sg, output_path)

    def dump_json(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for gid, sg in self.sg_list.items():
            output_path = os.path.join(output_dir, f'{gid}.json')
            data = json_graph.node_link_data(sg)
            json.dump(data, open(output_path, 'w'))


if __name__=='__main__':
    import sys
    train_pkl_file = sys.argv[1] # 'infer_outinfer_train37.pkl'
    val_pkl_file = sys.argv[2] # 'infer_outinfer_val37.pkl'
    output_dir = 'scenegraphs/graphs'

    out=SGConstructor.from_dict(graph_file=val_pkl_file)
    out.dump_json(output_dir)

    del out

    out=SGConstructor.from_dict(graph_file=train_pkl_file)
    out.dump_json(output_dir)
