import networkx as nx
from networkx.readwrite import json_graph
import glob
from copy import deepcopy
from collections import OrderedDict
from tqdm import tqdm
import shutil
import os
import json
import multiprocessing


class ProcessData():
    def __init__(self):
        self.replace_list = OrderedDict({
            "?": " ?",
            ",": " ,",
            "_": " ",
            "n't": " n't", # aren't, isn't
            "'s": " 's", # what's, bird's
            "s'": "s '", # birds'
            "tshirt": "t-shirt"
        })

    def _canonicalization(self, sentence):
        origin = sentence
        sentence = sentence.lower()
        for token in self.replace_list:
            sentence = sentence.replace(token, self.replace_list[token])

        debug = False
        if debug:
            old = origin
            new = sentence
            for t in [' ?', '?', ' ,', ',', " 's", "'s", "isn't", "is not"]:
                old = old.replace(t, '')
                new = new.replace(t, '')
            if old.lower() != new:
                print(sentence, origin)
        return sentence


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


# GQA DICT -> AGENT INPUT
class InstanceList():
    def __init__(self, sg_list):
        self.sg_list = sg_list

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

    @staticmethod
    def instance_from_dict(graph_dict, attribute_category=None):
        graph_dict = deepcopy(graph_dict)
        if not attribute_category: attribute_category = dict()

        vocab = ObjID()
        for k,v in graph_dict['objects'].items():
            vocab.add(k)

        nodes = []
        adjacency = []
        for k,v in graph_dict['objects'].items():
            nodes.append({'id':vocab[k], 'object':k, 'name':[v['name'],v['name']], 'weight':[0,0]})
            rels = v['relations']
            for rel in rels:
                rel.update({'id': vocab[rel['object']]})
            adjacency.append(rels)

        # attribute
        for k,v in graph_dict['objects'].items():
            for attr in v['attributes']:
                if attr in attribute_category:
                    vocab.add(k+attr)
                    nodes.append({'id':vocab[k+attr], 'name':[attr, attr], 'weight':[1,0]})
                    link = {'id':vocab[k+attr], 'name': 'has '+attribute_category[attr]}
                    # add link : node->attr
                    adjacency[vocab[k]].append(link)
                    # add empty link: attr -> node
                    adjacency.append([])
            # === #

        data = {
            'nodes':nodes,
            'adjacency':adjacency,
        }
        g = json_graph.adjacency_graph(data, multigraph=False, directed=True)

        # node id is contiguous [0, len)
        assert all(x==i for i, (x,y) in enumerate(g.nodes(data=True)))

        # add reverse links
        for r in g.edges():
            if r[::-1] not in g.edges():
                reverse_rel = 'reverse ' + g.edges[r]['name']
                g.add_edge(r[1], r[0], name=reverse_rel, id=r[0])

        return g

    def __len__(self):
        return len(self.sg_list)

    @property
    def vocabs(self):
        all_nodes = set()
        all_edges = set()
        for g in self.sg_list.values():
            all_nodes.update(self._get_entities(g))
            all_edges.update(self._get_edges(g))
        return all_nodes, all_edges

    def _get_entities(self, g):
        return set(entry for line in nx.get_node_attributes(g, 'name').values() for entry in line)

    def _get_edges(self, g):
        return set(entry for line in nx.get_edge_attributes(g, 'name').values() for entry in line)

    @classmethod
    def from_dict(cls, graphs, attrcat=None):
        sg_list = {}
        for gid, graph_dict in graphs.items():
            sg_list[gid] = cls.instance_from_dict(graph_dict, attrcat)
        return cls(sg_list)

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
            try:
                gid = os.path.basename(file).split('.')[0]
                js_graph = json.load(open(file,'r'))
                sg_list[gid] = json_graph.node_link_graph(js_graph)
            except:
                print(file)
        return cls(sg_list)

    @staticmethod
    def process(file):
        gid = os.path.basename(file).split('.')[0]
        js_graph = json.load(open(file, 'r'))
        g = json_graph.node_link_graph(js_graph)

        identifier = -1

        nodes = set(entry for line in nx.get_node_attributes(g, 'name').values() for entry in line)

        edges = set(entry for line in nx.get_edge_attributes(g, 'name').values() for entry in line)

        if nx.number_of_edges(g)>=1 and nx.number_of_nodes(g)>=1:
            identifier = gid

        del js_graph
        del g

        return (gid, identifier, nodes, edges) # identifier for filtering graphs || nodes/edges for vocab || nodes for answer set

    @staticmethod
    def postprocess(ret, dict_dir):
        """(gid, identifier, nodes, edges)"""
        print('total graphs', len(ret))
        # split index
        split_index = set(x[1] for x in ret)
        official_split_index = json.load(open(os.path.join(dict_dir, 'official_split_index.json')))
        new = {k: v for k, v in official_split_index.items() if k in split_index}
        print('number of filtered graphs', len(new))
        json.dump(new, open('split_index.json', 'w'))

        # embedding vocab
        all_nodes = set(i for x in ret for i in x[2])
        all_rels = set(i for x in ret for i in x[3])
        question_vocab = json.load(open(os.path.join(dict_dir, 'question_vocab.json')))

        embedding_vocab = {}
        relation_vocab = {}
        update_vocab(embedding_vocab, 'PAD')
        update_vocab(embedding_vocab, 'UNK')
        update_vocab(embedding_vocab, 'EOS')
        update_vocab(embedding_vocab, 'TRUE')
        update_vocab(embedding_vocab, 'FALSE')

        update_vocab(relation_vocab, 'PAD')
        update_vocab(relation_vocab, 'DUMMY_START_RELATION')
        update_vocab(relation_vocab, 'NO_OP')
        update_vocab(relation_vocab, 'UNK')
        update_vocab(relation_vocab, 'JUDGE')

        for node in all_nodes:
            update_vocab(embedding_vocab, node)

        for node in question_vocab:
            update_vocab(embedding_vocab, node)

        for rel in all_rels:
            update_vocab(relation_vocab, rel)
        print(f'embedding vocab {len(embedding_vocab)}, relation vocab {len(relation_vocab)}')
        json.dump(embedding_vocab, open('embedding_vocab.json', 'w'))
        json.dump(relation_vocab, open('relation_vocab.json', 'w'))

        # available_answers
        available_dict = {x[0]: list(x[2]) for x in ret}
        print('available answers dict', len(available_dict))
        json.dump(available_dict, open('available_answers_dict.json', 'w'))

    @classmethod
    def load_json_stream(cls, root_dir, dict_dir):

        files = glob.glob(os.path.join(root_dir, '*.json'))

        function = cls.process
        iterable = files
        total = len(iterable)
        with multiprocessing.Pool(4) as p:
            ret = list(tqdm(p.imap(function, iterable), total=total))

        cls.postprocess(ret, dict_dir)

        return ret

def update_vocab(vocab_dict, item):
    if item not in vocab_dict:
        vocab_dict[item] = len(vocab_dict)


def copy_question(question_path_dir):
    from collections import defaultdict
    import sys
    from codes.utils.data_io import DataIO

    answer_dict_path = 'available_answers_dict.json'

    availables = DataIO.from_json(answer_dict_path)
    availables = {k:set(v) for k,v in availables.items()}
    print(len(availables))

    # question tiers
    for tier in ['train_balanced', 'val_balanced']:
        questions = DataIO.load_list_from_text(question_path_dir+'/'+tier+'.txt')
        valid_set = set()
        stats=defaultdict(int)
        for line in questions:
            sid = line[0]
            qid = line[1]
            answer = line[3]

            if answer in {'yes','no'}:
                valid_set.add(qid)
                continue
            if sid in availables:
                if answer in availables[sid]:
                    valid_set.add(qid)

        new = []
        for line in questions:
            if line[1] in valid_set:
                new.append(line)

        DataIO.dump_list_to_text(new, tier.split('_')[0]+'.txt')


if __name__=='__main__':
    input_dir = 'graphs'
    dict_dir = '../../../data/dicts/'

    for f in ['official_split_index.json', 'question_vocab.json', 'answer_set.json']:
        assert os.path.isfile(os.path.join(dict_dir, f)), FileNotFoundError(f)

    lists = InstanceList.load_json_stream(input_dir, dict_dir)
    print('number of graphs',len(lists))

    shutil.copy(os.path.join(dict_dir, 'answer_set.json'), 'answer_set.json')

    copy_question(dict_dir)
