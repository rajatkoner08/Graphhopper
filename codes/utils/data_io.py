import json
import pickle
import h5py
import os
import glob
import shutil
import csv
from networkx.readwrite import json_graph


class DataIO(object):
    def __init__(self, data):
        self.data = data

    @staticmethod
    def is_fileexits(filename):
        if not os.path.isfile(filename):
            raise FileNotFoundError(filename)

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
            data = {k: f[k][:] for k in f.keys()}
            f.close()
            f = data
        return f

    @classmethod
    def dump_hdf5(cls, data, filename):
        f = h5py.File(filename, 'w')
        for k, v in data.items():
            f.create_dataset(k, data=v)
        f.close()

    @classmethod
    def dump_json(cls, data, filename):
        json.dump(data, open(filename, 'w'))

    @classmethod
    def dump_pickle(cls, data, filename):
        with open(filename, 'wb') as f:
            pickle.dump(data, f)


    @classmethod
    def read_json_graph(cls, filename):
        data = cls.from_json(filename)
        return json_graph.node_link_graph(data)

    @classmethod
    def load_list_from_text(cls, filename):
        cls.is_fileexits(filename)
        triples = []
        with open(filename, 'r') as f:
            data = csv.reader(f, delimiter='\t')
            for line in data:
                triples.append(line)
        return triples

    @classmethod
    def dump_list_to_text(cls, data, filename):
        with open(filename, 'w') as f:
            for line in data:
                line = '\t'.join(line)
                f.write(line + '\n')

    @classmethod
    def sample_dict(cls, data, nums=100):
        return {k:data[k] for k in list(data.keys())[:nums]}
    
    @classmethod
    def retrieve_ids(cls, root_dir, endswith='.json'):
        files = glob.glob(root_dir + '/*' + endswith)
        ids = [os.path.basename(x).split('.')[0] for x in files]
        return ids

    @classmethod
    def copy_files(cls, root_dir, ids, endswith, output_dir):
        files = [os.path.join(root_dir, x+endswith) for x in ids]
        assert all(os.path.isfile(x) for x in files), files
        os.makedirs(output_dir, exist_ok=True)
        for file in files:
            src = file
            dst = os.path.join(output_dir, os.path.basename(src))
            shutil.copy(src, dst)

    @classmethod
    def copy_folders(cls, root_dir, ids, output_dir):
        files = [os.path.join(root_dir, x) for x in ids]
        assert any(os.path.isdir(x) for x in files), files
        os.makedirs(output_dir, exist_ok=True)
        for file in files:
            src = file
            dst = os.path.join(output_dir, os.path.basename(src))
            try:
                shutil.copytree(src, dst)
            except:
                continue