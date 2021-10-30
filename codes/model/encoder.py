import array
import os
import six
import torch
from tqdm import tqdm
import sys
import numpy as np
import re


class EmptyLayer:
    def __call__(self, feature, *argv):
        return feature


class StaticTokenEncoder:
    def __init__(self, wv_type='glove.6B', wv_dir="/Users/xxx/Repositories/VQA/wordVector/", wv_dim=50,
                 max_word_length=30):
        self.embedding_size = wv_dim
        self.wv_dim = wv_dim
        self.glove_wv = GloveWordVector(wv_type, wv_dir, wv_dim)
        self.max_word_length = max_word_length

    def embedd_entity(self, entity):
        #dirty fix
        if entity=='tshirt':
            entity = 'shirt'
        if 'has_sportActivity' in entity:
            entity = entity.replace('sportActivity', 'sport activity')
        tokens = [x.lower() for x in re.split('[\'_\- ]', entity)]  # TODO check
        embedding = self.glove_wv.word_embedding(tokens)
        embedding = np.mean(embedding, axis=0)
        return embedding

    def embedd_entity_batch(self, entities):
        """
        input: entities: list of tokens
        """
        embeddings = [self.embedd_entity(x) for x in entities]
        embeddings = np.stack(embeddings, axis=0)
        assert embeddings.shape==(len(entities), self.wv_dim)
        return embeddings

    def padded_embedding(self, entities, max_num_entities):
        embeddings = self.embedd_entity_batch(entities)
        canvas = np.zeros((max_num_entities, self.wv_dim))
        l = embeddings.shape[0]
        if l>max_num_entities:
            canvas[:,:] = embeddings[:max_num_entities,:]
        else:
            canvas[:l,:] = embeddings
        return canvas

    def encode_question(self, token_sequence):
        embedding = self.glove_wv.word_embedding(token_sequence)
        return embedding # [#Words, Embdim]

    def encode_question_batch(self, token_sequences):
        """
        return padded embedding numpy array, mean embedding of the whole sentence
        """
        embedding_list = []
        embedding_mean_list = []
        for token_seq in token_sequences:
            embedding = self.encode_question(token_seq)
            tmp = np.ndarray((self.max_word_length, self.wv_dim))
            tmp[:embedding.shape[0], :] = embedding
            embedding_list.append(tmp)
            embedding_mean_list.append(embedding.mean(axis=0))

        embedding_batch = np.stack(embedding_list, axis=0) #[B, MaxWords, Embdim]
        embedding_mean_batch = np.stack(embedding_mean_list, axis=0)

        assert embedding_batch.shape == (len(token_sequences), self.max_word_length, self.wv_dim)
        assert embedding_mean_batch.shape == (len(token_sequences), self.wv_dim)

        return embedding_batch, embedding_mean_batch


class GloveWordVector:
    def __init__(self, wv_type='glove.6B', wv_dir="data/", wv_dim=300):
        self.wv_type = wv_type
        self.wv_dir = wv_dir
        self.wv_dim = wv_dim
        ret = self.load_word_vectors(wv_dir, wv_type, wv_dim)
        self.wv_dict, self.wv_arr, self.wv_size = ret

    def word_embedding(self, token_sequence):
        vectors = torch.Tensor(len(token_sequence), self.wv_dim)
        vectors.normal_(0,1)

        for i, token in enumerate(token_sequence):
            wv_index = self.wv_dict.get(token, None)
            if wv_index is not None:
                vectors[i] = self.wv_arr[wv_index]
            else:
                # Try the longest word (hopefully won't be a preposition
                lw_token = sorted(token.split(' '), key=lambda x: len(x), reverse=True)[0]
                print("{} -> {} ".format(token, lw_token))
                wv_index = self.wv_dict.get(lw_token, None)
                if wv_index is not None:
                    vectors[i] = self.wv_arr[wv_index]
                else:
                    print("fail on {}".format(token))
                    self.add_temporary_wv(token, vectors[i])

        return vectors.data.numpy()

    def add_temporary_wv(self, token, value):
        wv_length = self.wv_arr.shape[0]
        self.wv_arr = torch.cat([self.wv_arr, torch.unsqueeze(value, dim=0)], dim=0)
        self.wv_dict[token] = wv_length

    def load_word_vectors(self, root, wv_type, dim):
        """Load word vectors from a path, trying .pt, .txt, and .zip extensions."""
        if isinstance(dim, int):
            dim = str(dim) + 'd'
        fname = os.path.join(root, wv_type + '.' + dim)
        if os.path.isfile(fname + '.pt'):
            fname_pt = fname + '.pt'
            print('loading word vectors from', fname_pt)
            try:
                return torch.load(fname_pt)
            except Exception as e:
                print("""
                    Error loading the model from {}
    
                    This could be because this code was previously run with one
                    PyTorch version to generate cached data and is now being
                    run with another version.
                    You can try to delete the cached files on disk (this file
                      and others) and re-running the code
    
                    Error message:
                    ---------
                    {}
                    """.format(fname_pt, str(e)))
                sys.exit(-1)
        if os.path.isfile(fname + '.txt'):
            fname_txt = fname + '.txt'
            cm = open(fname_txt, 'rb')
            cm = [line for line in cm]
        else:
            raise RuntimeError('unable to load word vectors')

        wv_tokens, wv_arr, wv_size = [], array.array('d'), None
        if cm is not None:
            for line in tqdm(range(len(cm)), desc="loading word vectors from {}".format(fname_txt)):
                entries = cm[line].strip().split(b' ')
                word, entries = entries[0], entries[1:]
                if wv_size is None:
                    wv_size = len(entries)
                try:
                    if isinstance(word, six.binary_type):
                        word = word.decode('utf-8')
                except:
                    print('non-UTF8 token', repr(word), 'ignored')
                    continue
                wv_arr.extend(float(x) for x in entries)
                wv_tokens.append(word)
                print('load twice!')
        wv_dict = {word: i for i, word in enumerate(wv_tokens)}
        wv_arr = torch.Tensor(wv_arr).view(-1, wv_size)
        ret = (wv_dict, wv_arr, wv_size)
        torch.save(ret, fname + '.pt')
        return ret
